"""Paginated UI history, independent of bounded real-time delivery and model context."""
import json
import sqlite3
import threading
import time
import uuid


class ConversationStore:
    def __init__(self, path, provider='', legacy=()):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, started REAL, ended REAL, provider TEXT, outcome TEXT);
            CREATE TABLE IF NOT EXISTS messages(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, session_id TEXT, data TEXT);
        ''')
        self.db.execute("UPDATE sessions SET ended=?,outcome='recovered_after_restart' WHERE ended IS NULL AND outcome='running'", (time.time(),))
        if not self.db.execute('SELECT 1 FROM messages LIMIT 1').fetchone() and legacy:
            self.db.execute("INSERT OR IGNORE INTO sessions VALUES('legacy',NULL,NULL,'','imported')")
            for index, message in enumerate(legacy):
                if message.role not in {'user', 'assistant'}: continue
                data = {'id': f'legacy:{index}', 'role': message.role, 'text': message.content,
                        'timestamp': message.timestamp, 'session_id': 'legacy'}
                self.db.execute('INSERT INTO messages(id,session_id,data) VALUES(?,?,?)', (data['id'], 'legacy', json.dumps(data)))
        # Crash-interrupted messages remain readable and explicitly incomplete.
        for seq, raw in self.db.execute("SELECT seq,data FROM messages WHERE json_extract(data,'$.status')='running'").fetchall():
            data = json.loads(raw)
            if data.get('status') == 'running':
                data.update(status='abandoned', interrupted=True)
                self.db.execute('UPDATE messages SET data=? WHERE seq=?', (json.dumps(data), seq))
        self.session_id = str(uuid.uuid4())
        self.provider = provider
        self.logical_sessions = {self.session_id}
        self.db.execute("INSERT INTO sessions VALUES(?,?,NULL,?,'running')", (self.session_id, time.time(), provider))
        self.db.commit()
        self.closed = False
        self.cache = {}
        self.pending = {}
        self.stopped = threading.Event()
        def persist():
            while not self.stopped.wait(.25):
                with self.lock:
                    if not self.closed: self._flush()
        threading.Thread(target=persist, daemon=True, name='conversation-archive').start()

    def _flush(self):
        if not self.pending: return
        for message_id, data in self.pending.items():
            self.db.execute('INSERT INTO messages(id,session_id,data) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data',
                            (message_id, data['session_id'], json.dumps(data, ensure_ascii=False)))
        self.db.commit()
        self.pending.clear()

    def observe(self, event):
        if event.type not in {'chat.input', 'model.started', 'model.metrics', 'chat.delta', 'chat.completed', 'chat.interrupted', 'chat.cancelled', 'chat.interjection', 'model.error'} or not event.turn_id: return
        with self.lock:
            if self.closed: return
            user = event.type == 'chat.input'
            message_id = event.turn_id + (':user' if user else '')
            data = self.cache.get(message_id)
            row = self.db.execute('SELECT data FROM messages WHERE id=?', (message_id,)).fetchone() if data is None else None
            data = data if data is not None else json.loads(row[0]) if row else {'id': message_id, 'role': 'user' if user else 'assistant', 'text': '',
                'timestamp': event.timestamp, 'session_id': self.session_id, 'status': 'running'}
            p = event.payload
            if event.type == 'model.metrics': data['metrics'] = dict(p)
            if p.get('source') in {'discord','microphone','message'}:
                data['source'] = p['source']
                for key in ('conversation_id', 'user_id', 'channel_id', 'guild_id'):
                    if key in p: data[key] = p[key]
                if p['source'] == 'discord' and p.get('conversation_id'):
                    sid = p['conversation_id']
                    if sid not in self.logical_sessions:
                        self.db.execute("INSERT OR IGNORE INTO sessions VALUES(?,?,NULL,?,'running')", (sid, event.timestamp, self.provider))
                        self.logical_sessions.add(sid)
                    data['session_id'] = sid
            data['event_sequence'] = event.sequence
            if 'initiative' in p: data.update(initiative=p['initiative'], spoken=p.get('spoken', False))
            if 'user_name' in p: data['user_name'] = p['user_name']
            if event.type == 'chat.delta': data['text'] += p.get('text', '')
            elif event.type in {'chat.input', 'chat.completed'}: data.update(text=p.get('text', ''), status='completed')
            elif event.type == 'chat.interrupted': data.update(interrupted=True, cutoff=p.get('offset'), text=p.get('text', data['text']))
            elif event.type in {'chat.cancelled', 'model.error'}: data.update(interrupted=True, status='cancelled' if event.type == 'chat.cancelled' else 'error', error=p.get('error',''))
            elif event.type == 'chat.interjection':
                items = data.setdefault('interjections', [])
                last = items[-1] if items else None
                if last and 0 <= p['started_at'] - last['ended_at'] <= p.get('debounce_seconds', 1):
                    display = last.get('display_text',last['text'].removeprefix('[speaking over you] '))
                    last.update(text=last['text'] + ' ' + p['text'].removeprefix('[speaking over you] '), ended_at=p['ended_at'])
                    if 'display_text' in p:last.update(display_text=display+' '+p['display_text'],system_label=p.get('system_label'))
                else: items.append(dict(p))
            self.cache[message_id] = data
            self.pending[message_id] = data
            if event.type not in {'chat.delta','model.metrics'}: self._flush()
            if len(self.cache) > 200:
                self.cache.pop(next(iter(self.cache)))

    def page(self, before=None, limit=40):
        with self.lock:
            self._flush()
            rows = self.db.execute('SELECT seq,data,session_id FROM messages WHERE seq<? ORDER BY seq DESC LIMIT ?',
                (before if before is not None else 2**63-1, limit)).fetchall()
            cursor = rows[-1][0] if rows else before
            sessions = {sid: dict(zip(('id', 'started', 'ended', 'provider', 'outcome'),
                 self.db.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone())) for _, _, sid in rows}
            for sid, record in sessions.items(): record['source'] = 'discord' if sid.startswith('discord:') else 'desktop'
            return {'messages': [{**json.loads(raw), 'sequence': seq} for seq, raw, _ in reversed(rows)],
                    'sessions': sessions, 'before': cursor,
                    'has_more': bool(rows and self.db.execute('SELECT 1 FROM messages WHERE seq<? LIMIT 1', (cursor,)).fetchone())}

    def close(self):
        with self.lock:
            if self.closed: return
            self.stopped.set()
            self._flush()
            self.db.executemany("UPDATE sessions SET ended=?,outcome='stopped' WHERE id=?", [(time.time(), sid) for sid in self.logical_sessions])
            self.db.commit(); self.closed = True; self.db.close()
