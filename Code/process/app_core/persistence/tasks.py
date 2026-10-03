"""Transactional task records shared by the runtime and the task MCP server."""
import json
import math
from pathlib import Path
import re
import sqlite3
import uuid
from datetime import datetime


TASK_RULES = '''Task MCP rules:
- Use task_create/task_list/task_get/task_update for ongoing goals, progress and blockers.
- Task records are not automatically supplied. Fetch them with task_list/task_get only when relevant to the current conversation; do not routinely fetch the whole list every turn.
- Use a relevance query and a small limit for focused lookups. Surface a concise task list in conversation when it helps the user, not as recurring context clutter.
- Track explicit user goals or clearly agreed work, not every casual mention. Ask when intent is unclear.
- List/retrieve existing tasks before creating or updating; avoid duplicate goals.
- Separate planned actions from confirmed results. Never mark progress or completion merely because you promised to do something.
- Record blockers, next steps and the evidence/reason for updates. Do not invent deadlines, success or user commitments.
- Read the current revision before task_update; on a revision conflict re-read rather than overwriting a correction.
- Respect paused/dismissed tasks: do not keep urging the user about them. Completed tasks are history, not pending work.
- The user can correct, pause, complete or dismiss any task. Prefer these durable MCP records over the legacy todo_list tool.
'''

STATUSES = {'active', 'blocked', 'paused', 'completed', 'dismissed'}


class TaskConflict(ValueError):
    pass


class TaskStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = self.connect()
        try:
            with db:
                db.execute('PRAGMA journal_mode=WAL')
                db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, record TEXT NOT NULL)')
                db.execute('CREATE TABLE IF NOT EXISTS task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, record TEXT NOT NULL)')
        finally: db.close()

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def timestamp(): return datetime.now().astimezone().isoformat()

    def change_version(self):
        db = self.connect()
        try: return db.execute('SELECT COALESCE(MAX(id),0) FROM task_events').fetchone()[0]
        finally: db.close()

    def get(self, task_id, *, history=True):
        db = self.connect()
        try:
            db.execute('BEGIN')
            row = db.execute('SELECT record FROM tasks WHERE id=?', (task_id,)).fetchone()
            if row is None: raise KeyError(task_id)
            record = json.loads(row['record'])
            if history:
                record['history'] = [json.loads(event['record']) for event in db.execute('SELECT record FROM task_events WHERE task_id=? ORDER BY id', (task_id,))]
            return record
        finally: db.close()

    def list(self, *, include_closed=False, query='', limit=8, statuses=None):
        if not isinstance(limit, int) or not 1 <= limit <= 100: raise ValueError('Limit must be 1–100')
        if not isinstance(include_closed, bool) or not isinstance(query, str): raise ValueError('Invalid task filter')
        db = self.connect()
        try: records = [json.loads(row['record']) for row in db.execute('SELECT record FROM tasks')]
        finally: db.close()
        if not include_closed: records = [r for r in records if r['status'] not in {'completed', 'dismissed'}]
        if statuses is not None: records = [r for r in records if r['status'] in statuses]
        terms = set(re.findall(r'\w+', query.casefold()))
        def relevance(record):
            text = ' '.join(str(record[key]) for key in ('title', 'description', 'next_step', 'blocker'))
            return len(terms & set(re.findall(r'\w+', text.casefold())))
        if terms: records = [r for r in records if relevance(r) > 0]
        return sorted(records, key=lambda r: (relevance(r), r['updated_at']), reverse=True)[:limit]

    def context(self, query=''):
        records = self.list(query=query, limit=8, statuses={'active', 'blocked'})
        return [{**r, **{key: ' '.join(r[key].split()[:80]) for key in ('description', 'next_step', 'blocker')}} for r in records]

    def _validate(self, changes):
        allowed = {'title', 'description', 'status', 'progress', 'next_step', 'blocker'}
        if not isinstance(changes, dict) or not changes or set(changes) - allowed: raise ValueError('Invalid task fields')
        for key in ('title', 'description', 'next_step', 'blocker'):
            if key in changes and (not isinstance(changes[key], str) or len(changes[key]) > (300 if key == 'title' else 4000)):
                raise ValueError(f'Invalid {key}')
        if 'title' in changes and not changes['title'].strip(): raise ValueError('A task title is required')
        if 'status' in changes and changes['status'] not in STATUSES: raise ValueError('Invalid task status')
        if 'progress' in changes:
            value = changes['progress']
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1: raise ValueError('Progress must be 0–1')

    def _event(self, db, record, *, actor, reason, source_turn_id, kind):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000: raise ValueError('A nonempty update reason is required (max 4000 characters)')
        event = {'kind': kind, 'timestamp': self.timestamp(), 'actor': actor, 'reason': reason,
                 'source_turn_id': source_turn_id, 'revision': record['revision'], 'snapshot': record}
        db.execute('INSERT INTO task_events(task_id,record) VALUES (?,?)', (record['id'], json.dumps(event)))

    def create(self, title, *, description='', next_step='', actor='user', reason='Created task', source_turn_id=None):
        self._validate({'title': title, 'description': description, 'next_step': next_step})
        record = {'id': str(uuid.uuid4()), 'title': title.strip(), 'description': description, 'status': 'active',
                  'progress': 0.0, 'next_step': next_step, 'blocker': '', 'revision': 1,
                  'created_at': self.timestamp(), 'updated_at': self.timestamp()}
        db = self.connect()
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                # Exact active-title deduplication protects against repeated model calls.
                for row in db.execute('SELECT record FROM tasks'):
                    existing = json.loads(row['record'])
                    if existing['status'] not in {'completed', 'dismissed'} and ' '.join(existing['title'].casefold().split()) == ' '.join(title.casefold().split()):
                        return existing
                db.execute('INSERT INTO tasks VALUES (?,?,?)', (record['id'], 1, json.dumps(record)))
                self._event(db, record, actor=actor, reason=reason, source_turn_id=source_turn_id, kind='created')
        finally: db.close()
        self._changed(record)
        return record

    def update(self, task_id, expected_revision, changes, *, actor='user', reason='Updated task', source_turn_id=None):
        self._validate(changes)
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 1: raise ValueError('A current revision is required')
        db = self.connect()
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT record FROM tasks WHERE id=?', (task_id,)).fetchone()
                if row is None: raise KeyError(task_id)
                record = json.loads(row['record'])
                if record['revision'] != expected_revision: raise TaskConflict('Task changed; retrieve its current revision before updating')
                previous_status = record['status']
                record.update(changes)
                if record['status'] == 'completed': record['progress'] = 1.0
                elif previous_status == 'completed' and 'progress' not in changes: record['progress'] = 0.0
                if record['status'] != 'completed' and record['progress'] == 1: raise ValueError('Only completed tasks can have progress 1')
                if record['status'] == 'blocked' and not record['blocker'].strip(): raise ValueError('Blocked tasks require a blocker')
                if previous_status == 'blocked' and record['status'] == 'active' and 'blocker' not in changes: record['blocker'] = ''
                record['revision'] += 1
                record['updated_at'] = self.timestamp()
                db.execute('UPDATE tasks SET revision=?,record=? WHERE id=?', (record['revision'], json.dumps(record), task_id))
                self._event(db, record, actor=actor, reason=reason, source_turn_id=source_turn_id, kind='updated')
        finally: db.close()
        self._changed(record)
        return record

    @staticmethod
    def _changed(record):
        from ..events.bus import event_bus
        event_bus.publish('task.changed', task=record)


class TaskMCP:
    """Same MCP tools/call interface in-process or through the stdio entry point."""
    def __init__(self, store, source_turn=lambda: None, actor='model_mcp'):
        self.store, self.source_turn, self.actor = store, source_turn, actor

    def list_tools(self):
        text = {'type': 'string'}
        schemas = {
            'task_list': ({'include_closed': {'type': 'boolean'}, 'query': text, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100}}, []),
            'task_get': ({'task_id': text}, ['task_id']),
            'task_create': ({'title': text, 'description': text, 'next_step': text, 'reason': text}, ['title']),
            'task_update': ({'task_id': text, 'expected_revision': {'type': 'integer'}, 'reason': text,
                            'changes': {'type': 'object', 'additionalProperties': False, 'properties': {
                                'title': text, 'description': text, 'next_step': text, 'blocker': text,
                                'status': {'type': 'string', 'enum': sorted(STATUSES)}, 'progress': {'type': 'number', 'minimum': 0, 'maximum': 1}}}},
                            ['task_id', 'expected_revision', 'changes', 'reason'])}
        descriptions = {'task_list': 'Fetch tasks only when relevant. A nonempty query returns lexical matches only; use a small limit. Include closed tasks for completed/dismissed history.',
                        'task_get': 'Retrieve a task with full change history and provenance.',
                        'task_create': 'Track an explicit or agreed ongoing user goal; avoid casual mentions and duplicate tasks.',
                        'task_update': 'Update confirmed task progress, blockers, next steps or status. Requires current revision and evidence/reason; stale revisions are rejected.'}
        return [{'name': name, 'description': descriptions[name], 'inputSchema': {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}}
                for name, (properties, required) in schemas.items()]

    def call(self, name, arguments):
        try:
            definition = next((tool for tool in self.list_tools() if tool['name'] == name), None)
            if definition is None: raise ValueError('Unknown task tool')
            schema = definition['inputSchema']
            if not isinstance(arguments, dict) or set(arguments) - set(schema['properties']) or set(schema['required']) - set(arguments):
                raise ValueError('Invalid or missing task tool arguments')
            if name == 'task_list': result = {'tasks': self.store.list(**arguments)}
            elif name == 'task_get': result = self.store.get(**arguments)
            elif name in {'task_create', 'task_update'}:
                method = self.store.create if name == 'task_create' else self.store.update
                result = method(**arguments, actor=self.actor, source_turn_id=self.source_turn())
            else: raise ValueError('Unknown task tool')
            return {'content': [{'type': 'text', 'text': json.dumps(result, ensure_ascii=False)}], 'structuredContent': result, 'isError': False}
        except (ValueError, TypeError, KeyError) as exc:
            return {'content': [{'type': 'text', 'text': str(exc)}], 'isError': True}

    def close(self): pass
