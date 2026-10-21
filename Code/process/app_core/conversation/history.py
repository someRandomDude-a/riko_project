"""The conversation history the model sees (chat_history.json), with exactly one owner."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from ..kernel.messages import ChatMessage
from ..persistence.atomic import atomic_write
from ..persistence.preserve import preserve_unreadable

logger = logging.getLogger(__name__)
ROLES, SOURCES = {'system', 'user', 'assistant', 'tool'}, {'discord', 'microphone', 'message'}


class ConversationHistory:
    """The only writer of the history and of its file. ChatService commits a finished reply with append(); SessionManager
    rewrites a turn (speech over a reply, a cut, a stale redo) with replace() or rewrite(); everyone else reads a snapshot().

    Every change is one step under the history's lock and is saved before the lock is released, so a reader or the other
    writer never sees half of one. The lock is a component lock: SessionManager calls in while holding _voice_lock (never
    the reverse), and nothing here publishes or calls out. An unreadable file is kept aside (<name>.unreadable-<timestamp>)
    before the first save replaces it, or never saved over; a save that fails is logged, never raised (it follows a reply
    already shown or spoken), and the next change saves the whole history again."""
    def __init__(self, file: Path | None = None):
        self.file = file
        self._lock = threading.RLock()
        self._unreadable = self._unsaved = False
        self._messages: list[ChatMessage] = self._load()

    def __len__(self):
        with self._lock: return len(self._messages)

    def snapshot(self, start=0, end=None):
        """A copy of messages[start:end]: a reader never holds the live list."""
        with self._lock: return self._messages[start:end]

    def append(self, messages):
        """Add messages at the end and save; returns the index of the first."""
        messages = list(messages)
        def change(history):
            history.extend(messages)
            return len(history) - len(messages)
        return self.rewrite(change)

    def replace(self, start, end, messages):
        """Put messages in place of messages[start:end] (end None: to the end) and save; returns the index after them."""
        messages = list(messages)
        def change(history):
            history[start:end] = messages
            return start + len(messages)
        return self.rewrite(change)

    def rewrite(self, change):
        """Run change(history) on a copy of the history, then keep the copy and save it if change altered it (replaced,
        added or removed a message). One step for every reader and writer; returns what change returned. change runs under
        the history's lock, so it must not block, publish or call back into the session. Messages are never edited in
        place (a snapshot shares them): change puts a dataclasses.replace() copy in the list instead."""
        with self._lock:
            history = list(self._messages)
            result = change(history)
            if self._unsaved or len(history) != len(self._messages) or any(new is not old for new, old in zip(history, self._messages)):
                self._messages = history
                self.save()
            return result

    def save(self):
        """Rewrite the whole file; returns whether it now holds the history."""
        with self._lock:
            self._unsaved = not self._write()
            return not self._unsaved

    def _write(self):
        if not self.file: return True
        if self._unreadable and not self._preserve_unreadable(): return False
        try: atomic_write(self.file, json.dumps([m.as_record() for m in self._messages], indent=2))
        except OSError as exc:
            logger.error('Could not save chat history to %s (%s); the next save retries', self.file.name, exc)
            return False
        return True

    def _load(self):
        if not self.file or not self.file.exists(): return []
        try:
            raw = json.loads(self.file.read_text(encoding='utf-8'))
            if not isinstance(raw, list) or any(not isinstance(item, dict) or item.get('role') not in ROLES or not isinstance(item.get('content', ''), str) for item in raw):
                raise ValueError('Invalid history records')
            return [ChatMessage(x['role'], x.get('content', ''), tool_call_id=x.get('tool_call_id'), timestamp=x.get('timestamp'),
                source=x.get('source') if x.get('source') in SOURCES else None,
                conversation_id=str(x['conversation_id'])[:200] if x.get('conversation_id') else None) for x in raw if x.get('role') != 'system']
        except (OSError, ValueError, KeyError) as exc:
            logger.warning('Unable to load chat history (%s); starting with an empty history', exc)
            self._unreadable = True
            self._preserve_unreadable()
            return []

    def _preserve_unreadable(self):
        """Keep an unreadable history aside before any save replaces it. Fails closed."""
        try: backup = preserve_unreadable(self.file)
        except OSError as exc:
            logger.error('Could not back up unreadable %s (%s); chat history is not saved over it', self.file.name, exc)
            return False
        if backup: logger.warning('Kept the unreadable chat history as %s', backup.name)
        self._unreadable = False
        return True
