"""What happened recently, for the control window: tool calls (ToolRegistry reports them here: tools/tool.ToolActivity),
avatar actions, notifications, incoming messages and the Discord client's status. Each list keeps the newest 20."""
from __future__ import annotations

from copy import deepcopy
import threading
import time
import uuid


class ActivityLog:
    """Emits 'tool', 'actions', 'notification', 'input_received' and 'discord'."""
    __slots__ = ('events', '_lock', '_tools', '_actions', '_notifications', '_incoming', '_discord')

    def __init__(self, events):
        self.events, self._lock = events, threading.Lock()
        self._tools, self._actions, self._notifications, self._incoming = [], [], [], []
        self._discord = {'running': False, 'ready': False, 'status': 'stopped'}

    def snapshot(self):
        with self._lock:
            return {"tools": [dict(item) for item in self._tools], "actions": [dict(item) for item in self._actions],
                    'notifications': deepcopy(self._notifications), 'incoming': deepcopy(self._incoming), 'discord': deepcopy(self._discord)}

    def tool_started(self, name: str, arguments: dict):
        item = {"id": str(uuid.uuid4()), "name": name, "arguments": arguments, "status": "running", 'started_at': time.time()}
        with self._lock: self._tools = [item, *self._tools[:19]]
        self.events.emit("tool", dict(item))
        return item['id']

    def tool_finished(self, name: str, result, error: bool = False, activity_id=None):
        with self._lock:
            for item in self._tools:
                if item["name"] == name and item["status"] == "running" and (activity_id is None or item['id'] == activity_id):
                    item["status"] = "error" if error else "complete"; item["result"] = str(result)[:8000]
                    item['result_truncated'] = len(str(result)) > 8000
                    item['finished_at'] = time.time()
                    item['duration_ms'] = round((item['finished_at'] - item['started_at']) * 1000)
                    break
            tools = [dict(item) for item in self._tools]
        self.events.emit("tool", tools)

    def record_action(self, action):
        """An avatar action's latest state (the bus's action.* events), replacing its earlier entry."""
        with self._lock:
            self._actions = [action, *[item for item in self._actions if item.get("id") != action.get("id")][:19]]
            actions = [dict(item) for item in self._actions]
        self.events.emit("actions", actions)

    def notify(self, source, text, level='info'):
        item = {'id': str(uuid.uuid4()), 'source': source, 'text': str(text)[:500], 'level': level, 'timestamp': time.time()}
        with self._lock: self._notifications = [item, *self._notifications[:19]]
        self.events.emit('notification', dict(item))
        return item

    def observe_input(self, source, text, *, message_id, context=None):
        if source not in {'discord', 'microphone', 'message'}: raise ValueError('Unknown input source')
        item = {'id': f'{source}:{message_id}', 'source': source, 'text': str(text)[:1000], 'timestamp': time.time(), **(context or {})}
        with self._lock:
            if any(entry['id'] == item['id'] for entry in self._incoming): return
            self._incoming = [item, *self._incoming[:19]]
        self.events.emit('input_received', deepcopy(item))

    def set_discord(self, value):
        with self._lock: self._discord = deepcopy(value)
        self.events.emit('discord', deepcopy(value))
