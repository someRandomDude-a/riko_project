from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RuntimeEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
    turn_id: str | None = None
    action_id: str | None = None

    def as_dict(self): return asdict(self)


class EventBus:
    """Synchronous: listeners run on the publisher's thread. Never publish while holding a
    lock other than SessionManager._voice_lock; queue through events.outbox.Outbox instead."""
    def __init__(self):
        self._lock = threading.RLock()
        self._listeners = []
        self._sequence = 0
    def subscribe(self, listener: Callable[[RuntimeEvent], None]):
        with self._lock: self._listeners.append(listener)
        return lambda: self.unsubscribe(listener)
    def unsubscribe(self, listener):
        with self._lock:
            if listener in self._listeners: self._listeners.remove(listener)
    def cursor(self):
        with self._lock: return self._sequence
    def publish(self, event_type: str, *, turn_id=None, action_id=None, **payload):
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        event = RuntimeEvent(event_type, payload, sequence=sequence, turn_id=turn_id, action_id=action_id)
        with self._lock: listeners = tuple(self._listeners)
        for listener in listeners:
            try: listener(event)
            except Exception: logger.exception('Event listener %r failed on %s', listener, event_type)
        return event


event_bus = EventBus()
