"""Shared, cancellable companion actions.

The controller owns intent and lifecycle. The Electron renderer owns animation
frames and interpolates the expression values it receives.
"""
from __future__ import annotations

import threading
import math
from copy import deepcopy
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..events.bus import event_bus
from ..events.outbox import Outbox


@dataclass
class CompanionAction:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "running"
    _timer: threading.Timer | None = field(default=None, repr=False, compare=False)

    def as_dict(self):
        return {"id": self.id, "kind": self.kind, "payload": deepcopy(self.payload),
                "duration": self.duration, "status": self.status}


class ActionController:
    def __init__(self):
        self._lock = threading.RLock()
        self._actions: dict[str, CompanionAction] = {}
        self._closed = False
        self._outbox = Outbox(event_bus)

    def start(self, kind: str, payload: dict[str, Any] | None = None, duration: float = 0.0) -> CompanionAction:
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Action duration must be finite and non-negative")
        if not kind.strip():
            raise ValueError("Action kind is required")
        action = CompanionAction(kind, deepcopy(payload or {}), duration)
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Action controller is closed")
                # Each kind owns one presentation lane; newer intent replaces old.
                for previous in list(self._actions.values()):
                    if previous.kind == kind:
                        self._cancel_locked(previous.id)
                self._actions[action.id] = action
                if action.duration:
                    action._timer = threading.Timer(action.duration, self.complete, args=(action.id,))
                    action._timer.daemon = True
                self._outbox.put("action.started", action_id=action.id, action=action.as_dict())
                if action._timer and action.status == "running":
                    action._timer.start()
        finally: self._outbox.flush()
        return action

    def complete(self, action_id: str):
        with self._lock:
            action = self._actions.get(action_id)
            if not action or action.status != "running": return False
            action.status = "complete"
            if action._timer: action._timer.cancel()
            del self._actions[action_id]
            self._outbox.put("action.completed", action_id=action.id, action=action.as_dict())
        self._outbox.flush()
        return True

    def update(self, action_id: str, payload: dict):
        """Change continuous targets without restarting the presentation lane."""
        with self._lock:
            action = self._actions.get(action_id)
            if not action or action.status != 'running': return False
            action.payload = deepcopy(payload)
            self._outbox.put('action.updated', action_id=action.id, action=action.as_dict())
        self._outbox.flush()
        return True

    def cancel(self, action_id: str):
        with self._lock: cancelled = self._cancel_locked(action_id)
        self._outbox.flush()
        return cancelled

    def _cancel_locked(self, action_id):
        action = self._actions.get(action_id)
        if not action or action.status != "running": return False
        action.status = "cancelled"
        if action._timer: action._timer.cancel()
        del self._actions[action_id]
        self._outbox.put("action.cancelled", action_id=action.id, action=action.as_dict())
        return True

    def set_emotion(self, state):
        return self.start("emotion", state.as_dict(), duration=0)

    def gesture(self, name: str, intensity: float = 0.65, duration: float = 2.0):
        if name not in {"nod", "shake", "wave"}:
            raise ValueError("Gesture must be nod, shake or wave")
        if not math.isfinite(intensity) or not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between zero and one")
        if not math.isfinite(duration) or not 0.2 <= duration <= 30:
            raise ValueError("Gesture duration must be between 0.2 and 30 seconds")
        return self.start("gesture", {"name": name, "intensity": intensity}, duration)

    def active(self):
        with self._lock:
            return [action.as_dict() for action in self._actions.values() if action.status == "running"]

    def close(self):
        with self._lock:
            self._closed = True
            for action_id in list(self._actions): self._cancel_locked(action_id)
        self._outbox.flush()
