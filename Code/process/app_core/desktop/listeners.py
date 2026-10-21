"""The (event, value) listeners every desktop component reports its changes to; the backend (http/backend.py) republishes the
whole state.snapshot on each. A component emits only after releasing its own lock: a listener reads every component
(DesktopState.snapshot) and the bus listeners behind it take the session's _voice_lock."""
from __future__ import annotations

import logging
import threading


class DesktopEvents:
    __slots__ = ('_lock', '_listeners')

    def __init__(self):
        self._lock = threading.Lock()
        self._listeners = []

    def subscribe(self, listener):
        """listener(event, value), run on the emitting thread; returns its unsubscribe."""
        with self._lock: self._listeners.append(listener)
        def unsubscribe():
            with self._lock:
                if listener in self._listeners: self._listeners.remove(listener)
        return unsubscribe

    def emit(self, event, value=None):
        with self._lock: listeners = tuple(self._listeners)
        for listener in listeners:
            try: listener(event, value)
            except Exception: logging.getLogger(__name__).exception('Desktop state listener %r failed on %s', listener, event)
