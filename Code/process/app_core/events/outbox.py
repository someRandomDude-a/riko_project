"""Ordered delivery of events and callbacks without holding the caller's locks.

Bus listeners run on the publisher's thread and may take SessionManager._voice_lock,
so publishing while holding any other lock can deadlock. Queue work with put()/defer()
while holding your lock, then call flush() after releasing it. One thread drains at a
time, so items keep the order in which they were queued.
"""
from collections import deque
import threading


class Outbox:
    def __init__(self, bus):
        self.bus = bus
        self._pending = deque()
        self._lock = threading.Lock()
        self._draining = False

    def put(self, event_type, **payload):
        self._pending.append((self.bus.publish, (event_type,), payload))

    def defer(self, function, *args, **kwargs):
        self._pending.append((function, args, kwargs))

    def flush(self):
        while True:
            with self._lock:
                if self._draining or not self._pending: return
                self._draining = True
            try:
                while self._pending:
                    function, args, kwargs = self._pending.popleft()
                    function(*args, **kwargs)
            finally:
                with self._lock: self._draining = False
