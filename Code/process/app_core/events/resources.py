"""Snapshots originate at mutation events, never periodic status requests."""
import json
import logging
import threading

from ..kernel.workers import DaemonExecutor
from .outbox import Outbox

logger = logging.getLogger(__name__)


class ResourceUnavailable(LookupError):
    """A getter's feature is off or not built (tool approvals, the animation service): panels keep the previous value, and the
    HTTP layer answers 503 with this message (app_core/http/app.py)."""


class ResourceEvents:
    def __init__(self, bus, getters):
        """getters: {topic: domain getter} (http/backend.Backend.resource_getters), never HTTP handlers. They run on this
        object's worker, so a publisher, which may hold a component lock (WakeWord publishes under its own), never runs one."""
        self.bus, self.getters = bus, getters
        self.lock = threading.RLock()
        self.previous = {}
        self.generation = {}
        self.pending = set()
        self.closed = False
        self.outbox = Outbox(bus)
        self.worker = DaemonExecutor(1, 'resource-events', max_pending=len(getters) + 16)
        self.unsubscribe = bus.subscribe(self.observe)

    def snapshot(self):
        result = {}
        for topic, getter in self.getters.items():
            try: result[topic] = getter()
            except Exception: # Runtime may be starting/stopping.
                logger.debug('Resource %s unavailable for snapshot', topic, exc_info=True)
                result[topic] = None
        return result

    def emit(self, topic):
        # Getters may take session locks, so they run unlocked. A newer emit of the same
        # topic supersedes an older one still computing, and the outbox publishes in queue
        # order after the lock is released, so subscribers never see a stale value last.
        with self.lock:
            if self.closed: return
            generation = self.generation[topic] = self.generation.get(topic, 0) + 1
        try: value = self.getters[topic]()
        except ResourceUnavailable:
            logger.debug('Resource %s is unavailable; panels keep the previous value', topic, exc_info=True)
            return
        except Exception:
            logger.warning('Resource %s could not be read; panels keep the previous value', topic, exc_info=True)
            return
        key = json.dumps(value, sort_keys=True, default=str)
        with self.lock:
            if not self.closed and generation == self.generation[topic] and self.previous.get(topic) != key:
                self.previous[topic] = key
                self.outbox.put('resource.' + topic, **value)
        self.outbox.flush()

    def schedule(self, topic):
        """emit(topic) on the worker. A topic already waiting there is not queued twice: the waiting emit reads the newest
        value, so the queue holds at most one job per topic."""
        with self.lock:
            if self.closed or topic in self.pending or topic not in self.getters: return
            self.pending.add(topic)
        try: self.worker.submit(self._run, topic)
        except RuntimeError:  # closed meanwhile
            with self.lock: self.pending.discard(topic)

    def _run(self, topic):
        with self.lock: self.pending.discard(topic)  # an event from now on queues another emit, which sees its change
        self.emit(topic)

    def settle(self, timeout=5):
        """Wait until every emit scheduled so far has published (tests, and anything that must see the result)."""
        self.worker.submit(lambda: None).result(timeout)

    def observe(self, event):
        kind = event.type
        if kind.startswith('resource.'): return
        if kind.startswith('tool.approval'): self.schedule('approvals')
        elif kind.startswith(('initiative.', 'environment.')): self.schedule('initiative')
        elif kind.startswith('animation.'): self.schedule('animation')
        elif kind == 'avatar.models_changed': self.schedule('avatar_models')
        elif kind == 'task.changed': self.schedule('tasks')
        elif kind in {'voice.starting','voice.ready','voice.stopped','voice.error','voice.wake_status','voice.activated','voice.follow_up','voice.started','voice.resumed','voice.utterance_ended','voice.transcribing','voice.transcript','voice.waiting'}: self.schedule('voice')
        elif kind == 'runtime.ready':
            for topic in self.getters: self.schedule(topic)

    def close(self):
        self.unsubscribe()
        with self.lock: self.closed = True
        self.worker.shutdown()  # daemon thread: an emit still reading a closing session publishes nothing
