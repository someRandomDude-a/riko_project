"""Per-call approval gate; permissions never come from model output."""
from copy import deepcopy
from contextvars import ContextVar
import json
import threading
import time
import uuid
from ..events.bus import event_bus

approval_turn = ContextVar('approval_turn', default=None)


class ToolApprovals:
    def __init__(self, path, default=False):
        self.path, self.default = path, default
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.pending, self.policy = {}, {}
        self.closed = False
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            self.policy = {k: v for k, v in value.items() if isinstance(k, str) and type(v) is bool}
        except (OSError, ValueError, AttributeError): pass

    def snapshot(self):
        with self.lock:
            return {'default_required': self.default, 'policy': dict(self.policy),
                'pending': [deepcopy(v['request']) for v in self.pending.values() if v['decision'] is None]}

    def configure(self, policy, names):
        if not isinstance(policy, dict) or any(k not in names or type(v) is not bool for k, v in policy.items()):
            raise ValueError('Approval policy must map registered tool names to booleans')
        with self.lock:
            updated = {**self.policy, **policy}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(updated, indent=2), encoding='utf-8')
            temporary.replace(self.path)
            self.policy = updated
        event_bus.publish('tool.approval_policy')
        return self.snapshot()

    def resolve(self, request_id, approved):
        if type(approved) is not bool: raise ValueError('approved must be boolean')
        with self.lock:
            entry = self.pending.get(request_id)
            if not entry or entry['decision'] is not None or time.time() >= entry['request']['expires_at']:
                raise ValueError('Approval request expired or already resolved')
            entry['decision'] = approved
            self.condition.notify_all()
        event_bus.publish('tool.approval_resolved', id=request_id, approved=approved)

    def authorize(self, name, arguments, call_id, cancelled=lambda: False, timeout=120):
        with self.lock:
            if self.closed or cancelled(): return False
            if not self.policy.get(name, self.default): return True
            request_id = str(uuid.uuid4())
            request = {'id': request_id, 'name': name, 'arguments': deepcopy(arguments),
                'call_id': call_id, 'expires_at': time.time() + timeout}
            if approval_turn.get(): request['turn_id'] = approval_turn.get()
            entry = {'request': request, 'decision': None}
            self.pending[request_id] = entry
        event_bus.publish('tool.approval_requested', **request)
        deadline = time.monotonic() + timeout
        def wake(event):
            if event.type in {'turn.cancel_requested','chat.interrupted','voice.started','voice.activated','voice.wake_status','state.snapshot','initiative.settings','runtime.stopped'}:
                with self.condition: self.condition.notify_all()
        unsubscribe = event_bus.subscribe(wake)
        try:
            with self.condition:
                while time.monotonic() < deadline:
                    if self.closed or cancelled(): return False
                    if entry['decision'] is not None: return entry['decision']
                    self.condition.wait(max(0, deadline - time.monotonic()))
            return False
        finally:
            unsubscribe()
            with self.lock: self.pending.pop(request_id, None)
            event_bus.publish('tool.approval_finished', id=request_id)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
