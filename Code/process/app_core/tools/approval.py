"""Per-call approval gate; permissions never come from model output. A rule belongs to one tool of one source, keyed
(source, name) like Tool.approval_key, so a configured MCP server's tool never inherits a rule set for one of Riko's own
however it is named. tool_approvals.json is {"version": 2, "sources": {source: {name: bool}}, "legacy": {name: bool}},
true meaning ask before each call; a file from before sources ({name: bool}) loads as legacy and is rewritten on the
next save. The file also keeps a top-level {name: bool} mirror, ignored here: builds before sources read only that, so a
downgrade still asks before any tool that some source asks before (it fails closed, never open). A tool without a rule
follows its default: tools.require_approval for Riko's own, and for a configured MCP server's "ask first" unless that
server's mcp.json entry says "require_approval": false (server_defaults)."""
from collections.abc import Mapping
from copy import deepcopy
from contextvars import ContextVar
import json
import logging
import threading
import time
import uuid
from ..events.bus import event_bus
from ..persistence.atomic import atomic_write
from ..persistence.preserve import preserve_unreadable
from .tool import RIKO

approval_turn = ContextVar('approval_turn', default=None)
logger = logging.getLogger(__name__)
VERSION = 2


def approval_keys(tools):
    """{registered name: (source, name)} from ToolRegistry.tools (each Tool's approval_key), such a mapping, or plain
    names, which are Riko's own tools."""
    if isinstance(tools, Mapping): return {name: key if isinstance(key, tuple) else key.approval_key for name, key in list(tools.items())}
    return {name: (RIKO, name) for name in tools}


def flags(rules): return {name: rule for name, rule in rules.items() if type(rule) is bool}  # other values are ignored, as before sources


class ToolApprovals:
    def __init__(self, path, default=False, server_defaults=None):
        self.path, self.default, self.server_defaults = path, default, dict(server_defaults or {})  # {mcp:<name>: bool}
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.pending, self.policy, self.legacy = {}, {}, {}
        self.closed = False
        self.error = ''
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(value, dict): raise ValueError('Approval policy is not a JSON object')
            if type(value.get('version')) is not int: self.legacy = flags(value)  # saved before rules had sources
            elif value['version'] != VERSION: raise ValueError(f'Approval policy version {value["version"]} is not supported')
            else:
                sources, legacy = value.get('sources', {}), value.get('legacy', {})
                if not isinstance(sources, dict) or not isinstance(legacy, dict) or not all(isinstance(rules, dict) for rules in sources.values()):
                    raise ValueError('Approval policy sources are not JSON objects')
                self.policy = {(source, name): rule for source, rules in sources.items() for name, rule in flags(rules).items()}
                self.legacy = flags(legacy)
        except FileNotFoundError: pass
        except (OSError, ValueError) as exc:
            # Fail closed: an unreadable policy must not silently switch every approval off.
            self.error = f'{path.name} could not be read ({exc}); every tool needs approval until the policy is saved again'
            logger.error('Tool approval policy unreadable: %s', self.error)

    def rule(self, key):
        """The rule set for the tool (source, name), or None when the default applies. A legacy rule (saved before rules
        had sources, by name alone) still applies to Riko's own tool of that name; for a server's tool only a legacy
        'ask first' applies, so a server never inherits a built-in's free pass and a stricter rule is never lost."""
        with self.lock:
            if key in self.policy: return self.policy[key]
            legacy = self.legacy.get(key[1])
            return legacy if key[0] == RIKO or legacy is True else None

    def fallback(self, key):
        """Whether the tool (source, name) asks first without a rule: a configured server's does unless mcp.json says not."""
        return self.server_defaults.get(key[0], True) if key[0].startswith('mcp:') else self.default

    def required(self, key):
        """Whether each call of the tool (source, name) needs approval: every call while the policy is unreadable."""
        rule = self.rule(key)
        return bool(self.error) or (self.fallback(key) if rule is None else rule)

    def snapshot(self, tools=None):
        """policy maps each tool in tools (approval_keys) to its rule, required to whether it asks first now (its rule, else
        its default: fallback()); without tools, Riko's own tools by name."""
        with self.lock:
            keys = approval_keys(tools) if tools is not None else {name: (RIKO, name) for name in [*self.legacy, *(n for s, n in self.policy if s == RIKO)]}
            # Report the enforced default: while the policy is unreadable every tool needs approval.
            return {'default_required': self.default or bool(self.error), 'policy': {name: rule for name, key in keys.items() if (rule := self.rule(key)) is not None},
                'required': {name: self.required(key) for name, key in keys.items()},
                'error': self.error, 'pending': [deepcopy(v['request']) for v in self.pending.values() if v['decision'] is None]}

    def configure(self, policy, names):
        """Set rules for registered tools: policy maps names in names (approval_keys) to booleans."""
        keys = approval_keys(names)
        if not isinstance(policy, dict) or any(k not in keys or type(v) is not bool for k, v in policy.items()):
            raise ValueError('Approval policy must map registered tool names to booleans')
        with self.lock:
            if self.error:
                # Keep the unreadable original (raises OSError if impossible, so it is never
                # overwritten), and persist "require approval" for every tool the user did not
                # just change, rather than dropping back to the permissive default.
                preserve_unreadable(self.path)
                base, legacy = {key: True for key in keys.values()}, {}
            else: base, legacy = self.policy, self.legacy
            updated = {**base, **{keys[name]: rule for name, rule in policy.items()}}
            sources = {}
            for (source, name), rule in updated.items(): sources.setdefault(source, {})[name] = rule
            def enforced(key):  # this build's rule for key once saved: explicit, then legacy, then the default
                rule = updated.get(key)
                if rule is None: rule = legacy.get(key[1]) if key[0] == RIKO or legacy.get(key[1]) is True else None
                return self.fallback(key) if rule is None else rule
            mirror = dict(legacy)  # for older builds, by name alone: ask first when any registered source would ask
            for key in {*updated, *keys.values()}: mirror[key[1]] = mirror.get(key[1], False) or enforced(key)
            atomic_write(self.path, json.dumps({**mirror, 'version': VERSION, 'sources': sources, **({'legacy': legacy} if legacy else {})}, indent=2))
            self.policy, self.legacy, self.error = updated, dict(legacy), ''
        event_bus.publish('tool.approval_policy')
        return self.snapshot(keys)

    def resolve(self, request_id, approved):
        if type(approved) is not bool: raise ValueError('approved must be boolean')
        with self.lock:
            entry = self.pending.get(request_id)
            if not entry or entry['decision'] is not None or time.time() >= entry['request']['expires_at']:
                raise ValueError('Approval request expired or already resolved')
            entry['decision'] = approved
            self.condition.notify_all()
        event_bus.publish('tool.approval_resolved', id=request_id, approved=approved)

    def authorize(self, name, arguments, call_id, cancelled=lambda: False, timeout=120, *, key=None):
        """Whether the call of tool name (key (source, name); Riko's own tool by default) may run, asking when its rule says so."""
        key = key or (RIKO, name)
        with self.lock:
            if self.closed or cancelled(): return False
            if not self.required(key): return True
            request_id = str(uuid.uuid4())
            request = {'id': request_id, 'name': name, 'source': key[0], 'arguments': deepcopy(arguments),
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
