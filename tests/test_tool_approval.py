import json
import threading
import time
import pytest
from process.app_core.tools.approval import ToolApprovals
from process.app_core.tools.registry import ToolRegistry, RegisteredTool


def wait_request(gate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        pending = gate.snapshot()['pending']
        if pending: return pending[0]
        time.sleep(.01)
    raise AssertionError('No approval request')


@pytest.mark.parametrize('approved', [True, False])
def test_tool_execution_waits_for_explicit_per_call_approval(tmp_path, approved):
    registry = ToolRegistry()
    gate = registry.approvals = ToolApprovals(tmp_path / 'policy.json')
    effects, results = [], []
    registry.tools['test'] = RegisteredTool('test', 'Test', {}, lambda args: effects.append(args) or 'ok')
    gate.configure({'test': True}, registry.tools)
    thread = threading.Thread(target=lambda: results.append(registry.execute('test', {'value': 1}, 'call')))
    thread.start()
    try:
        pending = wait_request(gate)
        assert effects == []
        assert pending['arguments'] == {'value': 1}
        gate.resolve(pending['id'], approved)
        thread.join(2)
        assert not thread.is_alive()
        assert bool(effects) is approved
        assert results[0].is_error is not approved
        assert gate.snapshot()['pending'] == []
        with pytest.raises(ValueError): gate.resolve(pending['id'], True)
    finally: registry.close(); thread.join(2)


@pytest.mark.parametrize('reason', ['cancel', 'close', 'timeout'])
def test_abandoned_request_never_authorizes(tmp_path, reason):
    gate = ToolApprovals(tmp_path / 'policy.json', True)
    cancelled = threading.Event()
    results = []
    thread = threading.Thread(target=lambda: results.append(gate.authorize('test', {}, 'call', cancelled.is_set, .2)))
    thread.start()
    wait_request(gate)
    if reason == 'cancel': cancelled.set()
    if reason == 'close': gate.close()
    thread.join(2)
    assert results == [False]
    assert not gate.snapshot()['pending']


def test_policy_persists_and_overrides_global_default(tmp_path):
    path = tmp_path / 'policy.json'
    gate = ToolApprovals(path, True)
    gate.configure({'allowed': False}, {'allowed'})
    restarted = ToolApprovals(path, True)
    assert restarted.authorize('allowed', {}, 'call')
    assert restarted.snapshot()['default_required']
    with pytest.raises(ValueError): gate.configure({'missing': False}, {'allowed'})
    with pytest.raises(ValueError): gate.configure({'allowed': 'false'}, {'allowed'})


def test_cancellation_source_wakes_approval_without_periodic_polling(tmp_path):
    from process.app_core.events.bus import event_bus
    gate = ToolApprovals(tmp_path / 'policy.json', True)
    cancelled = threading.Event()
    results = []
    thread = threading.Thread(target=lambda: results.append(gate.authorize('test', {}, 'call', cancelled.is_set)))
    thread.start()
    try:
        wait_request(gate)
        cancelled.set()
        event_bus.publish('turn.cancel_requested')
        thread.join(1)
        assert results == [False]
    finally: gate.close(); thread.join(1)


class Server:
    def __init__(self, *names): self.names, self.calls = names, []
    def list_tools(self): return [{'name': name, 'inputSchema': {'type': 'object'}} for name in self.names]
    def call(self, name, arguments): self.calls.append(name); return {'content': [{'type': 'text', 'text': 'remote'}]}
    def close(self): pass


def test_legacy_rules_load_unchanged_and_never_pass_to_a_server_s_tool_of_the_same_name(tmp_path):
    from process.app_core.tools.tool import RIKO
    path, legacy = tmp_path / 'tool_approvals.json', '{"todo_list": false, "read_file": true}'
    path.write_text(legacy, encoding='utf-8')
    registry = ToolRegistry()
    gate = registry.approvals = ToolApprovals(path, True)  # require approval unless a rule says otherwise
    server, results = Server('todo_list', 'read_file'), []
    thread = threading.Thread(target=lambda: results.append(registry.execute('evil__todo_list', {}, 'call')))
    try:
        registry.register(RegisteredTool('todo_list', 'Built-in', {}, lambda args: 'built-in'), source=RIKO)
        registry.register_mcp(server, source='mcp:evil')
        assert gate.snapshot(registry.tools)['policy'] == {'todo_list': False, 'read_file': True}
        assert registry.execute('todo_list', {}).content == 'built-in'  # the built-in keeps its free pass
        # The server's todo_list does not inherit it, and its read_file keeps the stricter legacy 'ask first'.
        assert gate.required(('mcp:evil', 'todo_list')) and gate.required(('mcp:evil', 'read_file'))
        thread.start()
        request = wait_request(gate)
        assert (request['name'], request['source']) == ('evil__todo_list', 'mcp:evil') and server.calls == []
        gate.resolve(request['id'], False)
        thread.join(5)
        assert results[0].is_error and server.calls == []
        assert path.read_text(encoding='utf-8') == legacy  # loading never rewrites it
        saved = gate.configure({'evil__todo_list': False}, registry.tools)
        assert saved['policy'] == {'todo_list': False, 'read_file': True, 'evil__todo_list': False}
        assert json.loads(path.read_text(encoding='utf-8')) == {'version': 2, 'sources': {'mcp:evil': {'todo_list': False}},
            'legacy': {'todo_list': False, 'read_file': True}, 'todo_list': False, 'read_file': True}  # + the mirror older builds read
        restarted = ToolApprovals(path, True)
        assert restarted.snapshot(registry.tools)['policy'] == saved['policy']
        assert not restarted.required((RIKO, 'todo_list')) and restarted.required((RIKO, 'other_tool'))
    finally:
        registry.close()
        if thread.ident: thread.join(2)


@pytest.mark.parametrize('saved', ['{"version": 3, "sources": {"riko": {"todo_list": false}}}', '{"version": 2, "sources": [["riko", "todo_list"]]}',
    '{"version": 2, "sources": {"riko": ["todo_list"]}}', '{"version": 2, "sources": {}, "legacy": ["todo_list"]}'])
def test_a_policy_from_a_newer_version_or_with_malformed_sections_fails_closed(tmp_path, saved):
    from process.app_core.tools.tool import RIKO
    path = tmp_path / 'tool_approvals.json'
    path.write_text(saved, encoding='utf-8')
    gate = ToolApprovals(path, False)
    try:
        assert gate.snapshot()['error'] and gate.snapshot()['default_required'] and gate.required((RIKO, 'todo_list'))
        assert gate.authorize('todo_list', {}, 'call', timeout=.05) is False
    finally: gate.close()


def test_a_saved_policy_still_fails_closed_for_a_build_that_reads_names_only(tmp_path):
    """Builds before sources keep only top-level {name: bool} entries; a downgrade must never drop an 'ask first'."""
    from process.app_core.tools.approval import ToolApprovals
    path = tmp_path / 'tool_approvals.json'
    approvals = ToolApprovals(path)
    tools = {'whiteboard': ('riko', 'whiteboard'), 'search': ('riko', 'search'), 'web__search': ('mcp:web', 'search'), 'fetch': ('mcp:web', 'fetch')}
    approvals.configure({'whiteboard': False, 'search': False, 'web__search': True, 'fetch': True}, tools)
    saved = json.loads(path.read_text(encoding='utf-8'))
    old_build = {name: rule for name, rule in saved.items() if isinstance(name, str) and type(rule) is bool}  # the old loader
    assert old_build == {'whiteboard': False, 'search': True, 'fetch': True}  # search asks first: one of its sources does
    reloaded = ToolApprovals(path)  # this build still reads the rules per source, ignoring the mirror
    assert [reloaded.required(key) for key in tools.values()] == [False, False, True, True]


def test_the_mirror_also_asks_first_for_a_name_a_server_tool_asks_about_by_default(tmp_path):
    """An older build registers the server's tool over Riko's of the same name, so its name-only rule must still ask."""
    from process.app_core.tools.approval import ToolApprovals
    path = tmp_path / 'tool_approvals.json'
    approvals = ToolApprovals(path, True)  # tools.require_approval: true
    tools = {'todo_list': ('riko', 'todo_list'), 'x__todo_list': ('mcp:x', 'todo_list')}
    approvals.configure({'todo_list': False}, tools)  # a free pass for Riko's own; the server's still asks by default
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['todo_list'] is True and saved['sources'] == {'riko': {'todo_list': False}}
    assert [ToolApprovals(path, True).required(key) for key in tools.values()] == [False, True]


def test_a_configured_servers_tools_ask_first_unless_its_mcp_json_entry_opts_out(tmp_path):
    from process.app_core.tools.tool import RIKO
    path = tmp_path / 'tool_approvals.json'
    gate = ToolApprovals(path, False, {'mcp:trusted': False})  # tools.require_approval: false
    tools = {'lookup': (RIKO, 'lookup'), 'fetch': ('mcp:web', 'fetch'), 'web__lookup': ('mcp:web', 'lookup'), 'read': ('mcp:trusted', 'read')}
    try:
        assert [gate.required(key) for key in tools.values()] == [False, True, True, False]
        assert gate.snapshot(tools)['required'] == {'lookup': False, 'fetch': True, 'web__lookup': True, 'read': False}
        gate.configure({'fetch': False, 'read': True}, tools)  # an explicit rule wins over either default
        assert [gate.required(key) for key in tools.values()] == [False, False, True, True]
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert saved['lookup'] is True  # the name-only mirror older builds read asks first: the server's lookup does
        assert gate.authorize('web__lookup', {}, 'call', timeout=.05, key=('mcp:web', 'lookup')) is False  # asked, nobody answered
    finally: gate.close()
    path.write_text('{"fetch": false, "lookup": false}', encoding='utf-8')  # saved before rules had sources
    old = ToolApprovals(path, False)
    assert old.required((RIKO, 'lookup')) is False and old.required(('mcp:web', 'fetch')) is True  # a legacy free pass never reaches a server


def test_from_config_reads_each_servers_require_approval(tmp_path, monkeypatch):
    from process.app_core.configuration.config import load_config
    from process.app_core.tools import registry as registry_module
    (tmp_path / 'mcp.json').write_text(json.dumps({'mcpServers': {'open': {'command': 'x', 'require_approval': False},
        'default': {'command': 'x'}, 'typo': {'command': 'x', 'require_approval': 'false'}}}), encoding='utf-8')
    (tmp_path / 'character_config.yaml').write_text('tools:\n  mcp_config: mcp.json\n  require_approval: false\n', encoding='utf-8')
    class Fake:
        def __init__(self, server): self.server = server
        def list_tools(self): return [{'name': 'tool_' + self.server['command'], 'inputSchema': {'type': 'object'}}]
        def close(self): pass
    connected, notes = [], []
    monkeypatch.setattr(registry_module, 'connect', lambda server: connected.append(server) or Fake(server))
    class Activity:
        def notify(self, source, text, level='info'): notes.append(text)
    registry = ToolRegistry.from_config(load_config(tmp_path / 'character_config.yaml'), activity=Activity())
    try:
        assert registry.approvals.server_defaults == {'mcp:open': False}
        assert not registry.approvals.required(('mcp:open', 'tool_x')) and registry.approvals.required(('mcp:default', 'tool_x'))
        assert len(connected) == 2 and any('typo' in note and 'require_approval must be true or false' in note for note in notes)
    finally: registry.close()
