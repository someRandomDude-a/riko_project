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
