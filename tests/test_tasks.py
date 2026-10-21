import os
from pathlib import Path
import re
import sys

import pytest

from process.app_core.persistence.tasks import TaskStore, TaskMCP, TaskConflict
from process.app_core.tools.mcp import stdio as mcp_stdio
from process.app_core.tools.registry import ToolRegistry, StdioMCPClient
from process.app_core.conversation.chat import ChatDeps, ChatService
from process.app_core.kernel.messages import ChatMessage, ModelResponse, ToolCall
from process.app_core.configuration.paths import DataPaths


def test_tasks_survive_restart_and_preserve_provenance(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    task = store.create('Finish project', source_turn_id='turn1', actor='model_mcp')
    updated = store.update(task['id'], 1, {'status': 'blocked', 'blocker': 'Need dataset', 'progress': .4}, reason='User reported blocker')
    reloaded = TaskStore(store.path)
    record = reloaded.get(task['id'])
    assert record['progress'] == .4
    assert record['revision'] == 2
    assert record['history'][0]['source_turn_id'] == 'turn1'
    assert record['history'][0]['snapshot']['status'] == 'active'
    assert record['history'][1]['reason'] == 'User reported blocker'
    assert updated['blocker'] == 'Need dataset'
    with pytest.raises(TaskConflict): reloaded.update(task['id'], 1, {'title': 'Stale overwrite'})


def test_task_context_excludes_paused_and_closed_and_deduplicates(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    task = store.create('Write report')
    assert store.create(' write REPORT ')['id'] == task['id']
    store.update(task['id'], 1, {'status': 'paused'})
    assert store.context() == []
    active = store.create('Read dataset')
    store.update(active['id'], 1, {'status': 'completed'})
    assert store.context() == []
    assert len(store.list(include_closed=True)) == 2
    assert len(store.get(task['id'])['history']) == 2


def test_mcp_rules_tools_and_conflicts_are_reported_as_errors(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    server = TaskMCP(store, source_turn=lambda: 'turn42')
    registry = ToolRegistry()
    registry.register_mcp(server, source='riko')
    result = registry.execute('task_create', {'title': 'Build app'}, 'call1')
    assert not result.is_error
    task = result.content
    assert store.get(task['id'])['history'][0]['source_turn_id'] == 'turn42'
    store.update(task['id'], 1, {'title': 'Corrected app task'})
    result = registry.execute('task_update', {'task_id': task['id'], 'expected_revision': 1,
        'changes': {'status': 'completed'}, 'reason': 'Stale model claim'}, 'call2')
    assert result.is_error
    assert 'changed' in result.content


def test_task_get_tool_returns_recent_change_summaries_while_the_store_keeps_every_snapshot(tmp_path):
    from process.app_core.persistence.tasks import TOOL_HISTORY_DEFAULT, TOOL_HISTORY_MAX, TOOL_REASON_CHARS
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    task = store.create('Learn Japanese', description='Daily practice. ' * 200, reason='User asked')
    for revision in range(1, 31):
        store.update(task['id'], revision, {'progress': revision / 40, 'next_step': f'Lesson {revision} ' + 'vocabulary. ' * 300},
                     reason=f'Finished lesson {revision}. ' + 'detail ' * 500)
    store.update(task['id'], 31, {'status': 'blocked', 'blocker': 'No textbook'}, reason='User said so')
    registry = ToolRegistry()
    registry.register_mcp(TaskMCP(store), source='riko')
    try:
        schema = next(d['function']['parameters'] for d in registry.definitions() if d['function']['name'] == 'task_get')
        assert schema['properties']['history_limit'] == {'type': 'integer', 'minimum': 0, 'maximum': TOOL_HISTORY_MAX} and schema['required'] == ['task_id']
        result = registry.execute('task_get', {'task_id': task['id']}, 'call')
        assert not result.is_error
        got = result.content
        assert got['revision'] == 32 and got['status'] == 'blocked' and got['description'] == task['description']  # the current record is whole
        assert [e['revision'] for e in got['history']] == list(range(32 - TOOL_HISTORY_DEFAULT + 1, 33)) and got['history_omitted'] == 32 - TOOL_HISTORY_DEFAULT
        assert all('snapshot' not in e and len(e['reason']) <= TOOL_REASON_CHARS + len(' [truncated]') for e in got['history'])
        assert got['history'][-1]['changed'] == ['status', 'blocker'] and got['history'][-1]['status'] == 'blocked'
        assert got['history'][0]['changed'] == ['progress', 'next_step']  # named against the event before the window
        assert len(str(got)) < 16000  # was ~110 KB: every snapshot of every revision
        oldest = registry.execute('task_get', {'task_id': task['id'], 'history_limit': TOOL_HISTORY_MAX}, 'call').content
        assert len(oldest['history']) == TOOL_HISTORY_MAX and oldest['history_omitted'] == 32 - TOOL_HISTORY_MAX
        assert registry.execute('task_get', {'task_id': task['id'], 'history_limit': 0}, 'call').content['history'] == []
        created = registry.execute('task_get', {'task_id': store.create('Fresh task')['id']}, 'call').content['history']
        assert [e['kind'] for e in created] == ['created'] and 'changed' not in created[0]
        for bad in (TOOL_HISTORY_MAX + 1, -1, True, 2.5):
            refused = registry.execute('task_get', {'task_id': task['id'], 'history_limit': bad}, 'call')
            assert refused.is_error and 'history_limit' in refused.content
        full = store.get(task['id'])  # the panel and /api/tasks/{id} still read every snapshot
        assert len(full['history']) == 32 and full['history'][0]['snapshot']['status'] == 'active' and 'history_omitted' not in full
        with pytest.raises(ValueError): store.get(task['id'], history_limit=-1)
    finally: registry.close()


def test_tasks_are_requested_by_tools_not_preloaded(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    registry = ToolRegistry()
    registry.register_mcp(TaskMCP(store), source='riko')
    class Provider:
        count = 0
        def generate(self, messages, **options):
            self.count += 1
            if self.count == 1:
                assert 'Build app' not in '\n'.join(m.content for m in messages)
                return ModelResponse(ChatMessage('assistant', tool_calls=[ToolCall('call', 'task_list', {'query': 'app', 'limit': 3})]))
            assert messages[-1].role == 'tool'
            assert 'Build app' in messages[-1].content
            return ModelResponse(ChatMessage('assistant', 'Task recorded.'))
    chat = ChatService(Provider(), system_prompt='Riko', tool_registry=registry, deps=ChatDeps(task_store=store))
    store.create('Build app')
    chat.respond('What is my app project status?')


def test_real_stdio_mcp_handshake_and_task_round_trip(tmp_path):
    entry = Path(__file__).resolve().parents[1] / 'Code' / 'task_mcp_server.py'
    client = StdioMCPClient(sys.executable, [str(entry), '--store', str(tmp_path / 'tasks.sqlite3')])
    try:
        assert 'task_create' in {tool['name'] for tool in client.list_tools()}
        result = client.call('task_create', {'title': 'Via stdio ✦'})
        assert not result['isError']
        task = result['structuredContent']
        stored = TaskStore(tmp_path / 'tasks.sqlite3').get(task['id'])
        assert stored['title'] == 'Via stdio ✦'
        assert stored['history'][0]['actor'] == 'external_mcp'
        got = client.call('task_get', {'task_id': task['id'], 'history_limit': 1})['structuredContent']
        assert got['history'] == [{k: v for k, v in stored['history'][0].items() if k != 'snapshot'} | {'status': 'active', 'progress': 0.0}] and got['history_omitted'] == 0
        conflict = client.call('task_update', {'task_id': task['id'], 'expected_revision': 9, 'changes': {'status': 'completed'}, 'reason': 'Wrong revision'})
        assert conflict['isError']
    finally:
        client.close()
        client.process.wait(timeout=5)


def test_mcp_command_is_found_on_the_child_path_and_a_miss_names_that_path(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="'riko-missing-mcp' was not found on PATH: " + re.escape(str(tmp_path))):
        StdioMCPClient('riko-missing-mcp', env={'PATH': str(tmp_path)})
    seen = []  # Windows: which() applies PATHEXT, so a bare npx becomes its npx.CMD shim on the merged PATH.
    monkeypatch.setattr(mcp_stdio.shutil, 'which', lambda command, path=None: seen.append((command, path)) or 'C:/node/npx.CMD')
    assert mcp_stdio.resolve_command('npx', {'PATH': os.pathsep.join(['/a', '/b'])}) == 'C:/node/npx.CMD'
    assert seen == [('npx', os.pathsep.join(['/a', '/b']))]
    assert mcp_stdio.resolve_command('C:/proj/.venv/Scripts/python', {}) == 'C:/proj/.venv/Scripts/python'  # a path runs as written


@pytest.mark.skipif(os.name == 'nt', reason='POSIX shell wrapper')
def test_mcp_server_named_by_bare_command_on_a_configured_path(tmp_path):
    entry, store = Path(__file__).resolve().parents[1] / 'Code' / 'task_mcp_server.py', tmp_path / 'tasks.sqlite3'
    wrapper = tmp_path / 'riko-task-mcp'
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{entry}" --store "{store}"\n'); wrapper.chmod(0o755)
    client = StdioMCPClient('riko-task-mcp', env={'PATH': str(tmp_path)})
    try: assert 'task_create' in {tool['name'] for tool in client.list_tools()}
    finally: client.close()


def test_python_mcp_server_gets_utf8_mode_and_undecodable_stderr_never_blocks_it(tmp_path, monkeypatch):
    # A Windows server logs in the ANSI code page (0xE9 is cp1252 e-acute). The drain must survive it, or the server
    # blocks once the stderr pipe fills.
    server = tmp_path / 'server.py'
    server.write_text('''import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    if 'id' not in request: continue
    sys.stderr.buffer.write(b'caf\\xe9\\n' + b'x' * 262144 + b'\\n'); sys.stderr.flush()
    result = {'tools': []} if request['method'] == 'tools/list' else {'encoding': os.environ.get('PYTHONIOENCODING'), 'stdout': sys.stdout.encoding}
    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)
''', encoding='utf-8')
    monkeypatch.delenv('PYTHONIOENCODING', raising=False)
    client = StdioMCPClient(sys.executable, [str(server)])
    try:
        assert client.list_tools() == []
        assert client.call('probe', {}) == {'encoding': 'utf-8', 'stdout': 'utf-8'}
    finally: client.close()
    explicit = StdioMCPClient(sys.executable, [str(server)], env={'PYTHONIOENCODING': 'latin-1'})
    try: assert explicit.call('probe', {}) == {'encoding': 'latin-1', 'stdout': 'iso8859-1'}  # a server's own setting wins
    finally: explicit.close()


def test_unloadable_mcp_server_is_reported_to_the_desktop(tmp_path):
    from types import SimpleNamespace
    from process.app_core.desktop.state import DesktopState
    (tmp_path / 'mcp.json').write_text('{"mcpServers": {"broken": {"command": "riko-missing-mcp"}}}', encoding='utf-8')
    config = SimpleNamespace(root=tmp_path, paths=DataPaths.at(tmp_path), tools=SimpleNamespace(mcp_config=tmp_path / 'mcp.json', timeout_seconds=5, require_approval=True, disabled=()))
    state = DesktopState()
    registry = ToolRegistry.from_config(config, activity=state.activity)  # what the factory injects: the desktop's ActivityLog
    try:
        notice = state.snapshot()['notifications'][0]
        assert notice['source'] == 'tools' and notice['level'] == 'error' and 'broken' in notice['text'] and 'riko-missing-mcp' in notice['text']
    finally: registry.close()


def test_task_validation_does_not_commit_failed_changes(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    task = store.create('Actual goal')
    with pytest.raises(ValueError): store.update(task['id'], 1, {'status': 'blocked'})
    with pytest.raises(ValueError): store.update(task['id'], 1, {'progress': 1})
    with pytest.raises(ValueError): store.update(task['id'], 1, {'progress': float('nan')})
    assert store.get(task['id'])['revision'] == 1
    assert len(store.get(task['id'])['history']) == 1
    completed = store.update(task['id'], 1, {'status': 'completed'}, reason='User confirmed completion')
    assert completed['progress'] == 1
    assert not store.context()
    reopened = store.update(task['id'], 2, {'status': 'active'}, reason='User reopened')
    assert reopened['progress'] == 0


def test_task_relevance_query_does_not_return_unrelated_records(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    store.create('Build app', next_step='Implement microphone support')
    store.create('Buy groceries')
    assert [r['title'] for r in store.list(query='microphone')] == ['Build app']
    assert store.list(query='unmatchedword') == []
