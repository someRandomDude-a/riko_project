from pathlib import Path
import sys

import pytest

from process.app_core.persistence.tasks import TaskStore, TaskMCP, TaskConflict
from process.app_core.tools.registry import ToolRegistry, StdioMCPClient
from process.app_core.conversation.chat import ChatService
from process.app_core.conversation.messages import ChatMessage, ModelResponse, ToolCall


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
    registry.register_mcp(server)
    result = registry.execute('task_create', {'title': 'Build app'}, 'call1')
    assert not result.is_error
    task = result.content
    assert store.get(task['id'])['history'][0]['source_turn_id'] == 'turn42'
    store.update(task['id'], 1, {'title': 'Corrected app task'})
    result = registry.execute('task_update', {'task_id': task['id'], 'expected_revision': 1,
        'changes': {'status': 'completed'}, 'reason': 'Stale model claim'}, 'call2')
    assert result.is_error
    assert 'changed' in result.content


def test_tasks_are_requested_by_tools_not_preloaded(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    registry = ToolRegistry()
    registry.register_mcp(TaskMCP(store))
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
    chat = ChatService(Provider(), system_prompt='Riko', tool_registry=registry)
    chat.task_store = store
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
        conflict = client.call('task_update', {'task_id': task['id'], 'expected_revision': 9, 'changes': {'status': 'completed'}, 'reason': 'Wrong revision'})
        assert conflict['isError']
    finally:
        client.close()
        client.process.wait(timeout=5)


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
