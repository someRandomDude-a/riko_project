import json
from types import SimpleNamespace
import pytest

from process.app_core.conversation.chat import ChatDeps, ChatService
from process.app_core.desktop.state import DesktopState
from process.app_core.events.bus import event_bus
from process.app_core.runtime.initiative import Initiative, DEFAULTS, InitiativeDecisionError, parse_decision
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.runtime.session import SessionManager
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths


class Speech:
    def __init__(self, *args): self.items = []
    def submit(self, *args): self.items.append(args); return True
    def cancel(self): pass
    def close(self): pass


class Adapter:
    """An activity adapter (runtime/activity.py) answering each field from `answers` (a value, an exception, or a callable)."""
    def __init__(self, idle=None, active_app=None, unsupported=()):
        self.answers, self.unsupported, self.calls = {'idle': idle, 'active_app': active_app}, set(unsupported), []
    def capabilities(self):
        return {field: {'supported': field not in self.unsupported, 'reason': 'not here' if field in self.unsupported else ''} for field in ('idle', 'active_app')}
    def sample(self, **kwargs):
        self.calls.append(kwargs)
        [field] = [field for field, wanted in kwargs.items() if wanted]
        answer = self.answers[field]
        answer = answer() if callable(answer) else answer
        if isinstance(answer, Exception): raise answer
        return {'idle_seconds' if field == 'idle' else 'active_app': answer}


def engine(tmp_path, monkeypatch, *, adapter=None, generate=None, raw=None):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    config = SimpleNamespace(root=tmp_path, paths=DataPaths.at(tmp_path), raw=raw or {}, character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    provider = BaseProvider()
    provider.generate = generate or (lambda *args, **kwargs: ModelResponse(ChatMessage('assistant', json.dumps({'initiate': True, 'message': 'Would you like a hand?', 'urgent': False}))))
    chat = ChatService(provider, system_prompt='Riko')
    session = SessionManager(config, chat, DesktopState())
    session.initiative = Initiative(session, adapter=adapter, start_worker=False)
    return session.initiative


def test_default_disabled_and_sources_require_consent(tmp_path, monkeypatch):
    adapter = Adapter(idle=100)
    calls = adapter.calls
    initiative = engine(tmp_path, monkeypatch, adapter=adapter)
    initiative.poll()
    assert not calls
    initiative.update({'enabled': True})
    initiative.poll()
    assert not calls
    assert 'idle_seconds' not in initiative.environment
    initiative.update({'enabled': True, 'observe_idle': True})
    initiative.poll(initiative.last_sample + 3)
    assert calls == [{'idle': True}]  # one call per allowed field
    initiative.update({'enabled': False})
    assert initiative.environment == {}
    initiative.session.close()


def test_idle_return_and_app_transitions(tmp_path, monkeypatch):
    idle, apps = iter([0, 6, 0]), iter([{'window_title': 'Editor'}, {'window_title': 'Editor'}, {'window_title': 'Browser'}])
    initiative = engine(tmp_path, monkeypatch, adapter=Adapter(idle=lambda: next(idle), active_app=lambda: next(apps)))
    initiative.update({'enabled': True, 'observe_idle': True, 'observe_active_app': True, 'idle_seconds': 5})
    observed = []
    unsubscribe = event_bus.subscribe(lambda event: observed.append(event.type))
    try:
        base = initiative.last_tick
        for offset in (3, 6, 9): initiative.poll(base + offset)
        assert observed.count('environment.user_idle') == 1
        assert observed.count('environment.user_returned') == 1
        assert observed.count('environment.active_app_changed') == 1
    finally:
        unsubscribe()
        initiative.session.close()


def test_an_unsupported_field_is_never_sampled_and_is_not_an_error(tmp_path, monkeypatch):
    adapter = Adapter(idle=12, active_app={'app_name': 'Finder'}, unsupported={'active_app'})
    initiative = engine(tmp_path, monkeypatch, adapter=adapter)
    try:
        assert initiative.snapshot()['perception'] == {'idle': 'off', 'active_app': 'off'}
        initiative.update({'enabled': True, 'observe_idle': True, 'observe_active_app': True})
        assert initiative.snapshot()['perception'] == {'idle': 'waiting', 'active_app': 'unsupported'}
        initiative.poll(initiative.last_sample + 3)
        snapshot = initiative.snapshot()
        assert adapter.calls == [{'idle': True}] and snapshot['error'] == '' and snapshot['environment']['idle_seconds'] == 12
        assert snapshot['perception'] == {'idle': 'available', 'active_app': 'unsupported'}
        assert snapshot['capabilities']['active_app'] == {'supported': False, 'reason': 'not here'}
        assert initiative.session.runtime_snapshot()['perception'] | {'computer': None} == {'screen': 'unavailable', 'computer': None,
            'idle': 'available', 'active_application': 'unsupported'}
    finally: initiative.session.close()


def test_each_field_reports_its_own_failure_and_a_desktop_without_the_service_turns_unsupported(tmp_path, monkeypatch):
    from process.app_core.runtime.activity import ActivityUnsupported
    adapter = Adapter(idle=RuntimeError('D-Bus timed out'), active_app={'window_title': 'Editor', 'app_name': 'Code'})
    initiative = engine(tmp_path, monkeypatch, adapter=adapter)
    try:
        initiative.update({'enabled': True, 'observe_idle': True, 'observe_active_app': True})
        initiative.poll(initiative.last_sample + 3)
        snapshot = initiative.snapshot()
        assert snapshot['perception'] == {'idle': 'error', 'active_app': 'available'} and snapshot['error'] == 'Idle awareness: D-Bus timed out'
        assert snapshot['environment']['active_app']['app_name'] == 'Code' and 'idle_seconds' not in snapshot['environment']
        adapter.answers['idle'] = ActivityUnsupported('this desktop offers no idle-time service')
        initiative.poll(initiative.last_sample + 3)
        snapshot = initiative.snapshot()
        assert snapshot['perception'] == {'idle': 'unsupported', 'active_app': 'available'} and snapshot['error'] == ''
        initiative.error = 'Initiative returned empty decision JSON'  # a decision's error is not the sampler's to clear
        initiative.poll(initiative.last_sample + 3)
        assert initiative.error == 'Initiative returned empty decision JSON'
    finally: initiative.session.close()


def test_app_condition_matches_an_app_name_where_no_window_title_is_offered(tmp_path, monkeypatch):
    initiative = engine(tmp_path, monkeypatch)
    try:
        initiative.update({'enabled': True})
        rule = {**initiative.settings['rules'][0], 'app_contains': 'figma'}
        initiative.environment = {'active_app': {'app_name': 'Figma', 'bundle_id': 'com.figma.Desktop', 'process_id': 7}}
        assert initiative.evaluate(rule, {'type': 'initiative.tick'})
        initiative.last_presented = initiative.last_evaluation = float('-inf')
        initiative.rule_last.clear()
        initiative.environment = {'active_app': {'app_name': 'Safari', 'process_id': 8}}
        assert not initiative.evaluate(rule, {'type': 'initiative.tick'})
    finally: initiative.session.close()


def test_bubble_default_cooldown_and_settings_persistence(tmp_path, monkeypatch):
    initiative = engine(tmp_path, monkeypatch)
    initiative.update({'enabled': True})
    rule = initiative.settings['rules'][0]
    assert initiative.evaluate(rule, {'type': 'initiative.tick'})
    assert initiative.session.state.snapshot()['speech'] == 'Would you like a hand?'
    assert initiative.session.chat.history[-1].role == 'assistant'
    assert not initiative.session.speech.items
    assert not initiative.evaluate(rule, {'type': 'initiative.tick'})
    assert json.loads(initiative.path.read_text())['enabled']
    initiative.session.close()
    restored = engine(tmp_path, monkeypatch)
    assert restored.settings['enabled']
    restored.session.close()


def test_initiative_retains_character_but_omits_desktop_and_memory_context(tmp_path, monkeypatch):
    captured = []
    def generate(messages, **kwargs):
        captured.extend(messages)
        return ModelResponse(ChatMessage('assistant', '{"initiate":false,"message":"","urgent":false}'))
    initiative = engine(tmp_path, monkeypatch, generate=generate)
    chat = initiative.session.chat
    chat.system_prompt = 'Character identity must remain.'
    chat.conversation.append([ChatMessage('user', 'x' * 2000) for _ in range(24)])
    initiative.update({'enabled': True})
    try:
        assert not initiative.evaluate(initiative.settings['rules'][0], {})
        assert captured[0].content.startswith(chat.system_prompt)
        payload = json.loads(captured[1].content)
        assert set(payload) == {'rule_instruction', 'event_type', 'emotion', 'model_state', 'recent_messages'}
        assert len(payload['recent_messages']) == 24
        assert all(len(m['content']) == 2000 for m in payload['recent_messages'])
        assert captured[1].context_kind == 'initiative'
    finally: initiative.session.close()


def test_foreground_supersedes_background_decision(tmp_path, monkeypatch):
    initiative = engine(tmp_path, monkeypatch)
    initiative.update({'enabled': True})
    def generate(*args, **kwargs):
        initiative.session._interaction_revision += 1
        kwargs['on_delta']('decision ')
        raise AssertionError('Expected cancellation')
    initiative.session.chat.provider.generate = generate
    assert not initiative.evaluate(initiative.settings['rules'][0], {})
    assert not initiative.session.chat.history
    initiative.session.close()


def test_spoken_requires_permission_and_sleep_blocks_all_initiative(tmp_path, monkeypatch):
    initiative = engine(tmp_path, monkeypatch)
    initiative.update({'enabled': True})
    rule = {**initiative.settings['rules'][0], 'presentation': 'spoken'}
    assert initiative.evaluate(rule, {})
    assert not initiative.session.speech.items
    initiative.last_presented = initiative.last_evaluation = float('-inf')
    initiative.rule_last.clear()
    initiative.update({'spoken_enabled': True})
    assert initiative.evaluate(rule, {})
    assert initiative.session.speech.items
    initiative.session._speech_pending = 0
    initiative.session.state.set_sleep(True)
    assert not initiative.evaluate(rule, {})
    initiative.session.close()


def test_declarative_validation_and_custom_event_coalescing(tmp_path, monkeypatch):
    initiative = engine(tmp_path, monkeypatch)
    with pytest.raises(ValueError): initiative.update({'interval_seconds': 0})
    with pytest.raises(ValueError): initiative.update({'run_shell': 'anything'})
    rule = {**DEFAULTS['rules'][0], 'id': 'custom', 'event': 'user.custom'}
    initiative.update({'enabled': True, 'rules': [rule]})
    event_bus.publish('user.custom', context='First')
    event_bus.publish('user.custom', context='Latest')
    assert len(initiative.pending) == 1
    assert initiative.pending['custom'][1]['payload']['context'] == 'Latest'
    initiative.session.close()


def test_model_can_stay_silent_and_disabled_urgent_override_is_respected(tmp_path, monkeypatch):
    initiative = engine(tmp_path, monkeypatch, generate=lambda *a, **k: ModelResponse(ChatMessage('assistant', json.dumps({
        'initiate': False, 'message': '', 'urgent': False}))))
    initiative.update({'enabled': True})
    assert not initiative.evaluate(initiative.settings['rules'][0], {})
    assert not initiative.session.chat.history
    assert initiative.last_decision['initiate'] is False
    initiative.session.chat.provider.generate = lambda *a, **k: ModelResponse(ChatMessage('assistant', json.dumps({
        'initiate': True, 'message': 'Please check this.', 'urgent': True})))
    initiative.last_evaluation = float('-inf')
    assert initiative.evaluate(initiative.settings['rules'][0], {})
    assert not initiative.session.speech.items
    initiative.update({'enabled': False, 'allow_urgent_spoken': True})
    assert not initiative.evaluate(initiative.settings['rules'][0], {})
    initiative.session.close()


def test_corrupt_optional_settings_fail_closed_without_overwriting(tmp_path, monkeypatch):
    path = tmp_path / 'persistent_memories' / 'initiative_settings.json'
    path.parent.mkdir()
    path.write_text('broken JSON', encoding='utf-8')
    initiative = engine(tmp_path, monkeypatch)
    assert not initiative.settings['enabled']
    assert 'unable to load settings' in initiative.error
    assert path.read_text() == 'broken JSON'
    initiative.session.close()


def test_an_unknown_yaml_key_is_ignored_instead_of_disabling_initiative(tmp_path, monkeypatch):
    initiative = engine(tmp_path, monkeypatch, raw={'initiative': {'enabled': True, 'interval_secs': 30}})  # load_config reports it
    try:
        assert initiative.settings['enabled'] and not initiative.error and 'interval_secs' not in initiative.settings
        with pytest.raises(ValueError, match='Unknown'): initiative.update({'interval_secs': 30})  # a request is still checked strictly
    finally: initiative.session.close()


def test_idle_check_requests_tasks_through_read_only_tool(tmp_path, monkeypatch):
    from process.app_core.persistence.tasks import TaskStore, TaskMCP
    from process.app_core.tools.registry import ToolRegistry
    from process.app_core.kernel.messages import ToolCall
    initiative = engine(tmp_path, monkeypatch)
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    store.create('Write private report')
    registry = ToolRegistry()
    registry.register_mcp(TaskMCP(store), source='riko')
    chat = initiative.session.chat
    chat.deps, chat.tool_registry = ChatDeps(task_store=store), registry
    calls = []
    def generate(messages, **kwargs):
        calls.append(1)
        assert {t['function']['name'] for t in kwargs['tools']} == {'task_list', 'task_get'}
        if len(calls) == 1:
            assert 'Write private report' not in '\n'.join(m.content for m in messages)
            assert 'user is idle' in messages[0].content
            return ModelResponse(ChatMessage('assistant', tool_calls=[ToolCall('lookup', 'task_list', {'query': 'report', 'limit': 2})]))
        assert messages[-1].role == 'tool'
        assert 'Write private report' in messages[-1].content
        return ModelResponse(ChatMessage('assistant', json.dumps({'initiate': False, 'urgent': False, 'message': ''})))
    chat.provider.generate = generate
    initiative.update({'enabled': True})
    assert not initiative.evaluate(initiative.settings['rules'][0], {'type': 'environment.user_idle'})
    assert len(calls) == 2
    assert not chat.history
    initiative.session.close()


def test_initiative_offers_only_read_only_tools_and_refuses_a_call_to_any_other(tmp_path, monkeypatch):
    from process.app_core.tools.registry import ToolRegistry
    from process.app_core.tools.tool import RIKO, RegisteredTool
    from process.app_core.kernel.messages import ToolCall
    initiative = engine(tmp_path, monkeypatch)
    registry, effects, offered = ToolRegistry(), [], []
    registry.register(RegisteredTool('peek', 'Look', {'type': 'object'}, lambda args: 'seen', read_only=True), source=RIKO)
    registry.register(RegisteredTool('poke', 'Change', {'type': 'object'}, lambda args: effects.append(args) or 'done'), source=RIKO)
    chat = initiative.session.chat
    chat.tool_registry = registry
    def generate(messages, **kwargs):
        offered.append({tool['function']['name'] for tool in kwargs['tools']})
        return ModelResponse(ChatMessage('assistant', tool_calls=[ToolCall('call', 'poke', {})]))
    chat.provider.generate = generate
    initiative.update({'enabled': True})
    try:
        with pytest.raises(ValueError, match='only permits read-only tools'):
            initiative.evaluate(initiative.settings['rules'][0], {'type': 'environment.user_idle'})
        assert offered == [{'peek'}] and effects == []
    finally: registry.close(); initiative.session.close()


@pytest.mark.parametrize('content', ['', '   ', 'Not now.', 'null', '[]', '{}',
    '{"initiate": "false", "message": "", "urgent": false}', '{"initiate": false'])
def test_invalid_decisions_have_actionable_diagnostics(content):
    response = ModelResponse(ChatMessage('assistant', content), 'stop')
    with pytest.raises(InitiativeDecisionError) as error: parse_decision(response)
    assert 'decision JSON' in str(error.value)
    assert error.value.diagnostics['content_chars'] == len(content)
    assert content not in error.value.diagnostics.values() or content == ''


@pytest.mark.parametrize('wrapper', ['{}', '```json\n{}\n```', '```JSON\n{}\n```', '```{} ```'])
def test_decision_parser_accepts_complete_json_and_fences(wrapper):
    proposal = {'initiate': False, 'message': '', 'urgent': False}
    assert parse_decision(ModelResponse(ChatMessage('assistant', wrapper.format(json.dumps(proposal))))) == proposal


def test_empty_reasoning_only_decision_errors_without_retry(tmp_path, monkeypatch):
    calls = []
    def generate(messages, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return ModelResponse(ChatMessage('assistant', ''), 'length',
                {'output_tokens': 1024, 'output_tokens_details': {'reasoning_tokens': 1024}},
                {'status': 'incomplete', 'output': [{'type': 'reasoning'}]})
        raise AssertionError('Must not restart inference for repair')
    initiative = engine(tmp_path, monkeypatch, generate=generate)
    initiative.update({'enabled': True})
    try:
        with pytest.raises(InitiativeDecisionError, match='truncated'): initiative.evaluate(initiative.settings['rules'][0], {})
        assert [call['max_output_tokens'] for call in calls] == [1024]
        assert initiative.last_decision is None
        assert not initiative.session.chat.history
    finally: initiative.session.close()


def test_repeated_invalid_decision_fails_closed_and_publishes_diagnostics(tmp_path, monkeypatch):
    calls, events = [], []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return ModelResponse(ChatMessage('assistant', ''), 'length',
            {'output_tokens': kwargs['max_output_tokens'], 'output_tokens_details': {'reasoning_tokens': kwargs['max_output_tokens']}})
    initiative = engine(tmp_path, monkeypatch, generate=generate)
    initiative.update({'enabled': True})
    unsubscribe = event_bus.subscribe(events.append)
    try:
        import time
        initiative._evaluate_job(initiative.settings['rules'][0], {}, time.monotonic(), initiative.version)
        errors = [event.as_dict()['payload'] for event in events if event.type == 'initiative.error']
        assert len(calls) == 1
        assert len(errors) == 1
        assert errors[0]['reason'] == 'truncated'
        assert errors[0]['reasoning_tokens'] == 1024
        assert errors[0]['content_chars'] == 0
        assert 'Expecting value' not in initiative.error
        assert not initiative.session.chat.history
        assert not initiative.session.speech.items
    finally:
        unsubscribe()
        initiative.session.close()


def test_foreground_cancels_attempt_before_parsing(tmp_path, monkeypatch):
    calls = []
    initiative = engine(tmp_path, monkeypatch)
    initiative.update({'enabled': True})
    def generate(*args, **kwargs):
        calls.append(kwargs)
        initiative.session._interaction_revision += 1
        kwargs['on_delta']('partial')
        raise AssertionError('Cancellation should stop attempt')
    initiative.session.chat.provider.generate = generate
    try:
        assert not initiative.evaluate(initiative.settings['rules'][0], {})
        assert len(calls) == 1
        assert not initiative.session.chat.history
    finally: initiative.session.close()


def test_truncated_but_parseable_json_is_not_presented():
    response = ModelResponse(ChatMessage('assistant', '{"initiate":true,"message":"Do it","urgent":true}'), 'length')
    with pytest.raises(InitiativeDecisionError, match='truncated'): parse_decision(response)


def test_initiative_uses_its_own_configurable_budgets(tmp_path, monkeypatch):
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return ModelResponse(ChatMessage('assistant', '{"initiate":false,"message":"","urgent":false}'))
    initiative = engine(tmp_path, monkeypatch, generate=generate)
    initiative.update({'enabled': True, 'context_window_tokens': 6144, 'max_output_tokens': 1536})
    try:
        assert not initiative.evaluate(initiative.settings['rules'][0], {})
        assert calls[0]['max_output_tokens'] == 1536
        assert calls[0]['context_limit'] == 6144
        with pytest.raises(ValueError): initiative.update({'max_output_tokens': 6144})
    finally: initiative.session.close()
