import json
from types import SimpleNamespace
import threading
import pytest

from process.app_core.configuration.config import RuntimeConfig
from process.app_core.kernel.messages import ChatMessage
from process.app_core.inference.providers import OpenAIProvider
from process.app_core.kernel.streaming import WordDeltas
from process.app_core.kernel.turns import TurnContext


def provider(create):
    instance = OpenAIProvider.__new__(OpenAIProvider)
    instance.config = RuntimeConfig(provider='lm_studio', model='test')
    instance.api_mode = 'auto'
    instance.responses_enabled = True
    instance.response_cache = []
    instance.cache_lock = threading.Lock()
    instance.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return instance


def response(text='Hello', response_id='resp_1'):
    raw = {'id': response_id, 'status': 'completed', 'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}], 'usage': {}}
    return SimpleNamespace(model_dump=lambda: raw)


def test_exact_prefix_reuse_and_changed_history_rejection():
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return response()
    instance = provider(create)
    history = [ChatMessage('system', 'Stable instructions'), ChatMessage('user', 'Hi')]
    answer = instance.generate(history)
    instance.generate([*history, answer.message, ChatMessage('user', 'Next')])
    assert requests[-1]['previous_response_id'] == 'resp_1'
    assert requests[-1]['input'] == [{'role': 'user', 'content': 'Next'}]
    instance.generate([*history, ChatMessage('assistant', 'Interrupted'), ChatMessage('user', 'Next')])
    assert 'previous_response_id' not in requests[-1]
    instance.generate([ChatMessage('system', 'Reflection'), ChatMessage('user', 'Hi')])
    assert 'previous_response_id' not in requests[-1]


def test_expired_response_retries_full_context():
    requests = []
    class Expired(Exception): status_code = 404
    def create(**kwargs):
        requests.append(kwargs)
        if kwargs.get('previous_response_id'): raise Expired()
        return response()
    instance = provider(create)
    history = [ChatMessage('user', 'Hi')]
    answer = instance.generate(history)
    instance.generate([*history, answer.message, ChatMessage('user', 'Next')])
    assert len(requests) == 3
    assert 'previous_response_id' not in requests[-1]
    assert len(requests[-1]['input']) == 3


def test_stream_words_and_cancellation_do_not_cache_partial_response():
    class Stream:
        closed = False
        def __iter__(self):
            yield SimpleNamespace(type='response.output_text.delta', delta='Hel')
            yield SimpleNamespace(type='response.output_text.delta', delta='lo wor')
            yield SimpleNamespace(type='response.output_text.delta', delta='ld')
            yield SimpleNamespace(type='response.completed', response=response('Hello world'))
        def close(self): self.closed = True
    stream = Stream()
    instance = provider(lambda **kwargs: stream)
    received = []
    instance.generate([ChatMessage('user', 'Hi')], on_delta=received.append)
    assert received == ['Hello ', 'world']
    assert stream.closed
    instance.response_cache.clear()
    def cancelled(text): raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError):
        instance.generate([ChatMessage('user', 'Hi')], on_delta=cancelled)
    assert not instance.response_cache


def test_word_deltas_preserve_whitespace_and_punctuation():
    received = []
    words = WordDeltas(received.append)
    for token in ['ab', 'cd', '.ef', 'gh ', 'ij', '\n', 'kl']:
        words.feed(token)
    words.finish()
    assert received == ['abcd.efgh ', 'ij\n', 'kl']


def test_unsupported_responses_falls_back_to_chat_completions():
    class Unsupported(Exception): status_code = 404
    def create(**kwargs): raise Unsupported()
    instance = provider(create)
    raw = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Hello', tool_calls=[]), finish_reason='stop')], usage=None)
    instance.client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: raw))
    assert instance.generate([ChatMessage('user', 'Hi')]).message.content == 'Hello'
    assert not instance.responses_enabled


def test_response_tools_use_call_ids_and_reuse_for_tool_result():
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        raw = {'id': 'resp_tool', 'status': 'completed', 'output': [
            {'type': 'function_call', 'call_id': 'call_1', 'name': 'lookup', 'arguments': '{"query": "color"}'}]}
        return SimpleNamespace(model_dump=lambda: raw)
    instance = provider(create)
    history = [ChatMessage('user', 'Hi')]
    tools = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}]
    answer = instance.generate(history, tools=tools)
    instance.generate([*history, answer.message, ChatMessage('tool', 'green', tool_call_id='call_1')], tools=tools)
    assert requests[-1]['previous_response_id'] == 'resp_tool'
    assert requests[-1]['input'] == [{'type': 'function_call_output', 'call_id': 'call_1', 'output': 'green'}]
    assert requests[-1]['tools'][0]['name'] == 'lookup'


def test_tool_free_stream_respects_responses_mode_and_closes_on_early_exit():
    requests=[]
    class Stream:
        closed=False
        def __iter__(self):
            yield SimpleNamespace(type='response.output_text.delta',delta='Hello world ')
            yield SimpleNamespace(type='response.completed',response=response('Hello world '))
        def close(self): self.closed=True
    stream=Stream()
    def create(**kwargs): requests.append(kwargs);return stream
    instance=provider(create)
    iterator=instance.stream([ChatMessage('user','Hi')])
    assert next(iterator) == 'Hello world '
    iterator.close()
    assert stream.closed
    assert requests[0]['stream'] is True and requests[0]['input'] == [{'role':'user','content':'Hi'}]
    assert 'messages' not in requests[0]


def test_explicit_responses_mode_never_falls_back_when_unsupported():
    class Unsupported(Exception): status_code=404
    def create(**kwargs): raise Unsupported()
    instance=provider(create)
    instance.api_mode='responses'
    with pytest.raises(Unsupported): list(instance.stream([ChatMessage('user','Hi')]))
    assert instance.responses_enabled


def test_remote_responses_uses_typed_partial_assistant_when_history_changes():
    requests=[]
    def create(**kwargs): requests.append(kwargs);return response()
    instance=provider(create)
    history=[ChatMessage('user','Hi')]
    instance.generate(history)
    instance.generate([*history,ChatMessage('assistant','Interrupted reply'),ChatMessage('user','Continue')])
    assert 'previous_response_id' not in requests[-1]
    assert requests[-1]['input'][1] == {'type':'message','role':'assistant',
        'content':[{'type':'output_text','text':'Interrupted reply'}]}


def test_overflowing_history_still_reuses_previous_responses_between_trims():
    def conversation(state):
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return response(f'Reply {len(requests)}', f'resp_{len(requests)}')
        instance = provider(create)
        instance.config.n_ctx, instance.config.max_output_tokens = 4096, 256
        history = [ChatMessage('system', 'Stable instructions')]
        for turn in range(30):
            history.append(ChatMessage('user', f'turn {turn} ' + ' '.join(['word'] * 250)))  # ~250 tokens a turn
            history.append(instance.generate(history, context_state=state).message)
        return sum(bool(request.get('previous_response_id')) for request in requests[15:])  # history overflows from turn 15
    assert conversation(None) == 0  # a prefix sliding every turn never matches a cached response
    assert conversation({}) >= 9  # trims a few turns apart; the turns between them reuse


def chat_completion(text='Hello'):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=[]), finish_reason='stop')], usage=None)


@pytest.mark.parametrize('responses', [False, True])
def test_stock_tools_fit_default_budget_and_strict_templates_get_one_leading_system(tmp_path, responses):
    from process.app_core.configuration.config import load_config
    from process.app_core.conversation.chat import ChatDeps, ChatService
    from process.app_core.persistence.tasks import TaskMCP, TaskStore, TASK_RULES
    from process.app_core.tools.registry import ToolRegistry
    (tmp_path / 'character_config.yaml').write_text('runtime:\n  provider: lm_studio\n', encoding='utf-8')
    config = load_config(tmp_path / 'character_config.yaml')
    from process.app_core.desktop.tools import iter_tools as desktop_tools
    registry = ToolRegistry.from_config(config, local_tools={'desktop': desktop_tools()})  # the stock tools, as the factory registers them
    try:
        registry.register_mcp(TaskMCP(TaskStore(tmp_path / 'tasks.sqlite3')), source='riko')
        assert len(json.dumps(registry.definitions('openai'))) > 7000  # its schemas alone exceeded the byte budget
        requests = []
        instance = provider(lambda **kwargs: requests.append(kwargs) or response())
        instance.config, instance.responses_enabled = config.runtime, responses
        instance.client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: requests.append(kwargs) or chat_completion()))
        chat = ChatService(instance, system_prompt=config.system_prompt + '\n' + TASK_RULES + '\n', tool_registry=registry,
            history_file=tmp_path / 'chat_history.json', memory_context=lambda text: 'Riko likes green tea.',
            deps=ChatDeps(context_limit=min(config.runtime.n_ctx, config.memory.context_window_tokens + config.runtime.max_output_tokens)))  # as factory sets it
        observe = TurnContext(runtime=lambda: {'observed_at': '2026-10-08T21:14:03+02:00', 'runtime': {'listening': True}})
        chat.conversation.append([ChatMessage('user', 'User: Earlier'), ChatMessage('assistant', 'Past answer')])
        assert chat.respond('Hello there', context=observe).message.content == 'Hello'
        sent = requests[-1]['input' if responses else 'messages']
        assert [item.get('role') for item in sent] == ['system', 'user', 'assistant', 'user']
        assert sent[0]['content'].startswith(config.system_prompt) and TASK_RULES in sent[0]['content']
        current = sent[-1]['content']
        assert 'Riko likes green tea.' in current and '"listening": true' in current and current.endswith('User: Hello there')
    finally: registry.close()


def test_tool_loop_observations_keep_the_cached_prefix_so_response_ids_are_reused():
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        if len(requests) > 1: return response('Done', 'resp_final')
        raw = {'id': 'resp_tool', 'status': 'completed', 'output': [
            {'type': 'function_call', 'call_id': 'call_1', 'name': 'lookup', 'arguments': '{"query": "color"}'}]}
        return SimpleNamespace(model_dump=lambda: raw)
    instance = provider(create)
    messages = [ChatMessage('system', 'Stable'), ChatMessage('system', 'Relevant memories: blue', context_kind='optional'),
        ChatMessage('user', 'Hi'), ChatMessage('system', 'Observation 1', context_kind='optional')]
    answer = instance.generate(messages)
    assert [item.get('role') for item in requests[0]['input']] == ['system', 'user']
    instance.generate([*messages, answer.message, ChatMessage('tool', 'green', tool_call_id='call_1'),
        ChatMessage('system', 'Observation 2', context_kind='optional')])
    assert requests[-1]['previous_response_id'] == 'resp_tool'
    [result] = requests[-1]['input']
    assert result['type'] == 'function_call_output' and 'Observation 2' in result['output'] and 'Observation 1' not in result['output']
    assert result['output'].endswith('green')


def test_remote_token_counts_are_calibrated_estimates_not_bytes():
    instance = provider(None)
    text = 'Riko remembers that the user prefers green tea in the evening.'
    assert 10 <= instance.count_text_tokens(text) < len(text.encode('utf-8')) / 3  # memory recall budgets use it


def test_chat_tool_loop_requests_extend_the_previous_one(tmp_path):
    """The tool loop keeps each sent observation, so request 2 extends request 1 and reuses its response id."""
    from process.app_core.conversation.chat import ChatDeps, ChatService
    requests, ticks = [], iter(range(100))
    def create(**kwargs):
        requests.append(kwargs)
        if len(requests) > 1: return response('Done', 'resp_final')
        raw = {'id': 'resp_tool', 'status': 'completed', 'output': [
            {'type': 'function_call', 'call_id': 'call_1', 'name': 'lookup', 'arguments': '{"query": "color"}'}]}
        return SimpleNamespace(model_dump=lambda: raw)
    instance = provider(create)
    registry = SimpleNamespace(definitions=lambda kind: [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}],
        execute=lambda name, arguments, call_id, cancelled=None: SimpleNamespace(content='green', tool_call_id=call_id, name=name))
    chat = ChatService(instance, system_prompt='Stable', tool_registry=registry, history_file=tmp_path / 'chat_history.json',
        memory_context=lambda text: 'Riko likes green tea.')
    chat.respond('What colour?', context=TurnContext(runtime=lambda: {'observed_at': f'2026-10-08T21:14:{next(ticks):02d}', 'runtime': {'listening': True}}))
    assert len(requests) == 2 and requests[1].get('previous_response_id') == 'resp_tool'
    assert [item['type'] for item in requests[1]['input']] == ['function_call_output']  # only the new tool result is sent
    assert 'Current runtime observation' not in requests[1]['input'][0]['output']  # unchanged state: no second observation


def test_runtime_observation_ignores_nested_timestamps_and_idle_seconds_within_a_minute():
    """initiative's environment (perception.computer) has its own observed_at, and idle time grows on every sample: neither
    is a state change, so the tool loop does not append an observation per iteration; a new minute of idleness is."""
    from process.app_core.conversation.chat import observation_state
    def observed(second, idle):
        return {'observed_at': f'21:14:{second:02d}', 'runtime': {'listening': True},
                'perception': {'computer': {'idle_seconds': idle, 'active_app': {'app_name': 'Finder'}, 'observed_at': f'21:14:{second:02d}'}}}
    assert observation_state(observed(1, 884.9)) == observation_state(observed(3, 886.9))
    assert observation_state(observed(1, 884.9)) != observation_state(observed(59, 900.1))
    assert observation_state(observed(1, 5)) != observation_state({**observed(1, 5), 'perception': {'computer': {'idle_seconds': 5, 'active_app': {'app_name': 'Mail'}}}})
    assert observation_state([{'observed_at': 1, 'a': 2}]) == [{'a': 2}] and observation_state('text') == 'text'
