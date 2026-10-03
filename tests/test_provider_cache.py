from types import SimpleNamespace
import threading
import pytest

from process.app_core.configuration.config import RuntimeConfig
from process.app_core.conversation.messages import ChatMessage
from process.app_core.inference.providers import OpenAIProvider
from process.app_core.conversation.streaming import WordDeltas


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
