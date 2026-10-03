import json

import httpx
import pytest

from process.app_core.conversation.messages import ChatMessage, ToolCall
from process.app_core.inference.responses import response_input, response_tools, template_messages, assemble_responses, sse_events
from process.app_core.inference.llama_server import LlamaServerProvider


def terminal(text='Hello world', status='completed', extra=()):
    return {'type': 'response.' + status, 'response': {'status': status, 'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}, *extra], 'usage': {'input_tokens': 12}}}


def test_wire_preserves_tool_call_ids_and_separates_definitions():
    calls = [ToolCall('call_1', 'lookup', {'query': 'color'})]
    items = response_input([ChatMessage('user', 'Hi'), ChatMessage('assistant', '', tool_calls=calls),
                            ChatMessage('tool', 'green', tool_call_id='call_1')])
    assert items[1] == {'type': 'function_call', 'call_id': 'call_1', 'name': 'lookup', 'arguments': '{"query": "color"}'}
    assert items[2] == {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'green'}
    assert response_tools([{'type':'function','function':{'name':'lookup','parameters':{'type':'object'}}}]) == [
        {'type':'function','name':'lookup','parameters':{'type':'object'}}]


@pytest.mark.parametrize('text', ['Completed reply.', 'Audible partial reply', ''])
def test_replayed_assistant_text_has_explicit_responses_output_type(text):
    original=ChatMessage('assistant',text)
    assert response_input([original]) == [{'type':'message','role':'assistant',
        'content':[{'type':'output_text','text':text}]}]
    assert original.content == text


def test_interrupted_voice_history_can_be_replayed_with_call_ids_and_metadata():
    from process.app_core.runtime.interjections import Interjections
    interjections=Interjections(lambda:None)
    interjections.transcript('Wait, explain that.',5,1,2)
    history=[ChatMessage('system','Instructions'),ChatMessage('user','First question'),
        *interjections.messages('Hello there.'),ChatMessage('user','Please continue'),
        ChatMessage('system','Runtime observation')]
    items=response_input(template_messages(history))
    assistants=[item for item in items if item.get('role')=='assistant']
    assert [item['content'][0]['text'] for item in assistants] == ['Hello',' there.']
    assert all(item['type']=='message' for item in assistants)
    assert any('[speaking over you]' in item.get('content','') for item in items if item.get('role')=='user')
    assert items[-1]['content'].endswith('Please continue')


def test_qwen_layout_retains_stable_history_and_never_mutates_originals():
    messages = [ChatMessage('system', 'Stable instructions'), ChatMessage('user', 'Earlier'),
        ChatMessage('assistant', 'Past answer'), ChatMessage('system', 'Relevant memories: blue'),
        ChatMessage('user', 'Current request'), ChatMessage('system', 'Current runtime observation: idle')]
    normalized = template_messages(messages)
    assert [m.as_dict() for m in normalized[:3]] == [m.as_dict() for m in messages[:3]]
    assert [m.role for m in normalized] == ['system', 'user', 'assistant', 'user']
    assert 'Relevant memories: blue' in normalized[-1].content
    assert 'Current runtime observation: idle' in normalized[-1].content
    assert normalized[-1].content.endswith('Current request')
    assert messages[-2].content == 'Current request'


def test_qwen_layout_keeps_tool_results_and_merges_leading_systems():
    messages = [ChatMessage('system', 'Instructions'), ChatMessage('system', 'Additional instructions'),
        ChatMessage('user', 'Question'), ChatMessage('assistant', '', tool_calls=[ToolCall('id', 'lookup', {})]),
        ChatMessage('tool', 'Confirmed result', tool_call_id='id'), ChatMessage('system', 'Runtime metadata')]
    items = response_input(template_messages(messages))
    assert items[0]['content'] == 'Instructions\n\nAdditional instructions'
    assert items[-1]['type'] == 'function_call_output' and items[-1]['call_id'] == 'id'
    assert items[-1]['output'].endswith('Confirmed result')


def test_responses_text_reasoning_and_tools_are_separate_and_word_streamed():
    text, reasoning = [], []
    result = assemble_responses([
        {'type':'response.reasoning_text.delta','delta':'Explicit provider reasoning'},
        {'type':'response.output_text.delta','delta':'Hel'},
        {'type':'response.output_text.delta','delta':'lo wor'},
        {'type':'response.function_call_arguments.delta','delta':'{"query":'},
        {'type':'response.output_text.delta','delta':'ld'},
        terminal(extra=[{'type':'function_call','call_id':'id','name':'lookup','arguments':'{"query":"color"}'}]),
    ], text.append, on_reasoning=reasoning.append)
    assert text == ['Hello ', 'world']
    assert reasoning == ['Explicit provider reasoning']
    assert result.message.content == 'Hello world'
    assert result.message.tool_calls[0].arguments == {'query':'color'}
    assert result.finish_reason == 'tool_calls' and result.usage['input_tokens'] == 12


@pytest.mark.parametrize('ending', [[], [{'type':'error','error':{'message':'bad template'}}],
                                   [{'type':'response.failed','response':{'error':{'message':'failed'}}}]])
def test_unfinished_stream_never_flushes_tail(ending):
    text=[]
    with pytest.raises(RuntimeError):
        assemble_responses([{'type':'response.output_text.delta','delta':'Safe unfinished'}, *ending], text.append)
    assert text == ['Safe ']


def test_token_limit_completion_and_standard_sse_comments_and_named_events():
    event = terminal('Short', 'incomplete')
    lines = [': keepalive', '', 'event: response.incomplete', 'data: ' + json.dumps({'response':event['response']}), '']
    events = list(sse_events(lines))
    result = assemble_responses(events, lambda text: None)
    assert result.message.content == 'Short' and result.finish_reason == 'length'


@pytest.mark.parametrize('status,message', [(500,'System message must be at the beginning'), (404,'Not found')])
def test_server_errors_show_native_cause_and_missing_endpoint_never_falls_back(status,message):
    response = httpx.Response(status, json={'error':{'message':message}})
    with pytest.raises(RuntimeError, match=message): LlamaServerProvider._check_response(response)
