import json
import pytest
from process.app_core.inference.context_budget import pack_context
from process.app_core.conversation.messages import ChatMessage, ToolCall


def count(messages):
    return sum(len(m.content) + 4 for m in messages)


def test_keeps_system_current_input_and_tool_pairs_without_mutating_history():
    messages = [ChatMessage('system', 'identity'), ChatMessage('user', 'old' * 100),
        ChatMessage('assistant', '', tool_calls=[ToolCall('old', 'tool')]),
        ChatMessage('tool', 'past result', tool_call_id='old'), ChatMessage('user', 'current'),
        ChatMessage('assistant', '', tool_calls=[ToolCall('new', 'tool')]),
        ChatMessage('tool', 'current result', tool_call_id='new')]
    packed = pack_context(messages, count, 100, 20)
    assert count(packed) + 20 <= 100
    assert [m.content for m in packed] == ['identity', 'current', '', 'current result']
    assert packed[-2].tool_calls[0].id == packed[-1].tool_call_id
    assert len(messages) == 7


@pytest.mark.parametrize('size', [10, 100])
def test_initiative_recent_history_count_is_budget_dependent(size):
    data = {'emotion': {'primary': 'joy'}, 'recent_messages': [{'role': 'user', 'content': str(i) + 'x' * size} for i in range(20)]}
    original = ChatMessage('user', json.dumps(data), context_kind='initiative')
    messages = [ChatMessage('system', 'character'), original]
    packed = pack_context(messages, count, 700, 100)
    recent = json.loads(packed[1].content)['recent_messages']
    assert 0 < len(recent) < 20
    assert recent[-1]['content'].startswith('19')
    assert json.loads(original.content) == data
    assert count(packed) + 100 <= 700


def test_reflection_removes_optional_evidence_before_focal_text():
    data = {'focal_id': 'focal', 'evidence': [
        {'id': 'focal', 'text': 'fact', 'formation_context': {'history': ['x' * 300] * 10}},
        {'id': 'related', 'text': 'x' * 1000}]}
    packed = pack_context([ChatMessage('system', 'instructions'), ChatMessage('user', json.dumps(data), context_kind='reflection')], count, 500, 100)
    retained = json.loads(packed[1].content)
    assert retained['focal_id'] == 'focal'
    assert [e['id'] for e in retained['evidence']] == ['focal']
    assert retained['evidence'][0]['text'] == 'fact'
    assert count(packed) + 100 <= 500


def test_required_content_and_tools_never_silently_truncated():
    with pytest.raises(ValueError, match='inference not started'):
        pack_context([ChatMessage('system', 'x' * 1000), ChatMessage('user', 'current')], count, 500, 100)
    with pytest.raises(RuntimeError, match='cancelled'):
        pack_context([ChatMessage('user', 'hi')], count, 500, 100, cancelled=lambda: True)


def test_large_history_uses_logarithmic_tokenizer_calls():
    calls = []
    def counter(messages): calls.append(True); return count(messages)
    packed = pack_context([ChatMessage('system', 'identity'), *[ChatMessage('user', 'x' * 20) for _ in range(1000)]], counter, 500, 100)
    assert count(packed) + 100 <= 500
    assert len(calls) < 20
