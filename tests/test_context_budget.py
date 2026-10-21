import json
import pytest
from process.app_core.inference.context_budget import pack_context
from process.app_core.kernel.messages import ChatMessage, ToolCall


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


def exchanges(start, stop, size=60):
    return [m for i in range(start, stop) for m in (ChatMessage('user', f'U{i} ' + 'u' * size), ChatMessage('assistant', f'A{i} ' + 'a' * size))]


def counting(sizes):
    def counter(messages): sizes.append(count(messages)); return count(messages)
    return counter


def conversation(state):
    history, firsts, calls = exchanges(0, 40), [], []
    for turn in range(40, 80):
        sizes = []
        messages = [ChatMessage('system', 'identity'), *history, ChatMessage('system', 'recall', context_kind='optional'),
                    ChatMessage('user', f'U{turn} ' + 'u' * 60)]
        packed = pack_context(messages, counting(sizes), 2048, 256, state=state)
        assert count(packed) + 256 <= 2048 and packed[-1].content.startswith(f'U{turn}') and packed[-2].content == 'recall'
        firsts.append(packed[1].content.split()[0])
        calls.append(len(sizes))
        history += [messages[-1], ChatMessage('assistant', f'A{turn} ' + 'a' * 60)]
    return [turn for turn in range(1, 40) if firsts[turn] != firsts[turn - 1]], calls


def test_prefix_stays_fixed_between_overflows_with_one_count_per_turn():
    assert len(conversation(None)[0]) == 39  # stateless packing slides the prefix every turn
    changes, calls = conversation({})
    # Trims land a quarter of the history room below the limit, so the prefix changes only every few turns.
    assert 0 < len(changes) <= 10 and all(b - a >= 3 for a, b in zip(changes, changes[1:]))
    assert all(calls[turn] == 1 for turn in range(1, 40) if turn not in changes)
    assert max(calls) < 12


@pytest.mark.parametrize('state', [None, {}])
def test_long_history_is_measured_from_the_newest_turns(state):
    sizes = []
    messages = [ChatMessage('system', 'identity'), *exchanges(0, 10000), ChatMessage('user', 'current')]
    packed = pack_context(messages, counting(sizes), 2048, 256, state=state)
    assert count(packed) + 256 <= 2048 and len(packed) > 10
    assert len(sizes) < 25 and max(sizes) < 4 * 2048  # never the 1.3M-character whole


def test_window_reexpands_after_a_turn_that_needed_the_room():
    state, history = {}, exchanges(0, 30)
    def turn(*extra):
        messages = [ChatMessage('system', 'identity'), *history, ChatMessage('user', 'current'), *extra]
        return pack_context(messages, count, 2048, 256, state=state)
    normal = turn()
    spike = turn(ChatMessage('assistant', '', tool_calls=[ToolCall('call', 'tool')]), ChatMessage('tool', 'r' * 1200, tool_call_id='call'))
    assert len(spike) <= len(normal) // 2
    history += exchanges(30, 31)
    assert len(turn()) >= len(normal)


def test_oversized_current_tool_output_is_cut_with_a_marker_instead_of_failing():
    events = []
    from process.app_core.events.bus import event_bus
    unsubscribe = event_bus.subscribe(lambda event: events.append(event) if event.type == 'context.trimmed' else None)
    output = 'HEAD' + 'x' * 100000 + 'TAIL'
    messages = [ChatMessage('system', 'identity'), *exchanges(0, 5), ChatMessage('user', 'summarise report.pdf'),
        ChatMessage('assistant', '', tool_calls=[ToolCall('call', 'pdf_processor')]), ChatMessage('tool', output, tool_call_id='call'),
        ChatMessage('system', 'observation', context_kind='optional')]
    try: packed = pack_context(messages, count, 8192, 1024)
    finally: unsubscribe()
    assert count(packed) + 1024 <= 8192 and messages[-2].content == output
    tool = packed[-1]
    assert tool.role == 'tool' and tool.content.startswith('HEAD') and tool.content.endswith('TAIL') and len(tool.content) > 6000
    assert '[tool output truncated to fit the context: ' in tool.content
    assert events[-1].payload['tool_output_chars'] > 6000
    with pytest.raises(ValueError, match='inference not started'):
        pack_context([ChatMessage('system', 'x' * 2000), ChatMessage('user', 'hi'), ChatMessage('tool', output, tool_call_id='call')], count, 500, 100)


def test_recall_inside_the_newest_past_turn_outlives_that_turn():
    messages = [ChatMessage('system', 'identity'), ChatMessage('user', 'old ' + 'o' * 300), ChatMessage('system', 'recall', context_kind='optional'),
                ChatMessage('user', 'current')]
    assert [m.content for m in pack_context(messages, count, 200, 100)] == ['identity', 'recall', 'current']


# The largest count among the Qwen3.5, Qwen3, Llama 3.2, Gemma 3, Mistral Nemo and DeepSeek V3 tokenizers.
TOKENIZED = [
    ("Hey Riko, I need to finish the quarterly report by Friday but I keep procrastinating. Can you remind me after dinner? You're the best, thanks!", 36),
    (json.dumps({'type': 'function', 'function': {'name': 'task_update', 'description': 'Update one task; pass the revision you last read.', 'parameters': {'type': 'object',
        'properties': {'task_id': {'type': 'string'}, 'expected_revision': {'type': 'integer', 'minimum': 1}}, 'required': ['task_id', 'expected_revision']}}}), 86),
    ('6f9619ff-8b86-d011-b42d-00c04fc964ff 3e1a52c4-7b9d-4f0e-9a61-2d8c5b7e0f13 call_9b2e4d7a1c6f0835e2a4b7c9', 96),
    ('2026-10-08T21:14:03+02:00 observed at 1791400000.123, 4096 tokens, 1200000 items', 59),
    ("def pack(messages, limit):\n    for i, message in enumerate(messages):\n        if message.role == 'system' and i > 0:\n"
     "            raise ValueError('System message must be at the beginning.')\n    return messages[-limit:]\n", 55),
    ('お疲れさま！今日は何がそんなに大変だったの？少し休憩しようね。明日の朝九時の歯医者だね。', 35),
    ('今天天气很好，我们去公园散步吧。我想买一杯咖啡，然后看看书。', 29),
    ('안녕하세요, 오늘 날씨가 정말 좋네요. 같이 산책하러 갈까요?', 27),
    ('Привет! Как дела? Сегодня отличная погода, давай пойдём гулять в парк.', 28),
    ('Good morning ☀️ I made pancakes 🥞🍓 and coffee ☕ 😊👍', 29)]


@pytest.mark.parametrize('text,tokens', TOKENIZED)
def test_remote_estimate_tracks_real_tokenizers_instead_of_bytes(text, tokens):
    from process.app_core.inference.context_budget import estimate_text_tokens
    assert .9 * tokens <= estimate_text_tokens(text) <= 1.6 * tokens


def test_remote_estimate_counts_framing_tool_calls_and_schemas():
    from process.app_core.inference.context_budget import estimate_text_tokens, estimate_tokens
    tools = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}]
    plain = [ChatMessage('system', 'Instructions'), ChatMessage('user', 'Hi')]
    called = [*plain, ChatMessage('assistant', '', tool_calls=[ToolCall('call_1', 'lookup', {'query': 'tea'})]), ChatMessage('tool', 'green', tool_call_id='call_1')]
    assert estimate_tokens(plain) == 64 + 2 * 8 + estimate_text_tokens('Instructions') + estimate_text_tokens('Hi')
    assert estimate_tokens(called) > estimate_tokens(plain) + 2 * 8 + 24
    assert estimate_tokens(plain, tools) >= estimate_tokens(plain) + 64 + 1.25 * estimate_text_tokens(json.dumps(tools))


def entropy_samples():
    """Text that splits into far smaller pieces than words do; counts measured with the Qwen2.5 tokenizer."""
    import base64, random, string
    rng = random.Random(7)
    encoded = base64.b64encode(bytes(rng.getrandbits(8) for _ in range(300))).decode()
    key = 'sk-' + ''.join(rng.choice(string.ascii_letters + string.digits) for _ in range(48))
    letters = ' '.join(''.join(rng.choice(string.ascii_lowercase) for _ in range(10)) for _ in range(12))
    return [('WHY DID YOU NOT REMIND ME ABOUT THE DENTIST APPOINTMENT THIS MORNING? I MISSED IT AGAIN!', 26), (encoded, 296),
        ('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ'
         '.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c', 109), (key, 42), (letters, 65)]


@pytest.mark.parametrize('index', range(5))
def test_remote_estimate_does_not_undercount_base64_keys_capitals_or_random_letters(index):
    # Under-counting lets a prompt the estimate says fits overflow the server; UTF-8 bytes never under-counted.
    from process.app_core.inference.context_budget import estimate_text_tokens
    text, tokens = entropy_samples()[index]
    assert .9 * tokens <= estimate_text_tokens(text) <= 1.6 * tokens


def test_cold_state_keeps_a_history_that_already_fits():
    # A restart (or an edited anchor message) must not trim history that fits to the post-overflow target.
    messages = [ChatMessage('system', 'identity'), *[ChatMessage(role, f'{role} {i} ' + 'x' * 60) for i in range(20) for role in ('user', 'assistant')],
        ChatMessage('user', 'current')]
    assert count(messages) + 100 < 3200
    assert len(pack_context(messages, count, 3200, 100, state={})) == len(messages)
