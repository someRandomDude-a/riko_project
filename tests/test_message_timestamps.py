import json
from datetime import datetime, timedelta, timezone

from process.app_core.kernel.messages import ChatMessage, ModelResponse, conversation_sections
from process.app_core.conversation.chat import ChatService


def messages_at(seconds):
    base = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)
    return [ChatMessage('user' if i % 2 == 0 else 'assistant', f'turn {i}',
                        timestamp=(base + timedelta(seconds=value)).isoformat())
            for i, value in enumerate(seconds)]


def test_sections_use_adjacent_gaps_not_total_section_duration():
    messages = messages_at([0, 100, 350, 600, 900, 1201])
    rendered = conversation_sections(messages)
    assert [i for i, m in enumerate(rendered) if m.content.startswith('[')] == [0, 4, 5]
    assert all(m.content == f'turn {i}' for i, m in enumerate(messages))
    assert 'timestamp' not in messages[0].as_dict()
    assert messages[0].as_record()['timestamp']


def test_undated_history_and_tools():
    dated = messages_at([0, 301])
    rendered = conversation_sections([ChatMessage('user', 'old', timestamp=None), dated[0],
                                      ChatMessage('tool', 'result'), dated[1]])
    assert 'timestamp unavailable' in rendered[0].content
    assert rendered[1].content.startswith('[')
    assert rendered[2].content == 'result'
    assert rendered[3].content.startswith('[')


def test_timestamps_use_system_timezone_and_convert_legacy_utc():
    message = ChatMessage('user', 'Hello')
    parsed = datetime.fromisoformat(message.timestamp)
    assert parsed.utcoffset() == parsed.astimezone().utcoffset()
    legacy = messages_at([0])[0]
    expected = datetime.fromisoformat(legacy.timestamp).astimezone().strftime('%Y-%m-%dT%H:%M')
    assert conversation_sections([legacy])[0].content == f'[{expected}]\nturn 0'


def test_history_round_trip_and_prompt_timestamps(tmp_path):
    path = tmp_path / 'history.json'
    history = messages_at([0, 100])
    path.write_text(json.dumps([m.as_record() for m in history]), encoding='utf-8')
    class Provider:
        def generate(self, messages, **options):
            self.messages = list(messages)
            return ModelResponse(ChatMessage('assistant', 'Hello'))
    provider = Provider()
    chat = ChatService(provider, system_prompt='Test', history_file=path)
    chat.respond('New question')
    assert provider.messages[1].content.startswith('[')
    assert provider.messages[-1].content.startswith('[')
    assert not chat.history[-2].content.startswith('[')
    reloaded = ChatService(provider, system_prompt='Test', history_file=path)
    assert reloaded.history[0].timestamp == history[0].timestamp
    assert all(m.timestamp for m in reloaded.history)
