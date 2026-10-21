from types import SimpleNamespace

import pytest

from process.app_core.kernel.output_filter import OutputFilter, clean_output
from process.app_core.conversation.chat import ChatService
from process.app_core.kernel.messages import ChatMessage, ModelResponse


@pytest.mark.parametrize('stamp', ['[2026-10-01T14:30]', '[2026-10-01 14:30:10+05:30]',
    '[2026-10-01T14:30:10.123456+00:00]', '[timestamp unavailable]', '[YYYY-MM-DDTHH:MM]',
    '2026-10-01T14:30:10Z', '2026-10-01T14:30:10.123456-04:00'])
def test_echoed_stamps_removed_at_every_stream_split(stamp):
    value = 'Hello ' + stamp + '\n**World** at 14:30 on 2026-10-01.'
    expected = 'Hello \n**World** at 14:30 on 2026-10-01.'
    assert clean_output(value) == expected
    for split in range(len(value)+1):
        received=[]
        stream=OutputFilter(received.append)
        stream.feed(value[:split]); stream.feed(value[split:]); stream.finish()
        assert ''.join(received) == expected
    received=[]
    stream=OutputFilter(received.append)
    for char in value: stream.feed(char)
    stream.finish()
    assert ''.join(received) == expected


def test_normal_markdown_dates_numbers_and_ambiguous_prefixes_are_preserved():
    value='[Link](https://example.com) [2026 budget] [2026] 2026-10-01 14:30 is a date-time stamp.\n`x[2]` $x^2$ 1234.56 [unfinished'
    assert clean_output(value) == value.replace('2026-10-01 14:30', '')
    assert clean_output('Meet on 2026-10-01 at 14:30; [2026') == 'Meet on 2026-10-01 at 14:30; [2026'


def test_stream_holds_timestamp_fragments_before_any_consumer_sees_them():
    received=[]
    stream=OutputFilter(received.append)
    stream.feed('Hello [2026-10-01T')
    assert received == ['Hello ']
    stream.feed('14:30] welcome back')
    assert ''.join(received) == 'Hello  welcome '
    stream.finish()
    assert ''.join(received) == 'Hello  welcome back'


@pytest.mark.parametrize('streamed', [False, True])
def test_chat_cleans_visible_output_and_history_but_retains_raw_provenance(tmp_path, streamed):
    text='[2026-10-01T14:30]\nHello [timestamp unavailable]there.'
    raw={'original':text}
    def generate(messages, **options):
        if options.get('on_delta'):
            for char in text: options['on_delta'](char)
        return ModelResponse(ChatMessage('assistant',text),raw=raw)
    chat=ChatService(SimpleNamespace(generate=generate),system_prompt='Test',history_file=tmp_path/'history.json')
    deltas=[]
    response=chat.respond('Hello',on_delta=deltas.append if streamed else None)
    assert response.message.content == '\nHello there.'
    assert chat.history[-1].content == '\nHello there.'
    assert chat.history[0].content == 'User: Hello'
    assert response.raw is raw and raw['original'] == text
    if streamed: assert ''.join(deltas) == response.message.content


def test_legacy_chat_stream_is_filtered_too():
    chat=ChatService(SimpleNamespace(stream=lambda *args,**kwargs:iter(['[2026-', '10-01T14:30] Hello ', 'world'])),system_prompt='Test')
    assert ''.join(chat.stream_respond('Hi')) == ' Hello world'
    assert chat.history[-1].content == ' Hello world'


def test_transport_failure_does_not_flush_held_metadata_or_save_reply():
    def generate(messages, **options):
        options['on_delta']('Hello [2026-10-01T')
        raise RuntimeError('transport failed')
    chat=ChatService(SimpleNamespace(generate=generate),system_prompt='Test')
    deltas=[]
    with pytest.raises(RuntimeError,match='transport failed'): chat.respond('Hi',on_delta=deltas.append)
    assert deltas == ['Hello ']
    assert not chat.history
