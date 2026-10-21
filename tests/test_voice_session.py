from types import SimpleNamespace
from pathlib import Path

import pytest

from process.app_core.kernel.cancellation import TurnCancelled
from process.app_core.events.bus import RuntimeEvent
from process.app_core.conversation.chat import ChatDeps
from process.app_core.conversation.history import ConversationHistory
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.runtime.session import SessionManager
from process.app_core.desktop.state import DesktopState
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths


class FakeSpeech:
    def __init__(self, *args): self.items, self.cancelled = [], False
    def submit(self, *args): self.items.append(args)
    def cancel(self): self.cancelled = True
    def close(self): pass


def make_session(monkeypatch, chat):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', FakeSpeech)
    config = SimpleNamespace(raw={}, root=Path('.'), paths=DataPaths.at(Path('.')), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    return SessionManager(config, chat, DesktopState())


def test_cancel_keeps_capture_and_trims_at_current_playback(monkeypatch):
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), provider=BaseProvider())
    session = make_session(monkeypatch, chat)
    capture = SimpleNamespace(closed=SimpleNamespace(is_set=lambda: False))
    session.voice = capture
    def respond(text, user, **kwargs):
        kwargs['on_delta']('One two three four five. Six seven eight nine ten. ')
        session._playback_event(RuntimeEvent('speech.started', {'text': 'One two three four five.',
            'start_offset': 0, 'end_offset': 24, 'started_at': 100}, turn_id=session._active_turn))
        monkeypatch.setattr('process.app_core.runtime.session.time.monotonic', lambda: 100.9)
        session.cancel()
        raise TurnCancelled()
    chat.respond = respond
    with pytest.raises(TurnCancelled): session.respond('hello')
    assert session.voice is capture
    assert session._cutoff == len('One two')
    assert chat.conversation.snapshot()[-1].content == 'One two'
    assert session.speech.cancelled
    session.voice_transcript('Actually Tuesday', 10, 12, (session._active_turn, 0))
    assert chat.conversation.snapshot()[-1].content == '[speaking over you] Actually Tuesday'
    assert 'Six seven' not in str(chat.conversation.snapshot())
    session.voice = None
    session.close()


def test_interjection_rewrite_preserves_following_history(monkeypatch):
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), provider=BaseProvider())
    session = make_session(monkeypatch, chat)
    def respond(text, user, **kwargs):
        kwargs['on_delta']('One two three four five. ')
        chat.conversation.append([ChatMessage('user', text), *kwargs['response_history']('One two three four five. ')])
        return ModelResponse(ChatMessage('assistant', 'One two three four five. '))
    chat.respond = respond
    session.respond('hello')
    chat.conversation.append([ChatMessage('system', 'unrelated later entry')])
    session.voice_transcript('Tuesday', 1, 1.2, (session._active_turn, 8))
    session.voice_transcript('Not Monday', 1.5, 1.8, (session._active_turn, 10))
    assert chat.conversation.snapshot()[-1].content == 'unrelated later entry'
    users = [m.content for m in chat.conversation.snapshot() if m.role == 'user']
    assert users == ['hello', '[speaking over you] Tuesday Not Monday']
    session.close()


def test_speech_during_reasoning_extends_original_input_without_annotation(monkeypatch):
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), provider=BaseProvider())
    session = make_session(monkeypatch, chat)
    def respond(text, user, **kwargs):
        kwargs['on_reasoning']('Thinking about the answer')
        anchor = session.voice_anchor()
        assert not session.voice_speaking_over(anchor)
        assert session.voice_transcript('And tomorrow too', 1, 2, anchor) == 'reply'  # the cut reply is redone
        raise TurnCancelled()
    chat.respond = respond
    with pytest.raises(TurnCancelled): session.respond('What about today?')
    assert len(chat.conversation) == 1
    assert chat.conversation.snapshot()[0].content == 'User: What about today?\nAnd tomorrow too'
    assert not session._interjections.items
    session.close()


def test_stop_during_reasoning_ends_the_turn_at_the_next_reasoning_delta(monkeypatch):
    from process.app_core.events.bus import event_bus
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), close=lambda: None, provider=BaseProvider())  # its cancel() is a no-op: Stop still ends the turn
    session = make_session(monkeypatch, chat)
    shown, after_stop = [], []
    unsubscribe = event_bus.subscribe(lambda event: shown.append(event.payload['text']) if event.type == 'model.reasoning' else None)
    def respond(text, user, **kwargs):
        kwargs['on_reasoning']('Thinking')
        session.cancel()  # Stop while the model is still reasoning, before any visible word
        try: kwargs['on_reasoning']('still thinking')  # a provider streaming reasoning stops right here
        except TurnCancelled: after_stop.append('raised'); raise
        after_stop.append('kept reasoning')
        return ModelResponse(ChatMessage('assistant', 'Late answer'))
    chat.respond = respond
    try:
        with pytest.raises(TurnCancelled): session.respond('Hard question')
        assert after_stop == ['raised'] and shown == ['Thinking']
    finally: unsubscribe(); session.close()


def test_visible_text_counts_as_speaking_over_even_before_playback(monkeypatch):
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), provider=BaseProvider())
    session = make_session(monkeypatch, chat)
    def respond(text, user, **kwargs):
        kwargs['on_delta']('Visible answer')
        anchor = session.voice_anchor()
        assert session.voice_speaking_over(anchor)
        assert session.voice_transcript('One more thing', 1, 2, anchor) == 'preserved'
        chat.conversation.append([ChatMessage('user', text), *kwargs['response_history']('Visible answer')])
        return ModelResponse(ChatMessage('assistant', 'Visible answer'))
    chat.respond = respond
    session.respond('hello')  # nothing has played yet (FakeSpeech queues nothing)
    assert any(m.content == '[speaking over you] One more thing' for m in chat.conversation.snapshot())
    session.close()
