from types import SimpleNamespace
from pathlib import Path

import pytest

from process.app_core.runtime.cancellation import TurnCancelled
from process.app_core.events.bus import RuntimeEvent
from process.app_core.conversation.messages import ChatMessage, ModelResponse
from process.app_core.runtime.session import SessionManager
from process.app_core.desktop.state import DesktopState


class FakeSpeech:
    def __init__(self, *args): self.items, self.cancelled = [], False
    def submit(self, *args): self.items.append(args)
    def cancel(self): self.cancelled = True
    def close(self): pass


def make_session(monkeypatch, chat):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', FakeSpeech)
    config = SimpleNamespace(raw={}, root=Path('.'), character_name='Riko', tools=SimpleNamespace(max_iterations=8))
    return SessionManager(config, chat, DesktopState())


def test_cancel_keeps_capture_and_trims_at_current_playback(monkeypatch):
    chat = SimpleNamespace(history=[], _save_history=lambda: None, provider=SimpleNamespace(close=lambda: None))
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
    assert chat.history[-1].content == 'One two'
    assert session.speech.cancelled
    session.voice_transcript('Actually Tuesday', 10, 12, (session._active_turn, 0))
    assert chat.history[-1].content == '[speaking over you] Actually Tuesday'
    assert 'Six seven' not in str(chat.history)
    session.voice = None
    session.close()


def test_interjection_rewrite_preserves_following_history(monkeypatch):
    chat = SimpleNamespace(history=[], _save_history=lambda: None, provider=SimpleNamespace(close=lambda: None))
    session = make_session(monkeypatch, chat)
    def respond(text, user, **kwargs):
        kwargs['on_delta']('One two three four five. ')
        chat.history.extend([ChatMessage('user', text), *kwargs['response_history']('One two three four five. ')])
        return ModelResponse(ChatMessage('assistant', 'One two three four five. '))
    chat.respond = respond
    session.respond('hello')
    chat.history.append(ChatMessage('system', 'unrelated later entry'))
    session.voice_transcript('Tuesday', 1, 1.2, (session._active_turn, 8))
    session.voice_transcript('Not Monday', 1.5, 1.8, (session._active_turn, 10))
    assert chat.history[-1].content == 'unrelated later entry'
    users = [m.content for m in chat.history if m.role == 'user']
    assert users == ['hello', '[speaking over you] Tuesday Not Monday']
    session.close()


def test_speech_during_reasoning_extends_original_input_without_annotation(monkeypatch):
    chat = SimpleNamespace(history=[], _save_history=lambda: None, provider=SimpleNamespace(close=lambda: None))
    session = make_session(monkeypatch, chat)
    def respond(text, user, **kwargs):
        kwargs['on_reasoning']('Thinking about the answer')
        anchor = session.voice_anchor()
        assert not session.voice_speaking_over(anchor)
        assert session.voice_transcript('And tomorrow too', 1, 2, anchor)
        raise TurnCancelled()
    chat.respond = respond
    with pytest.raises(TurnCancelled): session.respond('What about today?')
    assert len(chat.history) == 1
    assert chat.history[0].content == 'User: What about today?\nAnd tomorrow too'
    assert not session._interjections.items
    session.close()


def test_visible_text_counts_as_speaking_over_even_before_playback(monkeypatch):
    chat = SimpleNamespace(history=[], _save_history=lambda: None, provider=SimpleNamespace(close=lambda: None))
    session = make_session(monkeypatch, chat)
    def respond(text, user, **kwargs):
        kwargs['on_delta']('Visible answer')
        anchor = session.voice_anchor()
        assert session.voice_speaking_over(anchor)
        session.voice_transcript('One more thing', 1, 2, anchor)
        chat.history.extend([ChatMessage('user', text), *kwargs['response_history']('Visible answer')])
        return ModelResponse(ChatMessage('assistant', 'Visible answer'))
    chat.respond = respond
    session.respond('hello', speak=False)
    assert any(m.content == '[speaking over you] One more thing' for m in chat.history)
    session.close()
