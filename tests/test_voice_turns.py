"""One voice-turn contract: SessionManager.voice_transcript says what each final transcript needs ('preserved',
'reply', 'fresh' or 'ignored') and VoiceInput acts on it, waiting for the turn gate rather than losing the words.
Every step runs on the test thread (or behind explicit events), so nothing depends on timing."""
import queue
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from process.app_core.audio.voice_input import VoiceInput
from process.app_core.audio.voice_segments import Segment
from process.app_core.conversation.chat import ChatDeps
from process.app_core.conversation.history import ConversationHistory
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.desktop.state import DesktopState
from process.app_core.events.bus import event_bus
from process.app_core.kernel.cancellation import TurnCancelled
from process.app_core.runtime.session import SessionManager
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths

DISCORD = {'source': 'discord', 'conversation_id': 'discord:client:dm:1', 'user_id': '1', 'channel_id': '1', 'message_id': 'd1'}


class Speech:
    """Muted output: nothing is queued, so a reply ends when generation ends."""
    def __init__(self, *args): pass
    def submit(self, *args): return False
    def cancel(self): pass
    def close(self): pass


@pytest.fixture
def parts(monkeypatch, tmp_path):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), calls=[], provider=BaseProvider())
    config = SimpleNamespace(raw={'animation': {'enabled': False}}, root=tmp_path, paths=DataPaths.at(tmp_path), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    session = SessionManager(config, chat, DesktopState())
    events = []
    unsubscribe = event_bus.subscribe(events.append)
    try: yield session, chat, events
    finally:
        unsubscribe()
        session.close()


def commit(chat, text, user_name, kwargs, reply):
    """ChatService's contract: the user message (when recorded) and the reply are committed on completion."""
    chat.calls.append((text, kwargs['record_user']))
    if kwargs['cancelled'](): raise TurnCancelled()
    kwargs['on_delta'](reply)
    chat.conversation.append([*([ChatMessage('user', f'{user_name}: {text}')] if kwargs['record_user'] else []), *kwargs['response_history'](reply)])
    return ModelResponse(ChatMessage('assistant', reply))


def answer(chat, reply, before=None):
    def respond(text, user_name, **kwargs):
        if before: before(kwargs)
        return commit(chat, text, user_name, kwargs, reply)
    return respond


def contents(chat): return [(message.role, message.content) for message in chat.conversation.snapshot()]


def voice_for(session, transcripts):
    """A VoiceInput whose ASR returns the given texts in order; dispatches are collected, then run by run_dispatches."""
    voice = VoiceInput.__new__(VoiceInput)
    voice.session, voice.closed, voice._parts = session, threading.Event(), {}
    voice._partial_lock, voice._partial_pending = threading.Lock(), set()
    texts = iter(transcripts)
    session.asr.model = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter([SimpleNamespace(text=next(texts))]), None))
    voice.dispatched = []
    voice.responses = SimpleNamespace(submit=lambda function, *args: voice.dispatched.append((function, args)))
    return voice


def transcribe(voice, anchor):
    """Run one final utterance through the ASR worker on this thread."""
    class Jobs(queue.Queue):
        def task_done(self):
            super().task_done()
            voice.closed.set()
    voice.closed.clear()
    voice.jobs = Jobs()
    voice.jobs.put(Segment(str(uuid.uuid4()), bytes(1024), anchor, 1.0, 2.0, 1.0, True))
    voice._asr()


def run_dispatches(voice):
    voice.closed.clear()
    pending, voice.dispatched = voice.dispatched, []
    for function, args in pending: function(*args)


def test_a_short_continuation_during_reasoning_redoes_the_reply_instead_of_going_silent(parts):
    session, chat, events = parts
    voice = voice_for(session, ['and tomorrow?'])
    def reasoning(text, user_name, **kwargs):
        # The user resumes after a pause while the model is still reasoning; the fragment is short, so VAD never
        # reaches interruption_seconds. The stale reply is cancelled to answer the combined question.
        transcribe(voice, session.voice_anchor())
        assert kwargs['cancelled']()
        raise TurnCancelled()
    chat.respond = reasoning
    with pytest.raises(TurnCancelled): session.respond("What's the weather today?")
    chat.respond = answer(chat, 'Sunny today, rain tomorrow.')
    run_dispatches(voice)
    assert contents(chat) == [('user', "User: What's the weather today?\nand tomorrow?"), ('assistant', 'Sunny today, rain tomorrow.')]
    assert chat.calls == [('and tomorrow?', False)]  # the merged input is already in history
    assert 'voice.error' not in [event.type for event in events]


def test_sustained_speech_during_reasoning_redoes_the_reply_with_the_whole_question(parts):
    session, chat, _ = parts
    voice = voice_for(session, ['and tomorrow, and the weekend'])
    anchors = []
    def reasoning(text, user_name, **kwargs):
        anchors.append(session.voice_anchor())
        session.voice_activity(2.0, anchors[0])  # the user keeps talking: VAD cuts the reply
        assert kwargs['cancelled']()
        raise TurnCancelled()
    chat.respond = reasoning
    with pytest.raises(TurnCancelled): session.respond("What's the weather today?")
    transcribe(voice, anchors[0])  # the final ASR arrives after the cut turn unwound
    chat.respond = answer(chat, 'Sunny, then rain all weekend.')
    run_dispatches(voice)
    assert contents(chat) == [('user', "User: What's the weather today?\nand tomorrow, and the weekend"), ('assistant', 'Sunny, then rain all weekend.')]
    assert chat.calls == [('and tomorrow, and the weekend', False)]


def test_words_after_a_stopped_reply_are_never_merged_into_its_question(parts):
    session, chat, _ = parts
    anchors = []
    def stopped(text, user_name, **kwargs):
        anchors.append(session.voice_anchor())
        session.cancel()  # the Stop button, while the model reasons
        raise TurnCancelled()
    chat.respond = stopped
    with pytest.raises(TurnCancelled): session.respond('What about today?')
    assert session.voice_transcript('and tomorrow', 1, 2, anchors[0]) == 'fresh'
    assert contents(chat) == [('user', 'User: What about today?')]


def test_a_continuation_transcribed_after_the_reply_finished_is_a_new_turn(parts):
    session, chat, _ = parts
    voice = voice_for(session, ['and what about tomorrow'])
    anchors = []
    chat.respond = answer(chat, 'Today is sunny.', before=lambda kwargs: anchors.append(session.voice_anchor()))
    session.respond('what about today?')  # the second fragment started while the model was reasoning...
    transcribe(voice, anchors[0])  # ...but its final ASR arrives after the reply completed
    chat.respond = answer(chat, 'Tomorrow it rains.')
    run_dispatches(voice)
    assert contents(chat) == [('user', 'User: what about today?'), ('assistant', 'Today is sunny.'),
                              ('user', 'User: and what about tomorrow'), ('assistant', 'Tomorrow it rains.')]


def test_speech_during_a_redone_reply_never_rewrites_an_older_user_message(parts):
    session, chat, events = parts
    session.state.toggle_audio()  # muted: the cut keeps the text the user read
    voice = voice_for(session, ['Wait, Tuesday', 'and bring snacks'])
    def first(text, user_name, **kwargs):
        kwargs['on_delta']('Monday works. ')
        anchor = session.voice_anchor()
        session.voice_activity(2.0, anchor)  # sustained speech over the visible reply cuts it
        transcribe(voice, anchor)
        raise TurnCancelled()
    chat.respond = first
    with pytest.raises(TurnCancelled): session.respond('Can we meet?')
    first_turn = session._active_turn
    # The redo (record_user=False) is still reasoning when the user adds something short.
    chat.respond = answer(chat, 'Tuesday it is.', before=lambda kwargs: transcribe(voice, session.voice_anchor()))
    run_dispatches(voice)
    chat.respond = answer(chat, 'Snacks noted.')
    run_dispatches(voice)
    assert contents(chat) == [('user', 'User: Can we meet?'), ('assistant', 'Monday works. '), ('user', '[speaking over you] Wait, Tuesday'),
                              ('assistant', 'Tuesday it is.'), ('user', 'User: and bring snacks'), ('assistant', 'Snacks noted.')]
    assert chat.calls == [('Wait, Tuesday', False), ('and bring snacks', True)]
    assert not [event for event in events if event.type == 'chat.input' and event.turn_id == first_turn and 'snacks' in event.payload['text']]


def test_an_initiative_resets_the_continuation_target(parts):
    session, chat, _ = parts
    chat.respond = answer(chat, 'Hi!')
    session.respond('hello')
    assert session.present_initiative('Want to take a break?', spoken=True)
    assert session._input_history_base is None
    # A transcript reported over HTTP (no anchor) after the initiative is not part of 'hello'.
    assert session.voice_transcript('sure', 1, 2) == 'fresh'
    assert contents(chat) == [('user', 'User: hello'), ('assistant', 'Hi!'), ('assistant', 'Want to take a break?')]


def test_a_remote_turn_leaves_the_local_wake_window_and_microphone_alone(parts):
    session, chat, _ = parts
    voice = voice_for(session, ['turn the lights off'])
    seen = {}
    def remote(kwargs):
        seen['wake_active'] = session.wake.active()
        anchor = seen['anchor'] = session.voice_anchor()
        session.voice_activity(5.0, anchor)  # someone talks in the room for five seconds
        seen['transcript'] = session.voice_transcript('turn the lights off', 1, 6, anchor)
        transcribe(voice, anchor)
        seen['cancelled'] = kwargs['cancelled']()
    chat.respond = answer(chat, 'Hello from Riko.', before=remote)
    session.respond('hi', 'Alice', speak=False, origin=DISCORD)
    assert seen == {'wake_active': False, 'anchor': None, 'transcript': 'fresh', 'cancelled': False}
    assert not session.wake.active()  # no follow-up window opened by the remote reply
    assert contents(chat) == [('user', 'Alice: hi'), ('assistant', 'Hello from Riko.')]
    chat.respond = answer(chat, 'Lights off.')
    run_dispatches(voice)  # the local request is answered as a local turn of its own
    assert contents(chat)[-2:] == [('user', 'User: turn the lights off'), ('assistant', 'Lights off.')]


def test_a_refused_turn_records_no_input(parts):
    session, _, _ = parts
    assert session._turns.try_begin()  # another turn is running
    try:
        with pytest.raises(RuntimeError, match='Riko is already handling another turn'):
            session.respond('are you there?', origin={'source': 'message', 'message_id': 'typed'})
    finally: session._turns.end()
    assert session.state.snapshot()['incoming'] == []


def test_a_finished_utterance_waits_for_an_initiative_that_takes_the_turn_first(parts, monkeypatch):
    session, chat, events = parts
    voice = voice_for(session, ['what time is it?'])
    transcribe(voice, None)  # finished while idle: a new turn is queued
    settled, proceed, presented = threading.Event(), threading.Event(), []
    def guard():  # present_initiative calls this while holding the turn
        settled.set()
        proceed.wait(5)
        return False
    def initiative():
        presented.append(session.present_initiative('Time for a stretch?', guard=guard))
        settled.set()
    thread = threading.Thread(target=initiative, daemon=True)
    observe = session.state.observe_input
    def racing_observe(*args, **kwargs):
        # An initiative commits in the gap between the voice dispatch's check and its turn.
        if not thread.is_alive() and not presented:
            thread.start()
            settled.wait(5)
        return observe(*args, **kwargs)
    monkeypatch.setattr(DesktopState, 'observe_input', lambda state, *args, **kwargs: racing_observe(*args, **kwargs))  # slots: patch the class
    chat.respond = answer(chat, 'Half past two.')
    try:
        run_dispatches(voice)
    finally:
        proceed.set()
        thread.join(5)
    assert chat.calls == [('what time is it?', True)]
    assert ('assistant', 'Half past two.') in contents(chat)
    assert 'voice.error' not in [event.type for event in events]


def test_a_queued_voice_turn_waits_for_a_running_turn_and_stops_waiting_on_close(parts):
    session, chat, events = parts
    voice = voice_for(session, ['first', 'second'])
    started, release = threading.Event(), threading.Event()
    def slow(kwargs):
        started.set()
        release.wait(5)
    chat.respond = answer(chat, 'Typed reply.', before=slow)
    typed = threading.Thread(target=session.respond, args=('typed',), daemon=True)
    typed.start()
    assert started.wait(5)
    transcribe(voice, None)
    chat.respond = answer(chat, 'Voice reply.')
    voice.closed.clear()
    function, args = voice.dispatched.pop()
    waiting = threading.Thread(target=function, args=args, daemon=True)
    waiting.start()
    waiting.join(0.3)
    assert waiting.is_alive() and chat.calls == []  # queued behind the typed turn, not refused
    release.set()
    typed.join(5)
    waiting.join(5)
    assert not waiting.is_alive()
    assert contents(chat)[-2:] == [('user', 'User: first'), ('assistant', 'Voice reply.')]
    # A dispatch still waiting when the microphone closes gives up: no turn and no error.
    assert session._turns.try_begin()
    try:
        transcribe(voice, None)
        voice.closed.clear()
        function, args = voice.dispatched.pop()
        waiting = threading.Thread(target=function, args=args, daemon=True)
        waiting.start()
        voice.closed.set()
        waiting.join(5)
        assert not waiting.is_alive()
    finally: session._turns.end()
    assert [call[0] for call in chat.calls] == ['typed', 'first']
    assert 'voice.error' not in [event.type for event in events]


def test_a_redo_queued_behind_another_turn_answers_the_words_as_a_new_turn(parts):
    session, chat, events = parts
    session.state.toggle_audio()  # muted
    voice = voice_for(session, ['Wait, Tuesday'])
    def first(text, user_name, **kwargs):
        kwargs['on_delta']('Monday works. ')
        anchor = session.voice_anchor()
        session.voice_activity(2.0, anchor)  # sustained speech cuts the reply; a redo is queued
        transcribe(voice, anchor)
        raise TurnCancelled()
    chat.respond = first
    with pytest.raises(TurnCancelled): session.respond('Can we meet?')
    chat.respond = answer(chat, 'Discord reply.')
    session.respond('ping', 'Alice', speak=False, origin=DISCORD)  # a Discord turn wins the lock first
    chat.respond = answer(chat, 'Tuesday it is.')
    run_dispatches(voice)
    # The words leave the overtaken reply's history and become the new turn: recorded once, then answered.
    assert contents(chat) == [('user', 'User: Can we meet?'), ('assistant', 'Monday works. '), ('user', 'Alice: ping'),
        ('assistant', 'Discord reply.'), ('user', 'User: Wait, Tuesday'), ('assistant', 'Tuesday it is.')]
    assert 'voice.error' not in [event.type for event in events]


def test_a_transcript_whose_turn_was_replaced_is_dispatched_not_dropped(parts):
    session, chat, events = parts
    voice = voice_for(session, ['remind me at noon'])
    anchors = []
    def speak_then_anchor(kwargs):
        kwargs['on_delta']('Hello! ')
        anchors.append(session.voice_anchor())  # the utterance starts while the first reply is visible
    chat.respond = answer(chat, 'there.', before=speak_then_anchor)
    session.respond('hello')
    chat.respond = answer(chat, 'Typed reply.')
    session.respond('a typed message')  # a second turn starts before the utterance's final ASR
    transcribe(voice, anchors[0])
    assert not [message for message in chat.conversation.snapshot() if 'noon' in message.content]  # not inserted into either turn
    chat.respond = answer(chat, 'Noon reminder set.')
    run_dispatches(voice)
    assert contents(chat)[-2:] == [('user', 'User: remind me at noon'), ('assistant', 'Noon reminder set.')]
    assert 'voice.error' not in [event.type for event in events]


def test_a_merged_continuation_overtaken_by_another_turn_is_asked_once_as_a_new_turn(parts):
    session, chat, events = parts
    voice = voice_for(session, ['and tomorrow?'])
    def reasoning(text, user_name, **kwargs):
        transcribe(voice, session.voice_anchor())  # merged into the question; the reply is cut for the redo
        raise TurnCancelled()
    chat.respond = reasoning
    with pytest.raises(TurnCancelled): session.respond("What's the weather today?")
    chat.respond = answer(chat, 'Discord reply.')
    session.respond('ping', 'Alice', speak=False, origin=DISCORD)  # wins the lock before the redo
    chat.respond = answer(chat, 'Sunny today, rain tomorrow.')
    run_dispatches(voice)
    assert contents(chat) == [('user', 'Alice: ping'), ('assistant', 'Discord reply.'),
        ('user', "User: What's the weather today?\nand tomorrow?"), ('assistant', 'Sunny today, rain tomorrow.')]


def test_an_utterance_queued_before_a_merged_continuation_is_answered_and_nothing_is_recorded_twice(parts):
    session, chat, _ = parts
    voice = voice_for(session, ['in Paris', 'and tomorrow?'])
    def reasoning(text, user_name, **kwargs):
        transcribe(voice, None)  # began before this turn: queued as its own turn
        transcribe(voice, session.voice_anchor())  # during reasoning: merged, and its redo queued behind that turn
        raise TurnCancelled()
    chat.respond = reasoning
    with pytest.raises(TurnCancelled): session.respond("What's the weather")
    replies = iter(['Paris is sunny.', 'Sunny today, rain tomorrow.'])
    chat.respond = lambda text, user_name, **kwargs: commit(chat, text, user_name, kwargs, next(replies))
    run_dispatches(voice)
    assert contents(chat) == [('user', 'User: in Paris'), ('assistant', 'Paris is sunny.'),
        ('user', "User: What's the weather\nand tomorrow?"), ('assistant', 'Sunny today, rain tomorrow.')]


def test_sustained_speech_after_stop_cuts_nothing_and_starts_a_turn_of_its_own(parts):
    session, chat, _ = parts
    anchors = []
    def stopped(text, user_name, **kwargs):
        anchors.append(session.voice_anchor())
        session.cancel()  # the Stop button, while the model reasons
        raise TurnCancelled()
    chat.respond = stopped
    with pytest.raises(TurnCancelled): session.respond('What about today?')
    session.voice_activity(2.0, anchors[0])  # the user keeps talking past the interruption threshold
    assert session.voice_transcript('and tomorrow', 1, 2, anchors[0]) == 'fresh'
    assert contents(chat) == [('user', 'User: What about today?')]


@pytest.mark.parametrize('animation', [False, None, 'off'])
def test_an_animation_setting_that_is_not_a_section_disables_animation_without_failing_the_session(monkeypatch, tmp_path, animation):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    config = SimpleNamespace(raw={'animation': animation}, root=tmp_path, paths=DataPaths.at(tmp_path), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    session = SessionManager(config, SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), provider=BaseProvider()), DesktopState())
    try: assert session.animation is None and (session.animation_error == '' if animation is False else 'must be a mapping' in session.animation_error)
    finally: session.close()


def test_a_reply_with_no_visible_text_keeps_its_placeholder_so_roles_still_alternate(parts):
    session, chat, events = parts
    def reasoning_only(text, user_name, **kwargs):  # the model reasoned through its whole budget: no visible delta
        chat.conversation.append([ChatMessage('user', f'{user_name}: {text}'), *kwargs['response_history']('...')])
        return ModelResponse(ChatMessage('assistant', '...'))  # ChatService's placeholder for an empty answer
    chat.respond = reasoning_only
    session.respond('Say hello')
    chat.respond = answer(chat, 'Hello there!')
    session.respond('Again?')
    assert contents(chat) == [('user', 'User: Say hello'), ('assistant', '...'), ('user', 'User: Again?'), ('assistant', 'Hello there!')]
    assert [event.payload['text'] for event in events if event.type == 'chat.completed'] == ['...', 'Hello there!']


def test_an_empty_reply_is_not_visible_output_that_speech_could_be_over(parts):
    session, chat, _ = parts
    anchors = []
    def reasoning_only(text, user_name, **kwargs):
        chat.conversation.append([ChatMessage('user', f'{user_name}: {text}'), *kwargs['response_history']('...')])
        return ModelResponse(ChatMessage('assistant', '...'))
    chat.respond = reasoning_only
    original = session._rewrite_history
    def rewrite(*args):  # speech starting while the finished turn is still generating
        anchors.append(session.voice_anchor())
        return original(*args)
    session._rewrite_history = rewrite
    session.respond('Say hello')
    assert anchors and anchors[0][2] is False  # nothing visible: the user's words are a turn of their own, not speech over it


def test_a_spoken_initiative_with_bad_voice_settings_commits_nothing(parts):
    session, chat, _ = parts
    session.config.voice = replace(session.config.voice, interjection_debounce_seconds=0)  # load_config rejects this; built another way
    for _ in range(2):
        with pytest.raises(ValueError): session.present_initiative('Time for a stretch?', spoken=True)
    assert contents(chat) == []  # neither attempt left a message nobody saw or heard
