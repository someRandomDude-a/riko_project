import json
from pathlib import Path
from types import SimpleNamespace
import queue
import sys
import threading

from process.app_core.conversation.chat import ChatService
from process.app_core.desktop.state import DesktopState
from process.app_core.events.bus import RuntimeEvent
from process.app_core.runtime.interjections import Interjections
from process.app_core.conversation.messages import ChatMessage, ModelResponse, ToolCall
from process.app_core.runtime.session import SessionManager
from process.app_core.tools.registry import ToolRegistry
from process.app_core.audio.voice_input import VoiceInput
from process.app_core.audio.voice_segments import Segment


class Speech:
    def __init__(self, *args): self.cancelled = False
    def submit(self, *args): return True
    def cancel(self): self.cancelled = True
    def close(self): pass


def make_session(monkeypatch, provider=None):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    config = SimpleNamespace(raw={'voice': {}}, root=Path('.'), character_name='Riko', tools=SimpleNamespace(max_iterations=8))
    chat = ChatService(provider or SimpleNamespace(close=lambda: None), system_prompt='Riko', tool_registry=ToolRegistry())
    return SessionManager(config, chat, DesktopState())


def activate(session):
    session._active_turn = 'turn'
    session._generation_active = True
    session._interjections = Interjections(session.cancel)


def test_legacy_todo_contents_are_not_automatically_injected(monkeypatch):
    session = make_session(monkeypatch)
    session.state.tool_activity = [{'name': 'todo_list', 'status': 'complete',
                                    'arguments': {'task': 'private task'}, 'result': 'private task list'}]
    assert session.runtime_snapshot()['desktop']['tools'] == [{'name': 'todo_list', 'status': 'complete'}]
    session.close()


def test_priority_preserves_overlap_and_restores_normal_threshold(monkeypatch):
    now = [100.0]
    monkeypatch.setattr('process.app_core.runtime.session.time.monotonic', lambda: now[0])
    session = make_session(monkeypatch)
    activate(session)
    anchor = session.voice_anchor()
    result = session.interrupt_user('Let me finish this point')
    assert result['interruption_seconds'] == 6
    assert not session.interrupt_user('Again')['granted']
    session.voice_activity(2.0, anchor)
    assert not session.speech.cancelled
    assert not session.user_interrupted(anchor)
    session._generated = 'One two three four five.'
    session.voice_transcript('But what about tomorrow?', 100, 102, anchor)
    assert '[speaking over you] But what about tomorrow?' in str(session._response_history(session._generated))
    assert session.runtime_snapshot()['runtime']['speaking_priority']['active']
    now[0] = 116
    assert session.interruption_threshold() == 1.5
    session.voice_activity(2, anchor)
    assert session.speech.cancelled
    assert session.user_interrupted(anchor)
    session.close()


def test_priority_still_allows_sustained_interruption_and_explicit_stop(monkeypatch):
    session = make_session(monkeypatch)
    activate(session)
    session.interrupt_user('Important point')
    session.voice_activity(5.9, session.voice_anchor())
    assert not session.speech.cancelled
    session.voice_activity(6, session.voice_anchor())
    assert session.speech.cancelled
    assert session._assertive_until == 0
    session.close()
    session = make_session(monkeypatch)
    activate(session)
    session.interrupt_user('Important point')
    session.cancel()
    assert session.speech.cancelled
    session.close()


def test_priority_ends_when_playback_finishes(monkeypatch):
    session = make_session(monkeypatch)
    activate(session)
    session.interrupt_user('Important point')
    session._generation_active = False
    session._speech_pending = 1
    session._playback_event(RuntimeEvent('speech.completed', {'end_offset': 10}, turn_id='turn'))
    assert session.interruption_threshold() == 1.5
    session.close()


def test_runtime_context_refreshes_after_tool_and_includes_outcomes(monkeypatch):
    observed = []
    class Provider:
        def generate(self, messages, **options):
            snapshot = json.loads(messages[-1].content.split('\n', 1)[1])
            observed.append(snapshot)
            if len(observed) == 1:
                return ModelResponse(ChatMessage('assistant', tool_calls=[ToolCall('call1', 'interrupt_user', {'reason': 'Let me explain'})]))
            assert snapshot['runtime']['speaking_priority']['active']
            options['on_delta']('One two three four five. ')
            return ModelResponse(ChatMessage('assistant', 'One two three four five.'))
        def close(self): pass
    session = make_session(monkeypatch, Provider())
    session.state.add_whiteboard('text', {'text': 'A current note'})
    session.state.actions = [{'id': 'a', 'status': 'completed'}]
    session.respond('Please explain')
    assert len(observed) == 2
    assert observed[0]['runtime']['generating']
    assert not observed[0]['runtime']['listening']
    assert not observed[0]['runtime']['speaking_priority']['active']
    assert observed[0]['desktop']['whiteboard'][0]['payload']['text'] == 'A current note'
    assert observed[0]['desktop']['actions'][0]['status'] == 'completed'
    assert observed[0]['perception']['screen'] == 'unavailable'
    session.close()


def test_delayed_asr_preserves_tolerated_speech_without_reply_dispatch(monkeypatch):
    session = make_session(monkeypatch)
    activate(session)
    anchor = session.voice_anchor()
    session.interrupt_user('Finish my explanation')
    session.voice_activity(2, anchor)
    # Native ASR is stubbed; use a silent PCM job and a deterministic transcript.
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=object))
    voice = VoiceInput.__new__(VoiceInput)
    voice.session, voice.closed, voice.jobs, voice._parts = session, threading.Event(), queue.Queue(), {}
    submitted = []
    voice.responses = SimpleNamespace(submit=lambda *args: submitted.append(args))
    def transcribe(*args, **kwargs):
        # Simulate the priority window expiring while ASR is finishing.
        session._assertive_until = 0
        return iter([SimpleNamespace(text='But tomorrow?')]), None
    voice.model = SimpleNamespace(transcribe=transcribe)
    voice.asr_lock = threading.Lock()
    class Jobs(queue.Queue):
        def task_done(self):
            super().task_done()
            voice.closed.set()
    voice.jobs = Jobs()
    voice.jobs.put(Segment('utterance', bytes(1024), anchor, 100, 102, 2, True))
    voice._asr()
    assert not submitted
    assert session._interjections.items[0].text == 'But tomorrow?'
    assert not session.speech.cancelled
    session.close()
