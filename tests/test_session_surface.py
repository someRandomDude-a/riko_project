"""SessionManager's public surface: a typed TurnBusy that every route answers with 409, one RuntimeStatus whose predicates
every busy/idle reader shares, is_open/active_turn()/cancel_turn(), the one Whisper model, the Discord socket's event
filter, and no module outside session.py touching the session's private state. Threads meet at events with generous
bounded waits; nothing depends on timing."""
import ast
import queue
import sys
import threading
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from process.app_core.animation.runtime import AnimationRuntime
from process.app_core.audio import asr
from process.app_core.audio.voice_input import VoiceInput
from process.app_core.audio.voice_segments import Segment
from process.app_core.conversation.chat import ChatDeps
from process.app_core.conversation.history import ConversationHistory
from process.app_core.desktop.state import DesktopState
from process.app_core.events.bus import RuntimeEvent, event_bus
from process.app_core.http.chat import BUSY
from process.app_core.integrations.discord import api as discord_api
from process.app_core.kernel.cancellation import TurnBusy, TurnCancelled
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.kernel.turns import TurnGate
from process.app_core.runtime import warmup
from process.app_core.runtime.initiative import Initiative
from process.app_core.runtime.session import SessionManager
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths
from conftest import client_for  # the backend fixture comes from conftest.py
from test_private_access import CODE, private_access

REFUSED = 'Stop the current reply before calibration or testing'


class Speech:
    def __init__(self, *args): pass
    def submit(self, *args): return False
    def cancel(self): pass
    def warmup(self): pass
    def close(self): pass


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), provider=BaseProvider())
    config = SimpleNamespace(raw={'animation': {'enabled': False}}, root=tmp_path, paths=DataPaths.at(tmp_path), character_name='Riko',
        tools=SimpleNamespace(max_iterations=8), runtime=SimpleNamespace(startup_timeout_seconds=5), **audio_sections({}))
    session = SessionManager(config, chat, DesktopState())
    try: yield session
    finally: session.close()


def answer(chat, reply):
    def respond(text, user_name, **kwargs):
        kwargs['on_delta'](reply)
        chat.conversation.append([ChatMessage('user', f'{user_name}: {text}'), *kwargs['response_history'](reply)])
        return ModelResponse(ChatMessage('assistant', reply))
    return respond


def test_a_busy_session_raises_a_typed_turn_busy_that_keeps_the_old_message(session):
    assert issubclass(TurnBusy, RuntimeError) and str(TurnBusy()) == 'Riko is already handling another turn'
    session.chat.respond = lambda *args, **kwargs: pytest.fail('a refused turn never reaches the model')
    assert session._turns.try_begin()  # a voice or Discord turn holds the session
    try:
        with pytest.raises(TurnBusy, match='^Riko is already handling another turn$'): session.respond('hello')
        assert not session.present_initiative('Time for a stretch?')  # a background proposal never waits for the turn
    finally: session._turns.end()
    assert session.chat.conversation.snapshot() == [] and session.state.snapshot()['incoming'] == []


def test_the_turn_gate_waits_only_when_asked_and_gives_up_when_the_waiter_does():
    gate, give_up, outcome = TurnGate(poll=0.01), threading.Event(), []
    def waiter():
        try: gate.begin(wait=give_up.is_set); outcome.append('taken')
        except TurnCancelled: outcome.append('abandoned')
    gate.begin()
    assert gate.busy and not gate.try_begin()
    with pytest.raises(TurnBusy): gate.begin()
    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    give_up.set()
    thread.join(5)
    assert outcome == ['abandoned'] and gate.busy  # the running turn still holds it
    give_up.clear()
    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    gate.end()  # the waiter takes it as the running turn ends, or at once if it starts later
    thread.join(5)
    assert outcome == ['abandoned', 'taken'] and gate.busy
    gate.end()
    assert not gate.busy and gate.try_begin()
    gate.end()


def test_every_route_answers_turn_busy_with_409_and_only_the_type_counts(backend, monkeypatch, session):
    backend.session = session
    client = client_for(backend, raise_server_exceptions=False)
    router = FastAPI(); router.include_router(discord_api.create_router(lambda: session))
    remote = TestClient(router, raise_server_exceptions=False)
    message = {'text': 'hi', 'turn_id': str(uuid.uuid4())}
    session.chat.respond = answer(session.chat, 'Hello again.')
    with client.websocket_connect('ws://127.0.0.1:8765/ws/chat') as socket:
        session._turns.begin()  # a voice turn, or a reply still stopping
        try:
            assert client.post('/api/chat', json={'text': 'hello'}).status_code == 409
            assert remote.post('/api/discord/chat', json=message).status_code == 409
            socket.send_json({'text': 'hello'})
            assert socket.receive_json() == {'type': 'chat.started'}
            assert socket.receive_json() == {'type': 'chat.busy', 'payload': {'status': 409, 'detail': BUSY}}
        finally: session._turns.end()
        socket.send_json({'text': 'hello'})  # the socket stays open for the next message
        assert socket.receive_json() == {'type': 'chat.started'}
        assert socket.receive_json() == {'type': 'chat.completed', 'payload': {'text': 'Hello again.'}}
    # Words alone are not the conflict: a RuntimeError that only carries the same message is a server error.
    def impostor(*args, **kwargs): raise RuntimeError(TurnBusy.MESSAGE)
    monkeypatch.setattr(session, 'respond', impostor)
    assert client.post('/api/chat', json={'text': 'hello'}).status_code == 500
    assert remote.post('/api/discord/chat', json=message).status_code == 500


def test_every_busy_and_idle_reader_agrees_through_one_runtime_status(backend, monkeypatch, session):
    """The avatar, wake feedback, initiative, the emotion probe's idle check, wake calibration and barge-in used to
    disagree, e.g. in the gap between a reply's generation and its first sentence playing."""
    initiative = Initiative(session, adapter=SimpleNamespace(sample=lambda **kwargs: {}), start_worker=False)
    animation = AnimationRuntime(session, start=False)
    feedback = []
    monkeypatch.setattr(session.wake_feedback, 'trigger', lambda emotion, model_state, **kwargs: feedback.append(model_state))
    backend.session = session
    client = client_for(backend)
    def verdicts():
        feedback.clear()
        event_bus.publish('voice.activated', source='keyword')
        calibration = client.post('/api/voice/calibration', json={'action': 'begin'})  # refused, or fails later: no microphone
        assert calibration.status_code == 400
        return {'quiet': session.chat.provider.expression_idle(), 'background': initiative.available(),
                'presented': session.present_initiative('Time for a stretch?'), 'anchor': session.voice_anchor() is not None,
                'calibration_refused': calibration.json()['detail'] == REFUSED, 'avatar': animation._observe()[0].mode,
                'wake_feedback': feedback}
    try:
        assert verdicts() == {'quiet': True, 'background': True, 'presented': True, 'anchor': False,
            'calibration_refused': False, 'avatar': 'idle', 'wake_feedback': ['idle']}
        session._active_turn, session._speech_pending = 'turn', 1  # generation ended; a sentence is still queued to play
        assert verdicts() == {'quiet': False, 'background': False, 'presented': False, 'anchor': True,
            'calibration_refused': True, 'avatar': 'thinking', 'wake_feedback': ['thinking']}
        session._speech_pending = 0
        session.set_user_speaking(True)  # what VoiceInput reports from VAD
        assert verdicts() == {'quiet': False, 'background': False, 'presented': False, 'anchor': False,
            'calibration_refused': False, 'avatar': 'listening', 'wake_feedback': ['idle']}  # the speech was the wake word
        session.set_user_speaking(False)
        session._generation_active, session._turn_speak = True, False  # a Discord reply: generating, but nothing plays here
        assert verdicts() == {'quiet': False, 'background': False, 'presented': False, 'anchor': False,
            'calibration_refused': True, 'avatar': 'thinking', 'wake_feedback': ['thinking']}
        session._generation_active, session._turn_speak = False, True
        session.wake.calibrating = True  # calibration owns the microphone: quiet, but no background proposal
        assert verdicts() == {'quiet': True, 'background': False, 'presented': False, 'anchor': False,
            'calibration_refused': False, 'avatar': 'idle', 'wake_feedback': ['idle']}
        session.wake.calibrating = False
        session.state.set_sleep(True)
        assert verdicts() == {'quiet': True, 'background': False, 'presented': False, 'anchor': False,
            'calibration_refused': False, 'avatar': 'sleeping', 'wake_feedback': ['sleeping']}
    finally:
        animation.close()
        initiative.close()


def test_cancel_turn_stops_only_the_named_turn_while_it_generates(session):
    seen = {}
    def respond(text, user_name, **kwargs):
        seen['active'] = session.active_turn()
        seen['other'] = session.cancel_turn('another-turn') or kwargs['cancelled']()
        seen['own'] = session.cancel_turn('discord-turn')
        if kwargs['cancelled'](): raise TurnCancelled()
    session.chat.respond = respond
    assert session.is_open and session.active_turn() is None and not session.cancel_turn(None)
    with pytest.raises(TurnCancelled):
        session.respond('hi', 'Alice', speak=False, turn_id='discord-turn', origin={'source': 'discord'})
    assert seen == {'active': 'discord-turn', 'other': False, 'own': True}
    assert session.active_turn() == 'discord-turn' and not session.cancel_turn('discord-turn')  # finished: nothing to stop
    session.close()
    assert not session.is_open


def test_the_microphone_warmup_and_discord_share_one_whisper_model(session, monkeypatch):
    built, building, release = [], threading.Event(), threading.Event()
    class WhisperModel:
        def __init__(self, name, **options):
            built.append((name, options)); building.set(); release.wait(5)
        def transcribe(self, samples, **options): return iter([SimpleNamespace(text=' heard you ')]), None
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=WhisperModel))
    monkeypatch.setattr(asr, 'supported_types', lambda: {'cpu': frozenset({'float32', 'int8'})})
    monkeypatch.setattr(warmup, 'warm_components', lambda jobs, timeout: [job() for name, job in jobs if name == 'asr'])
    voice = VoiceInput.__new__(VoiceInput)
    voice.session, voice.closed, voice._parts, dispatched = session, threading.Event(), {}, []
    voice._partial_lock, voice._partial_pending = threading.Lock(), set()
    voice.responses = SimpleNamespace(submit=lambda function, text, segment: dispatched.append(text))
    class Jobs(queue.Queue):
        def task_done(self):
            super().task_done()
            voice.closed.set()
    voice.jobs = Jobs()
    voice.jobs.put(Segment('utterance', bytes(1024), None, 1.0, 2.0, 1.0, True))
    heard = []
    workers = [threading.Thread(target=warmup.warm_session, args=(session,), daemon=True)]
    workers[0].start()
    assert building.wait(5)  # startup warmup is building the model; the microphone and Discord arrive meanwhile
    workers += [threading.Thread(target=voice._asr, daemon=True),
                threading.Thread(target=lambda: heard.append(discord_api.transcribe_pcm(session, bytes(3200))), daemon=True)]
    for worker in workers[1:]: worker.start()
    release.set()
    for worker in workers: worker.join(5)
    assert built == [('distil-small.en', {'device': 'cpu', 'compute_type': 'int8'})]  # one model, not one per caller
    assert heard == ['heard you'] and dispatched == ['heard you']
    session.asr.close()
    with pytest.raises(RuntimeError, match='closed'): session.transcribe(bytes(320))
    assert session.asr.model is None and len(built) == 1


def test_the_discord_socket_carries_exactly_the_events_the_bot_handles():
    tree = ast.parse((CODE / 'process/app_core/integrations/discord/bot.py').read_text(encoding='utf-8'))
    handler = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == 'backend_event')
    kinds = {constant.value for node in ast.walk(handler) if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == 'kind'
             for constant in ast.walk(node) if isinstance(constant, ast.Constant) and isinstance(constant.value, str)}
    # resource.snapshot opens every socket (events.stream); everything else the bot reads passes the filter, and only that.
    assert kinds - {'resource.snapshot'} == discord_api.CLIENT_EVENTS | discord_api.TURN_EVENTS
    relevant = discord_api.client_event_filter()
    def event(kind, turn=None, **payload): return relevant(RuntimeEvent(kind, payload, turn_id=turn))
    assert all(event(kind) for kind in discord_api.CLIENT_EVENTS)
    assert not event('model.started', 'remote', source='discord')  # not forwarded, but it makes the turn the bot's
    assert event('model.reasoning', 'remote', text='thinking') and event('tool.approval_requested', 'remote', id='a')
    assert not event('chat.delta', 'local', text='a local reply', source='message') and not event('tool.approval_requested', 'local', id='b')
    assert not any(event(kind) for kind in ('voice.level', 'voice.transcript', 'state.snapshot', 'chat.completed', 'model.reasoning'))


def test_no_module_outside_session_py_touches_the_sessions_private_state():
    tree = ast.parse((CODE / 'process/app_core/runtime/session.py').read_text(encoding='utf-8'))
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SessionManager')
    fields = {node.attr for node in ast.walk(manager) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id == 'self' and node.attr.startswith('_') and not node.attr.startswith('__')}
    assert {'_turns', '_voice_lock', '_closed', '_active_turn', '_generation_active', '_playing', '_speech_pending',
            '_user_speaking', '_voice_phase', '_assertive_until', '_interaction_revision'} <= fields
    found = private_access([*sorted((CODE / 'process' / 'app_core').rglob('*.py')), *sorted(CODE.glob('*.py'))])
    reach_ins = sorted(f'{file}: .{name}' for file, name in found if name in fields and file != 'process/app_core/runtime/session.py')
    assert not reach_ins, 'Use SessionManager.status(), is_open, active_turn(), cancel_turn() or another public method:\n' + '\n'.join(reach_ins)
