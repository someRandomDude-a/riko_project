import queue
import threading
import sys
from concurrent.futures import Future
from types import SimpleNamespace

from process.app_core.audio.speech import AudioClip, SpeechQueue


def test_master_volume_scales_pcm_without_changing_duration():
    import struct
    from process.app_core.audio.volume import scale_pcm16
    pcm = struct.pack('<hhh', 1000, -1000, 32767)
    assert scale_pcm16(pcm, 1) == pcm
    assert scale_pcm16(pcm, 0) == bytes(len(pcm))
    assert struct.unpack('<hhh', scale_pcm16(pcm, .5)) == (500, -500, 16383)


def test_volume_changes_apply_between_blocks_without_cancelling_speech():
    import struct
    speech = SpeechQueue.__new__(SpeechQueue)
    speech.state = SimpleNamespace(audio_enabled=True, audio_volume=1)
    speech.closed = threading.Event()
    chunks = queue.Queue()
    for _ in range(3): chunks.put(struct.pack('<h', 1000))
    future = Future(); future.set_result(.1)
    written = []
    def write(pcm):
        written.append(struct.unpack('<h', pcm)[0])
        speech.state.audio_volume = .25 if len(written) == 1 else 0
    result = speech._write_audio(SimpleNamespace(write=write), 'test', 'volume-turn', chunks, future, threading.Event(), 32000)
    assert written == [1000, 250, 0]
    assert result is not None


def test_symbol_only_tail_never_reaches_tts():
    # Exercise submission without opening devices or starting a worker.
    speech = SpeechQueue.__new__(SpeechQueue)
    speech.state = SimpleNamespace(audio_enabled=True)
    speech.closed = threading.Event()
    speech.queue = queue.Queue()
    speech.submit('.🕊️', 'turn')
    assert speech.queue.empty()


def test_cancel_clears_queue_cancels_pending_and_closes_only_old_streams():
    speech = SpeechQueue.__new__(SpeechQueue)
    speech._submit_lock = threading.Lock()
    speech._response_lock = threading.Lock()
    speech._interrupt = threading.Event()
    old_interrupt = speech._interrupt
    future = Future()
    speech._futures = {future}
    speech.queue = queue.Queue()
    speech.queue.put(('sentence', 'turn', queue.Queue(), future, old_interrupt, 0, 8))
    closed = []
    class Response:
        def __init__(self, name): self.name = name
        def close(self): closed.append(self.name)
    old, new = Response('old'), Response('new')
    speech._responses = {old: old_interrupt, new: threading.Event()}
    aborted = []
    speech._output = (SimpleNamespace(abort=lambda: aborted.append(True)), old_interrupt)
    cleanup = speech.cancel()
    cleanup.join(timeout=1)
    assert old_interrupt.is_set()
    assert not speech._interrupt.is_set()
    assert speech.queue.empty()
    assert speech.queue.unfinished_tasks == 0
    assert future.cancelled()
    assert closed == ['old']
    assert aborted == [True]


def test_missing_sovits_keeps_chat_alive_and_next_reply_recovers(tmp_path, monkeypatch):
    import requests
    from process.app_core.conversation.chat import ChatService
    from process.app_core.desktop.state import DesktopState
    from process.app_core.events.bus import event_bus
    from process.app_core.conversation.messages import ChatMessage, ModelResponse
    from process.app_core.runtime.session import SessionManager
    from process.app_core.runtime.warmup import warm_components

    available, attempts, played = [False], [], []
    class Response:
        def raise_for_status(self): pass
        def iter_content(self, **kwargs): yield bytes(2048)
        def close(self): pass
    def post(url, **kwargs):
        attempts.append(kwargs['json']['text'])
        if not available[0]: raise requests.ConnectionError('Connection refused')
        return Response()
    monkeypatch.setattr(requests, 'post', post)
    class Output:
        def __init__(self, **kwargs):
            assert available[0], 'Do not open playback when synthesis is unavailable'
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def write(self, pcm): played.append(pcm)
        def abort(self): pass
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(RawOutputStream=Output))
    class Provider:
        def generate(self, messages, **kwargs):
            text = 'Here is your answer.'
            kwargs['on_delta'](text)
            return ModelResponse(ChatMessage('assistant', text))
        def close(self): pass
    config = SimpleNamespace(root=tmp_path, character_name='Riko', raw={}, tools=SimpleNamespace(max_iterations=8))
    chat = ChatService(Provider(), system_prompt='test')
    session = SessionManager(config, chat, DesktopState())
    events, failed, completed = [], threading.Event(), threading.Event()
    def observe(event):
        events.append(event)
        if event.type == 'speech.error': failed.set()
        if event.type == 'speech.completed': completed.set()
    unsubscribe = event_bus.subscribe(observe)
    try:
        # Optional TTS startup failure is visible but does not prevent startup.
        warm_components([('tts', session.speech.warmup)], 2)
        assert session.speech.last_error and any(event.type == 'speech.unavailable' for event in events)
        assert session.respond('First message').message.content == 'Here is your answer.'
        assert failed.wait(2)
        assert session._speech_pending == 0 and not session._cancel.is_set()
        assert session.state.audio_enabled and not session.speech.closed.is_set()
        assert not session.speech.submit('Remaining chunk.', session._active_turn)
        assert chat.history[-1].content == 'Here is your answer.'
        # Starting the external service needs no toggle or application restart.
        available[0] = True
        assert session.respond('Second message').message.content == 'Here is your answer.'
        assert completed.wait(2)
        assert played and not session.speech.last_error
        assert session._speech_pending == 0
        assert len(attempts) == 3  # Warmup, failed reply, successful next reply.
        assert not any(event.type == 'model.error' for event in events)
    finally:
        unsubscribe()
        session.close()
        session.speech.worker.join(timeout=1)


def test_local_wake_cue_uses_native_rate_and_never_emits_speech_events(tmp_path, monkeypatch):
    import wave
    import pytest
    from process.app_core.events.bus import event_bus
    path = tmp_path / 'cue.wav'
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b'\x00\x20' * 1600 * 2)
    writes, opened, events = [], [], []
    class Output:
        def __init__(self, **kwargs): opened.append(kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def write(self, pcm): writes.append(pcm.copy())
        def abort(self): pass
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(OutputStream=Output))
    speech = SpeechQueue.__new__(SpeechQueue)
    speech.closed, speech._response_lock, speech._output = threading.Event(), threading.Lock(), None
    speech.state = SimpleNamespace(audio_enabled=True)
    unsubscribe = event_bus.subscribe(events.append)
    try:
        speech._play_clip(AudioClip(path, .5, 3, {'emotion': 'joy', 'model_state': 'idle'}), threading.Event())
        assert opened == [{'samplerate': 16000, 'channels': 2, 'dtype': 'float32'}]
        assert sum(len(pcm) for pcm in writes) == 1600
        assert float(writes[0][0][0]) == .125
        assert [event.type for event in events] == ['wake.feedback.started', 'wake.feedback.completed']
        assert speech._output is None
        with pytest.raises(ValueError, match='at most'):
            speech._play_clip(AudioClip(path, .5, .05), threading.Event())
        speech.state.audio_enabled = False
        speech._play_clip(AudioClip(path, .5, 3), threading.Event())
        speech.state.audio_enabled = True
        speech._play_clip(AudioClip(path, .5, 3, expires_at=0), threading.Event())
        assert len(opened) == 1
    finally: unsubscribe()


def test_wake_cue_is_queued_on_existing_playback_lane(monkeypatch):
    from process.app_core.events.bus import event_bus
    release, started, finished, order = threading.Event(), threading.Event(), threading.Event(), []
    def clip(self, item, interrupt):
        order.append('cue')
        started.set()
        assert release.wait(2)
    def receive(self, text, chunks, interrupt, turn_id=None): return 0.1
    def play(self, *args):
        order.append('speech')
        return {'first_audio_seconds': .1, 'audio_seconds': .1}
    monkeypatch.setattr(SpeechQueue, '_play_clip', clip)
    monkeypatch.setattr(SpeechQueue, '_receive', receive)
    monkeypatch.setattr(SpeechQueue, '_play', play)
    speech = SpeechQueue(SimpleNamespace(raw={}), SimpleNamespace(audio_enabled=True))
    unsubscribe = event_bus.subscribe(lambda event: finished.set() if event.type == 'speech.completed' else None)
    try:
        assert speech.submit_clip('cue.wav')
        assert started.wait(1)
        assert speech.submit('Reply.', 'turn')
        assert order == ['cue']
        release.set()
        assert finished.wait(1) and order == ['cue', 'speech']
    finally:
        release.set()
        unsubscribe()
        speech.close()
        speech.worker.join(timeout=1)
