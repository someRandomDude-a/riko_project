import queue
import threading
import sys
from contextlib import nullcontext
from types import SimpleNamespace

from process.app_core.audio.voice_input import VoiceInput
from process.app_core.audio.voice_segments import Segment
from process.app_core.audio.wake_word import WakeWord


def test_capture_only_queues_post_keyword_request_for_asr(tmp_path, monkeypatch):
    now = [9.9]
    monkeypatch.setattr('process.app_core.audio.wake_word.time.monotonic', lambda: now[0])
    config = SimpleNamespace(root=tmp_path, character_name='Riko', raw={'voice': {}})
    wake = WakeWord(config)
    keyword, activation, request, silence = (bytes([marker, 0]) * 512 for marker in (1, 2, 3, 0))
    monkeypatch.setattr(wake, 'feed', lambda frame, speaking: wake.activate(after_keyword=True) if frame == activation else None)
    class VAD:
        def __call__(self, samples, rate): return float(samples[0] != 0)
        def reset_states(self): pass
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda samples: samples))
    monkeypatch.setitem(sys.modules, 'silero_vad', SimpleNamespace(load_silero_vad=VAD))
    voice = VoiceInput.__new__(VoiceInput)
    voice.session = SimpleNamespace(config=config, wake=wake, state=SimpleNamespace(mic_enabled=True),
        _voice_lock=threading.RLock(), _assertive_until=0, voice_anchor=lambda: None)
    voice.closed, voice.jobs, voice._last_overflow = threading.Event(), queue.Queue(), 0
    class Frames(queue.Queue):
        def get(self, **kwargs):
            if self.empty():
                voice.closed.set()
                raise queue.Empty
            frame, timestamp = super().get(**kwargs)
            now[0] = timestamp
            return frame, timestamp
    voice.frames = Frames()
    packets = [(keyword, 9.9), (activation, 10.0), (keyword, 10.1)]
    packets += [(silence, 10.2 + index * .032) for index in range(10)]
    packets += [(request, 10.6)]
    packets += [(silence, 10.7 + index * .032) for index in range(32)]
    for packet in packets: voice.frames.put(packet)
    try:
        voice._run()
        segments = list(voice.jobs.queue)
        assert segments and sum(part.final for part in segments) == 1
        assert b''.join(part.pcm for part in segments) == request + silence * 10
    finally:
        wake.close()


def test_asr_keeps_legitimate_wake_name_mentions_in_request(monkeypatch):
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=object))
    voice = VoiceInput.__new__(VoiceInput)
    voice.closed, voice._parts, voice.asr_lock = threading.Event(), {}, threading.Lock()
    voice.session = SimpleNamespace(config=SimpleNamespace(raw={'voice': {}}),
        wake=SimpleNamespace(calibrating=False, testing=False, mode='wake_word', phrase='Riko'))
    text = 'Riko is the character in my story.'
    voice.model = SimpleNamespace(transcribe=lambda *a, **k: (iter([SimpleNamespace(text=text)]), None))
    submitted = []
    voice.responses = SimpleNamespace(submit=lambda *args: submitted.append(args))
    class Jobs(queue.Queue):
        def task_done(self):
            super().task_done()
            voice.closed.set()
    voice.jobs = Jobs()
    voice.jobs.put(Segment('request', bytes(1024), None, 10, 11, 1, True))
    voice._asr()
    assert len(submitted) == 1 and submitted[0][1] == text
