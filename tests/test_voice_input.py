import queue
import threading
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from process.app_core.audio.voice_input import VoiceInput
from process.app_core.audio.voice_segments import Segment
from process.app_core.audio.wake_word import WakeWord
from process.app_core.kernel.turns import RuntimeStatus
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths


def test_capture_only_queues_post_keyword_request_for_asr(tmp_path, monkeypatch):
    now = [9.9]
    monkeypatch.setattr('process.app_core.audio.wake_word.time.monotonic', lambda: now[0])
    config = SimpleNamespace(root=tmp_path, paths=DataPaths.at(tmp_path), character_name='Riko', raw={'voice': {}}, **audio_sections({}))
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
        set_user_speaking=lambda speaking: None, status=RuntimeStatus, voice_anchor=lambda: None)
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
    voice.closed, voice._parts = threading.Event(), {}
    text = 'Riko is the character in my story.'
    voice.session = SimpleNamespace(config=SimpleNamespace(raw={'voice': {}}, **audio_sections({})),
        wake=SimpleNamespace(calibrating=False, testing=False, mode='wake_word', phrase='Riko'), transcribe=lambda pcm, **options: text)
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


@pytest.mark.parametrize('barge_in, voiced, gated, transcribed', [('echo_gate', .1, True, False), ('stop_only', .19, True, False),
    ('full_duplex', .1, True, True), ('echo_gate', .2, True, True), ('echo_gate', .1, False, True)])
def test_a_tiny_utterance_among_her_gated_echo_is_never_transcribed(monkeypatch, barge_in, voiced, gated, transcribed):
    from process.app_core.audio.voice_input import SILENT
    voice = VoiceInput.__new__(VoiceInput)
    voice.closed, voice._parts = threading.Event(), {}
    calls, submitted = [], []
    voice.session = SimpleNamespace(config=SimpleNamespace(raw={'voice': {}}, **audio_sections({'voice': {'barge_in': barge_in}})),
        wake=SimpleNamespace(calibrating=False, testing=False), transcribe=lambda pcm, **options: calls.append(pcm) or 'you')
    voice.responses = SimpleNamespace(submit=lambda *args: submitted.append(args))
    class Jobs(queue.Queue):
        def task_done(self):
            super().task_done()
            voice.closed.set()
    voice.jobs = Jobs()
    # A few frames that slipped through the gate, padded with her echo fed in as silence (or, ungated, the room's noise).
    padding = SILENT if gated else b'\x01\x00' * 512
    voice.jobs.put(Segment('fragment', padding * 10 + b'\x00\x10' * 512 * 3 + padding * 10, None, 10, 11, voiced, True))
    voice._asr()
    assert bool(calls) == transcribed and bool(submitted) == transcribed


class NoRaw(dict):
    def get(self, *args): raise AssertionError('voice settings must come from config.voice, checked once by load_config')
    __getitem__ = get


@pytest.mark.parametrize('threshold, heard', [(.5, True), (.75, False)])
def test_each_frame_compares_speech_probability_with_the_typed_vad_threshold(tmp_path, monkeypatch, threshold, heard):
    config = SimpleNamespace(root=tmp_path, paths=DataPaths.at(tmp_path), character_name='Riko', raw=NoRaw(), **audio_sections({'voice': {'mode': 'continuous', 'vad_threshold': threshold}}))
    wake = WakeWord(config)
    class VAD:
        def __call__(self, samples, rate): return .6 if samples[0] else 0.0
        def reset_states(self): pass
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda samples: samples))
    monkeypatch.setitem(sys.modules, 'silero_vad', SimpleNamespace(load_silero_vad=VAD))
    speaking = []
    voice = VoiceInput.__new__(VoiceInput)
    voice.session = SimpleNamespace(config=config, wake=wake, state=SimpleNamespace(mic_enabled=True),
        set_user_speaking=speaking.append, status=RuntimeStatus, voice_anchor=lambda: None)
    voice.closed, voice.jobs, voice._last_overflow = threading.Event(), queue.Queue(), 0
    class Frames(queue.Queue):
        def get(self, **kwargs):
            if self.empty():
                voice.closed.set()
                raise queue.Empty
            return super().get(**kwargs)
    voice.frames = Frames()
    for index in range(3): voice.frames.put((bytes([1, 0]) * 512, 10 + index * .032))
    try:
        voice._run()
        assert speaking == [heard] * 3
    finally: wake.close()


def gated_run(tmp_path, monkeypatch, voice_raw, frames, playback, anchor=None):
    """VoiceInput._run over (int16 value, timestamp) frames with Riko's playback level given; the fake VAD calls any nonzero
    frame speech. Returns what the session, the wake detector and ASR saw."""
    from process.app_core.audio.voice_input import SILENT
    config = SimpleNamespace(root=tmp_path, paths=DataPaths.at(tmp_path), character_name='Riko', raw=NoRaw(),
        **audio_sections({'voice': {'mode': 'continuous', **voice_raw}}))
    class VAD:
        def __call__(self, samples, rate): return .95 if samples[0] else 0.0
        def reset_states(self): pass
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda samples: samples))
    monkeypatch.setitem(sys.modules, 'silero_vad', SimpleNamespace(load_silero_vad=VAD))
    seen = SimpleNamespace(speaking=[], fed=[], activity=[])
    wake = SimpleNamespace(feed=lambda frame, speaking: seen.fed.append(frame == SILENT), calibrating=False, testing=False, capture_state=lambda: (True, None))
    voice = VoiceInput.__new__(VoiceInput)
    voice.session = SimpleNamespace(config=config, wake=wake, state=SimpleNamespace(mic_enabled=True), speech=SimpleNamespace(playback=playback),
        set_user_speaking=seen.speaking.append, status=lambda: RuntimeStatus(voice_phase='capturing'), voice_anchor=lambda: anchor,
        voice_activity=lambda seconds, anchor: seen.activity.append(seconds))
    voice.closed, voice.jobs, voice._last_overflow = threading.Event(), queue.Queue(), 0
    voice._partial_lock, voice._partial_pending = threading.Lock(), set()
    class Frames(queue.Queue):
        def get(self, **kwargs):
            if self.empty():
                voice.closed.set()
                raise queue.Empty
            return super().get(**kwargs)
    voice.frames = Frames()
    for value, timestamp in frames: voice.frames.put((value.to_bytes(2, 'little', signed=True) * 512, timestamp))
    voice._run()
    seen.segments = list(voice.jobs.queue)
    return seen


def test_stop_only_hears_nothing_while_riko_plays_and_her_echo_never_becomes_pre_roll(tmp_path, monkeypatch):
    from process.app_core.audio.echo_gate import PlaybackLevel, REACH
    playback = PlaybackLevel()
    playback.wrote(.2, .5, now=10)  # she is heard until 10.5, her echo for voice.echo_tail_seconds (.3) more
    gated = [(3000, 10 + index * .032) for index in range(28)]  # up to 10.864: her echo, loud and speech-like
    after = [(3000, 10.95 + index * .032) for index in range(8)] + [(0, 11.25 + index * .032) for index in range(12)]
    assert gated[-1][1] < 10.5 + .3 + REACH < after[0][1]
    seen = gated_run(tmp_path, monkeypatch, {'barge_in': 'stop_only'}, gated + after, playback, anchor=('turn', 0, True))
    assert seen.speaking[:28] == [False] * 28 and seen.fed[:28] == [True] * 28
    assert seen.speaking[28:36] == [True] * 8 and len(seen.activity) == 8  # once she stopped, speech counts again
    segment = next(segment for segment in seen.segments if segment.pcm)
    pre_roll = 28 * 1024  # every gated frame went to the pre-roll as silence
    assert segment.pcm[:pre_roll] == bytes(pre_roll) and segment.pcm[pre_roll:pre_roll + 2] == (3000).to_bytes(2, 'little')


def test_echo_gate_passes_the_user_over_her_and_drops_frames_no_louder_than_her_echo(tmp_path, monkeypatch):
    from process.app_core.audio.echo_gate import MARGIN, PlaybackLevel
    playback = PlaybackLevel()
    playback.wrote(.2, 10, now=10, run='turn')
    # Her echo reaches the microphone at .1 of her level (.02 here), so the user needs MARGIN * .02.
    echo, user = 600, 3000  # RMS .018 and .092
    assert echo / 32768 < MARGIN * .02 < user / 32768
    frames = [(echo if index % 2 else user, 10.2 + index * .032) for index in range(20)]
    seen = gated_run(tmp_path, monkeypatch, {'barge_in': 'echo_gate', 'echo_coupling': .1}, frames, playback, anchor=('turn', 0, True))
    assert seen.speaking == [index % 2 == 0 for index in range(20)] and seen.fed == [index % 2 == 1 for index in range(20)]
    assert len(seen.activity) == 10  # only the user's frames count toward voice.interruption_seconds


def test_full_duplex_feeds_every_frame_as_heard_and_never_reads_playback(tmp_path, monkeypatch):
    class Untouchable:
        def __getattr__(self, name): raise AssertionError('full_duplex must not read playback')
    frames = [(3000, 10 + index * .032) for index in range(5)]
    seen = gated_run(tmp_path, monkeypatch, {}, frames, Untouchable(), anchor=('turn', 0, True))
    assert seen.speaking == [True] * 5 and seen.fed == [False] * 5 and len(seen.activity) == 5
