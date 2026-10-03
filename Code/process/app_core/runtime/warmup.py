"""Startup preflight without microphone capture, history writes or audible output."""
import logging
import threading
import time

from ..events.bus import event_bus
from .workers import DaemonExecutor


def warm_components(jobs, timeout):
    executor = DaemonExecutor(max_workers=max(1, len(jobs)), thread_name_prefix='startup-warmup')
    def run(name, fn):
        event_bus.publish('runtime.warmup', component=name, status='loading')
        try:
            fn()
            event_bus.publish('runtime.warmup', component=name, status='ready')
        except Exception as exc:
            logging.getLogger(__name__).warning('Warmup %s unavailable: %s', name, exc)
            event_bus.publish('runtime.warmup', component=name, status='error', error=str(exc))
    futures = [executor.submit(run, name, fn) for name, fn in jobs]
    deadline = time.monotonic() + timeout
    try:
        for future in futures: future.result(timeout=max(.01, deadline - time.monotonic()))
    finally: executor.shutdown()


def warm_core(memory, emotion, timeout):
    jobs = []
    if memory.decider: jobs.append(('memory_classifier', lambda: memory.decider.decide('Startup warmup; not a memory.')))
    if memory.config.embeddings_enabled:
        jobs.append(('memory_embeddings', lambda: memory._embed(['Warmup'])))
    if emotion: jobs.append(('emotion', lambda: emotion._interpret('user', 'Startup warmup.')))
    warm_components(jobs, timeout)


def warm_session(session):
    session.asr_lock = threading.Lock()
    def asr():
        import numpy as np
        from faster_whisper import WhisperModel
        settings = session.config.raw.get('voice', {})
        model = WhisperModel(settings.get('asr_model', 'distil-small.en'), device=settings.get('asr_device', 'cuda'),
            compute_type=settings.get('asr_compute_type', 'int8_float16'))
        segments, _ = model.transcribe(np.zeros(16000, dtype='float32'), beam_size=1, vad_filter=False)
        list(segments)
        if not session._closed: session.warmed_asr = model
    def vad():
        import torch
        from silero_vad import load_silero_vad
        model = load_silero_vad()
        model(torch.zeros(512), 16000)
        model.reset_states()
        if not session._closed: session.warmed_vad = model
    def wake():
        import numpy as np
        model = session.wake._backend()
        model.audioToVector(np.zeros(model.window_frames, dtype='float32'))
    jobs = [('asr', asr), ('vad', vad)]
    if session.wake.mode == 'wake_word': jobs.append(('wake_detector', wake))
    if session.state.audio_enabled: jobs.append(('tts', session.speech.warmup))
    warm_components(jobs, session.config.runtime.startup_timeout_seconds)
