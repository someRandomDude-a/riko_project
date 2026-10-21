"""Startup preflight without microphone capture, history writes or audible output."""
import logging
import time

from ..events.bus import event_bus
from ..kernel.audio_config import FRAME_SAMPLES, SAMPLE_RATE
from ..kernel.workers import DaemonExecutor


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
    def asr(): session.asr.warm()  # builds the session's one Whisper model, which the microphone and Discord then use
    def vad():
        import torch
        from silero_vad import load_silero_vad
        model = load_silero_vad()
        model(torch.zeros(FRAME_SAMPLES), SAMPLE_RATE)
        model.reset_states()
        if session.is_open: session.warmed_vad = model
    def wake():
        import numpy as np
        model = session.wake._backend()
        model.audioToVector(np.zeros(model.window_frames, dtype='float32'))
    jobs = [('asr', asr), ('vad', vad)]
    if session.wake.mode == 'wake_word' and not session.wake.unavailable: jobs.append(('wake_detector', wake))
    if session.state.audio_enabled: jobs.append(('tts', session.speech.warmup))
    warm_components(jobs, session.config.runtime.startup_timeout_seconds)
