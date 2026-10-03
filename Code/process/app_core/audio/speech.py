"""Ordered sentence playback, independent of the model generation thread."""
import queue
import threading
import time
import logging
from dataclasses import dataclass, field
from ..runtime.workers import DaemonExecutor

from ..events.bus import event_bus

logger = logging.getLogger(__name__)


@dataclass
class AudioClip:
    path: object
    volume: float
    max_seconds: float
    context: dict = field(default_factory=dict)
    expires_at: float = field(default_factory=lambda: time.monotonic() + 2)


class SpeechQueue:
    def __init__(self, config, state):
        self.config, self.state = config, state
        self.queue = queue.Queue(maxsize=64)
        self.closed = threading.Event()
        concurrency = config.raw.get("sovits_ping_config", {}).get("max_in_flight_requests", 4)
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("sovits_ping_config.max_in_flight_requests must be a positive integer")
        self.requests = DaemonExecutor(max_workers=concurrency, thread_name_prefix="tts-request")
        self._submit_lock = threading.Lock()
        self._interrupt = threading.Event()
        self._futures = set()
        self._responses = {}
        self._response_lock = threading.Lock()
        self._output = None
        self._failed_turn = None
        self.last_error = ''
        self.worker = threading.Thread(target=self._run, daemon=True, name="sentence-playback")
        self.worker.start()

    def submit(self, text, turn_id, start_offset=0, end_offset=None):
        # Never send emoji/punctuation-only tails: SoVITS may yield no segments.
        if not any(character.isalnum() for character in text): return False
        if not self.state.audio_enabled or self.closed.is_set(): return False
        with self._submit_lock:
            if self.closed.is_set(): return False
            if self._failed_turn == turn_id: return False
            if self.queue.full():
                event_bus.publish("speech.error", turn_id=turn_id, error="Speech queue full")
                return False
            chunks = queue.Queue(maxsize=256)
            interrupt = self._interrupt
            future = self.requests.submit(self._receive, text, chunks, interrupt, turn_id)
            self._futures.add(future)
            def finished(completed):
                with self._submit_lock: self._futures.discard(completed)
            # A done callback can run inline; register it outside the submit lock.
            self.queue.put_nowait((text, turn_id, chunks, future, interrupt, start_offset, end_offset))
        future.add_done_callback(finished)
        return True

    def warmup(self):
        from types import SimpleNamespace
        # Exercise the configured remote voice once, discarding every PCM chunk.
        try:
            self._receive('This is a silent warmup.', SimpleNamespace(put=lambda *a, **k: None), self._interrupt)
        except Exception as exc:
            self.last_error = str(exc)
            event_bus.publish('speech.unavailable', error=self.last_error)
            raise  # Optional-component warmup catches this without aborting startup.
        return True

    def submit_clip(self, path, *, volume=.65, max_seconds=3.0, context=None):
        """Queue a local cue on the same output lane, without speech/history events."""
        if not self.state.audio_enabled or self.closed.is_set(): return False
        clip = AudioClip(path, volume, max_seconds, dict(context or {}))
        with self._submit_lock:
            if self.closed.is_set(): return False
            if self.queue.full():
                event_bus.publish('wake.feedback.error', asset='audio', error='Audio queue full; wake cue skipped')
                return False
            self.queue.put_nowait((clip, None, None, None, self._interrupt, 0, None))
        event_bus.publish('wake.feedback.queued', **clip.context)
        return True

    def _run(self):
        while not self.closed.is_set():
            try: text, turn_id, chunks, future, interrupt, start_offset, end_offset = self.queue.get(timeout=0.2)
            except queue.Empty: continue
            try:
                if isinstance(text, AudioClip):
                    self._play_clip(text, interrupt)
                    continue
                if self._failed_turn == turn_id:
                    future.cancel()
                    event_bus.publish('speech.cancelled', turn_id=turn_id)
                    continue
                started = time.perf_counter()
                metrics = self._play(text, turn_id, chunks, future, interrupt, start_offset, end_offset)
                if metrics is None:
                    event_bus.publish("speech.cancelled", turn_id=turn_id)
                    continue
                metrics.update(sentences=1, characters=len(text),
                               request_playback_seconds=round(time.perf_counter() - started, 3),
                               queued_sentences=self.queue.qsize())
                logger.info("TTS sentence: %s", metrics)
                event_bus.publish("speech.metrics", turn_id=turn_id, **metrics)
                event_bus.publish("speech.completed", turn_id=turn_id, end_offset=end_offset)
            except Exception as exc:
                if isinstance(text, AudioClip):
                    event_bus.publish('wake.feedback.error', asset='audio', error=str(exc), **text.context)
                else:
                    with self._submit_lock: self._failed_turn = turn_id
                    self.last_error = str(exc)
                    event_bus.publish("speech.error", turn_id=turn_id, error=self.last_error, chat_only=True)
            finally:
                # If the audio device fails, release a producer blocked on its buffer.
                while future is not None and not future.done() and not self.closed.is_set() and not interrupt.is_set():
                    try: chunks.get(timeout=0.1)
                    except queue.Empty: pass
                self.queue.task_done()

    def _play_clip(self, clip, interrupt):
        if interrupt.is_set() or self.closed.is_set() or not self.state.audio_enabled or time.monotonic() > clip.expires_at:
            return
        import soundfile as sf
        import sounddevice as sd
        import numpy as np
        with sf.SoundFile(clip.path) as source:
            if not 8000 <= source.samplerate <= 192000 or source.channels not in (1, 2):
                raise ValueError('Wake cue requires mono/stereo audio at 8–192 kHz')
            if not 0 < source.frames <= source.samplerate * clip.max_seconds:
                raise ValueError(f'Wake cue must contain audio and be at most {clip.max_seconds:g} seconds')
            pcm = np.clip(source.read(dtype='float32', always_2d=True) * clip.volume, -1, 1)
            rate, channels = source.samplerate, source.channels
        if interrupt.is_set() or self.closed.is_set() or time.monotonic() > clip.expires_at: return
        with sd.OutputStream(samplerate=rate, channels=channels, dtype='float32') as output:
            with self._response_lock: self._output = (output, interrupt)
            try:
                event_bus.publish('wake.feedback.started', **clip.context)
                for start in range(0, len(pcm), 512):
                    if interrupt.is_set() or self.closed.is_set() or not self.state.audio_enabled:
                        output.abort()
                        event_bus.publish('wake.feedback.cancelled', **clip.context)
                        return
                    output.write(pcm[start:start + 512])
                event_bus.publish('wake.feedback.completed', **clip.context)
            finally:
                with self._response_lock:
                    if self._output and self._output[0] is output: self._output = None

    def _receive(self, text, chunks, interrupt, turn_id=None):
        if interrupt.is_set() or self.closed.is_set(): return None
        if turn_id is not None and self._failed_turn == turn_id: return None
        import requests
        from .tts_http import request_payload
        url, payload = request_payload(self.config, text)
        request_start = time.perf_counter()
        first_audio = None
        try:
            response = requests.post(url, json=payload, stream=True, timeout=(5, 30))
        except requests.RequestException as exc:
            raise RuntimeError(f'GPT-SoVITS unavailable at {url}. Continuing in chat-only mode; speech will retry on the next reply. {exc}') from exc
        with self._response_lock:
            if interrupt.is_set() or self.closed.is_set():
                response.close()
                return None
            self._responses[response] = interrupt
        try:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=2048):
                if not chunk: continue
                if first_audio is None: first_audio = time.perf_counter() - request_start
                while not self.closed.is_set() and not interrupt.is_set():
                    try:
                        chunks.put(chunk, timeout=0.2)
                        break
                    except queue.Full: continue
                if self.closed.is_set() or interrupt.is_set(): return None
            return first_audio
        finally:
            with self._response_lock: self._responses.pop(response, None)
            response.close()

    def _play(self, text, turn_id, chunks, future, interrupt, start_offset=0, end_offset=None):
        if interrupt.is_set(): return None
        # Check synthesis before opening an audio device. A missing server should
        # report its connection error, not initialize playback or fail the chat.
        while chunks.empty():
            if interrupt.is_set() or self.closed.is_set(): return None
            if future.done():
                future.result()
                if not chunks.empty(): break
                raise RuntimeError('GPT-SoVITS returned no audio')
            interrupt.wait(.02)
        import sounddevice as sd
        rate = int(self.config.raw.get("sovits_ping_config", {}).get("sample_rate", 32000))
        with sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16") as output:
            with self._response_lock: self._output = (output, interrupt)
            try:
                return self._write_audio(output, text, turn_id, chunks, future, interrupt, rate, start_offset, end_offset)
            finally:
                with self._response_lock:
                    if self._output and self._output[0] is output: self._output = None

    def _write_audio(self, output, text, turn_id, chunks, future, interrupt, rate, start_offset=0, end_offset=None):
        pending, audio_bytes = b"", 0
        cancelled = False
        try:
            while not self.closed.is_set() and not interrupt.is_set():
                try: chunk = chunks.get(timeout=0.1)
                except queue.Empty:
                    if future.done(): break
                    continue
                if not self.state.audio_enabled: cancelled = True
                if cancelled: continue  # Drain producers so subsequent requests cannot deadlock.
                pending += chunk
                size = len(pending) // 2 * 2
                if size:
                    if not audio_bytes:
                        self.last_error = ''
                        event_bus.publish("speech.started", turn_id=turn_id, text=text,
                                          start_offset=start_offset, end_offset=end_offset,
                                          started_at=time.monotonic())
                    output.write(pending[:size])
                    audio_bytes += size
                    pending = pending[size:]
            if self.closed.is_set() or interrupt.is_set(): output.abort()
        except Exception:
            if self.closed.is_set() or interrupt.is_set(): return None
            raise
        if self.closed.is_set() or interrupt.is_set() or cancelled: return None
        first_audio = future.result()
        if not audio_bytes:
            raise RuntimeError("GPT-SoVITS returned no audio")
        return {"first_audio_seconds": round(first_audio, 3),
                "audio_seconds": round(audio_bytes / (rate * 2), 3)}

    def cancel(self):
        with self._submit_lock:
            self._interrupt.set()
            self._interrupt = threading.Event()
            futures = list(self._futures)
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty: break
        # Never cancel futures under the lock used by their completion callback.
        for future in futures: future.cancel()
        with self._response_lock:
            responses = [response for response, interrupt in self._responses.items() if interrupt.is_set()]
            output = self._output[0] if self._output and self._output[1].is_set() else None
        # Closing a Requests stream can wait for an ongoing socket read. Never
        # block VAD/capture (or hold the session lock) on transport cleanup.
        def cleanup():
            if output is not None:
                try: output.abort()
                except Exception: logger.exception("Unable to abort audio playback")
            for response in responses:
                try: response.close()
                except Exception: logger.exception("Unable to close cancelled TTS response")
        cleanup_thread = threading.Thread(target=cleanup, daemon=True, name="tts-cancel")
        cleanup_thread.start()
        return cleanup_thread

    def close(self):
        self.cancel()
        with self._submit_lock:
            self.closed.set()
        self.requests.shutdown(wait=False, cancel_futures=True)
