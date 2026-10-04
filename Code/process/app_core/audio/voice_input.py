"""Python capture, VAD, incremental GPU ASR and turn dispatch are separate workers."""
import logging
import queue
import threading
import time
from ..runtime.workers import DaemonExecutor

from ..runtime.cancellation import TurnCancelled
from ..events.bus import event_bus
from .voice_segments import VoiceSegments

logger = logging.getLogger(__name__)


class VoiceInput:
    def __init__(self, session):
        self.session = session
        self.frames = queue.Queue(maxsize=256)
        self.jobs = queue.Queue(maxsize=64)
        self.closed = threading.Event()
        self.model = getattr(session, 'warmed_asr', None)
        self.asr_lock = getattr(session, 'asr_lock', threading.Lock())
        self.responses = DaemonExecutor(max_workers=1, thread_name_prefix="voice-turn", max_pending=8)
        self._last_overflow = 0
        self._parts = {}
        self._partial_lock = threading.Lock()
        self._partial_pending = set()
        for name, target in (("voice-vad", self._run), ("voice-asr", self._asr), ("microphone-capture", self._capture)):
            threading.Thread(target=target, daemon=True, name=name).start()

    def _capture(self):
        try:
            import sounddevice as sd
            settings = self.session.config.raw.get("voice", {})
            device = sd.query_devices(settings.get("input_device"), 'input')
            hostapi = sd.query_hostapis(device['hostapi'])['name']
            self.session.wake.bind_device({'name': device['name'], 'hostapi': hostapi,
                                          'channels': device['max_input_channels'], 'sample_rate': 16000})
            def callback(data, frames, timing, status):
                self.feed(bytes(data))
            with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", blocksize=512,
                                   device=settings.get("input_device"), callback=callback):
                self.closed.wait()
        except Exception as exc:
            if not self.closed.is_set(): event_bus.publish("voice.error", error=f"Microphone capture failed: {exc}")
            self.closed.set()
        finally:
            if getattr(self.session, 'voice', None) in (None, self): event_bus.publish("voice.stopped")

    def feed(self, data):
        if self.closed.is_set(): return
        try: self.frames.put_nowait((data, time.monotonic()))
        except queue.Full:
            # Never kill the microphone because inference or cancellation lagged.
            try: self.frames.get_nowait()
            except queue.Empty: pass
            try: self.frames.put_nowait((data, time.monotonic()))
            except queue.Full: pass
            self._last_overflow = time.monotonic()

    def _enqueue(self, segment):
        if segment.final: event_bus.publish('voice.utterance_ended', utterance_id=segment.utterance_id)
        if segment.provisional:
            with self._partial_lock:
                if self._partial_pending: return # At most one partial decode queued/running; capture never waits.
                self._partial_pending.add(segment.utterance_id)
        try: self.jobs.put_nowait(segment)
        except queue.Full:
            if segment.provisional:
                with self._partial_lock: self._partial_pending.discard(segment.utterance_id)
                return
            event_bus.publish("voice.error", error="Transcription backlog full; utterance discarded, microphone remains active")

    def _run(self):
        try:
            import numpy as np
            import torch
            from silero_vad import load_silero_vad
            settings = self.session.config.raw.get("voice", {})
            from copy import deepcopy
            warmed = getattr(self.session, 'warmed_vad', None)
            vad = deepcopy(warmed) if warmed is not None else load_silero_vad()
            def on_start():
                anchor = self.session.voice_anchor()
                event_bus.publish("voice.started", utterance_id=segmenter.utterance_id, speaking_over=bool(anchor and (len(anchor) < 3 or anchor[2])))
                return anchor
            def activity(seconds, anchor):
                if getattr(self.session, '_voice_phase', '') != 'capturing': event_bus.publish('voice.resumed', utterance_id=segmenter.utterance_id)
                if anchor is not None: self.session.voice_activity(seconds, anchor)
            segmenter = VoiceSegments(self._enqueue, on_start, activity,
                pre_roll=float(settings.get("pre_roll_seconds", 1.0)),
                gap=float(settings.get("transcription_gap_seconds", 0.3)),
                endpoint=float(settings.get("utterance_end_seconds", 1.0)),
                max_segment=float(settings.get("max_segment_seconds", 15.0)),
                partial_interval=float(settings.get('live_transcript_interval_seconds', 2.0)))
            last_level = overflow = 0.0
            if self.closed.is_set(): return
            event_bus.publish("voice.ready")
            while not self.closed.is_set():
                try: data, timestamp = self.frames.get(timeout=0.1)
                except queue.Empty: continue
                if overflow != self._last_overflow:
                    overflow = self._last_overflow
                    segmenter.reset()
                    vad.reset_states()
                    event_bus.publish("voice.error", error="Microphone processing fell behind; recording resumed")
                for index in range(0, len(data) - 1023, 1024):
                    frame = data[index:index + 1024]
                    samples = np.frombuffer(frame, dtype='<i2').astype('float32') / 32768
                    if timestamp - last_level >= 0.05:
                        event_bus.publish("voice.level", rms=float(np.sqrt(np.mean(samples ** 2))), peak=float(np.max(np.abs(samples))))
                        last_level = timestamp
                    if not self.session.state.mic_enabled:
                        with self.session._voice_lock: self.session._user_speaking = False
                        segmenter.reset()
                        vad.reset_states()
                        continue
                    with torch.inference_mode(): probability = float(vad(torch.from_numpy(samples), 16000))
                    speaking = probability >= float(settings.get("vad_threshold", 0.5))
                    with self.session._voice_lock: self.session._user_speaking = speaking
                    # A tool may claim speaking priority while an utterance that
                    # began before the reply is still being captured.
                    if segmenter.utterance_id and segmenter.anchor is None and time.monotonic() < self.session._assertive_until:
                        segmenter.anchor = self.session.voice_anchor()
                    self.session.wake.feed(frame, speaking)
                    if self.session.wake.calibrating or getattr(self.session.wake, 'testing', False):
                        segmenter.reset()
                        continue
                    active, boundary = self.session.wake.capture_state()
                    segmenter.activate(boundary)
                    # Keep active recordings intact while follow-up deadlines expire.
                    if not active and segmenter.utterance_id is None:
                        # Detector-only audio must never become transcription pre-roll.
                        segmenter.reset()
                        continue
                    segmenter.feed(frame, speaking, timestamp)
        except Exception as exc:
            logger.exception("Voice detection failed")
            if not self.closed.is_set(): event_bus.publish("voice.error", error=str(exc))
            self.closed.set()

    def _asr(self):
        while not self.closed.is_set():
            try: segment = self.jobs.get(timeout=0.1)
            except queue.Empty: continue
            try:
                if self.session.wake.calibrating or self.session.wake.testing:
                    self._parts.pop(segment.utterance_id, None)
                    continue
                import numpy as np
                from faster_whisper import WhisperModel
                config = self.session.config.raw.get("voice", {})
                if not segment.provisional: event_bus.publish('voice.transcribing', utterance_id=segment.utterance_id)
                parts = self._parts.setdefault(segment.utterance_id, [])
                partial_text = ''
                if segment.pcm:
                    audio = np.frombuffer(segment.pcm, dtype='<i2').astype('float32') / 32768
                    with self.asr_lock:
                        if self.model is None:
                            self.model = getattr(self.session, 'warmed_asr', None)
                            if self.model is None:
                                self.model = WhisperModel(config.get("asr_model", "distil-small.en"),
                                    device=config.get("asr_device", "cuda"), compute_type=config.get("asr_compute_type", "int8_float16"))
                                self.session.warmed_asr = self.model
                        segments, _ = self.model.transcribe(audio, beam_size=1, vad_filter=False,
                                                           condition_on_previous_text=False)
                        text = " ".join(item.text.strip() for item in segments).strip()
                    if text:
                        if segment.provisional: partial_text = text
                        else: parts.append(text)
                text = " ".join([*parts, *([partial_text] if partial_text else [])])
                if self.closed.is_set(): continue
                event_bus.publish("voice.transcript", utterance_id=segment.utterance_id,
                    text=text, final=segment.final, speaking_over=bool(segment.anchor and (len(segment.anchor) < 3 or segment.anchor[2])))
                if segment.final:
                    self._parts.pop(segment.utterance_id, None)
                    if text:
                        observer = getattr(getattr(self.session, 'state', None), 'observe_input', None)
                        if observer: observer('microphone', text, message_id=segment.utterance_id)
                        # Preserve interjections immediately, even if the reply
                        # dispatch worker is still waiting on an earlier turn.
                        accepted = True
                        if segment.anchor:
                            accepted = self.session.voice_transcript(text, segment.started_at, segment.ended_at, segment.anchor)
                        # Follow actual VAD cancellation, not a threshold re-evaluated
                        # after slow ASR or speaking-priority expiry.
                        if accepted and (segment.anchor is None or self.session.user_interrupted(segment.anchor)):
                            self.responses.submit(self._dispatch, text, segment)
            except Exception as exc:
                logger.exception("Transcription failed")
                self._parts.pop(segment.utterance_id, None)
                if not self.closed.is_set(): event_bus.publish("voice.error", error=str(exc))
            finally:
                if segment.provisional:
                    with self._partial_lock: self._partial_pending.discard(segment.utterance_id)
                self.jobs.task_done()

    def _dispatch(self, text, segment):
        try:
            if self.session.wake.calibrating or self.session.wake.testing: return
            anchor = segment.anchor
            # Waiting/generation must never block the ASR or microphone workers.
            while not self.closed.is_set():
                if self.session.wake.calibrating or self.session.wake.testing: return
                if self.session._turn_lock.acquire(timeout=0.1):
                    self.session._turn_lock.release()
                    if anchor and anchor[0] != self.session._active_turn:
                        event_bus.publish("voice.error", error="Interrupted turn changed before reply dispatch; transcript was preserved")
                        return
                    self.session.respond(text, record_user=anchor is None, origin={'source':'microphone', 'message_id':segment.utterance_id})
                    return
        except TurnCancelled:
            pass
        except Exception as exc:
            event_bus.publish("voice.error", error=str(exc))

    def close(self):
        self.closed.set()
        self.responses.shutdown(wait=False, cancel_futures=True)
