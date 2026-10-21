"""Python capture, VAD, incremental ASR and turn dispatch are separate workers."""
import logging
import queue
import threading
import time
from ..kernel.workers import DaemonExecutor

from ..kernel.audio_config import FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE
from ..kernel.cancellation import TurnCancelled
from ..events.bus import event_bus
from .echo_gate import EchoGate
from .voice_segments import VoiceSegments

logger = logging.getLogger(__name__)
SILENT = bytes(FRAME_BYTES)  # what the wake detector and the segmenter get for a frame of Riko's own echo
# In echo_gate or stop_only, an utterance among her gated echo with less speech than this is an echo fragment that slipped
# through, never transcribed: Whisper turns a few frames padded with silence into 'you' or 'Yeah.'.
MIN_VOICED = .2


class VoiceInput:
    def __init__(self, session):
        self.session = session
        self.frames = queue.Queue(maxsize=256)
        self.jobs = queue.Queue(maxsize=64)
        self.closed = threading.Event()
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
            device_id = self.session.config.voice.input_device
            device = sd.query_devices(device_id, 'input')
            hostapi = sd.query_hostapis(device['hostapi'])['name']
            self.session.wake.bind_device({'name': device['name'], 'hostapi': hostapi,
                                          'channels': device['max_input_channels'], 'sample_rate': SAMPLE_RATE})
            def callback(data, frames, timing, status):
                self.feed(bytes(data))
            with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES,
                                   device=device_id, callback=callback):
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
            voice = self.session.config.voice  # checked once by load_config (VoiceConfig.from_raw), not per frame
            from copy import deepcopy
            warmed = getattr(self.session, 'warmed_vad', None)
            vad = deepcopy(warmed) if warmed is not None else load_silero_vad()
            def on_start():
                anchor = self.session.voice_anchor()
                event_bus.publish("voice.started", utterance_id=segmenter.utterance_id, speaking_over=bool(anchor and (len(anchor) < 3 or anchor[2])))
                return anchor
            def activity(seconds, anchor):
                if self.session.status().voice_phase != 'capturing': event_bus.publish('voice.resumed', utterance_id=segmenter.utterance_id)
                if anchor is not None: self.session.voice_activity(seconds, anchor)
            segmenter = VoiceSegments(self._enqueue, on_start, activity, pre_roll=voice.pre_roll_seconds,
                gap=voice.transcription_gap_seconds, endpoint=voice.utterance_end_seconds, max_segment=voice.max_segment_seconds,
                partial_interval=voice.live_transcript_interval_seconds)
            # voice.barge_in: the default never looks at playback; the others gate frames heard while Riko's audio plays.
            gate = EchoGate(voice, self.session.speech.playback if voice.barge_in != 'full_duplex' else None)
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
                for index in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
                    frame = data[index:index + FRAME_BYTES]
                    samples = np.frombuffer(frame, dtype='<i2').astype('float32') / 32768
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    if timestamp - last_level >= 0.05:
                        event_bus.publish("voice.level", rms=rms, peak=float(np.max(np.abs(samples))))
                        last_level = timestamp
                    if not self.session.state.mic_enabled:
                        self.session.set_user_speaking(False)
                        segmenter.reset()
                        vad.reset_states()
                        continue
                    with torch.inference_mode(): probability = float(vad(torch.from_numpy(samples), SAMPLE_RATE))
                    speaking, echo = gate.decide(probability, rms, timestamp)
                    heard = SILENT if echo else frame  # her echo never becomes an utterance's audio or pre-roll
                    self.session.set_user_speaking(speaking)
                    # A tool may claim speaking priority while an utterance that
                    # began before the reply is still being captured.
                    if segmenter.utterance_id and segmenter.anchor is None and self.session.status().speaking_priority:
                        segmenter.anchor = self.session.voice_anchor()
                    self.session.wake.feed(heard, speaking)
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
                    segmenter.feed(heard, speaking, timestamp)
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
                if not segment.provisional: event_bus.publish('voice.transcribing', utterance_id=segment.utterance_id)
                parts = self._parts.setdefault(segment.utterance_id, [])
                partial_text = ''
                if segment.pcm and not self._echo_fragment(segment):
                    # The session's one Whisper model, shared with warmup and Discord (audio/asr.py AsrService).
                    text = self.session.transcribe(segment.pcm, beam_size=1, vad_filter=False, condition_on_previous_text=False)
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
                        # Place words spoken during a reply now, even if the dispatch worker is still waiting on an
                        # earlier turn; the session says whether they need a reply (SessionManager.voice_transcript).
                        disposition = self.session.voice_transcript(text, segment.started_at, segment.ended_at, segment.anchor) if segment.anchor else 'fresh'
                        if disposition == 'fresh': self.responses.submit(self._dispatch, text, segment)
                        elif disposition == 'reply': self.responses.submit(self._dispatch, text, segment, segment.anchor[0])
            except Exception as exc:
                logger.exception("Transcription failed")
                self._parts.pop(segment.utterance_id, None)
                if not self.closed.is_set(): event_bus.publish("voice.error", error=str(exc))
            finally:
                if segment.provisional:
                    with self._partial_lock: self._partial_pending.discard(segment.utterance_id)
                self.jobs.task_done()

    def _echo_fragment(self, segment):
        """A segment of an utterance with under MIN_VOICED of speech whose audio holds frames gated as her echo (SILENT)."""
        if segment.speech_seconds >= MIN_VOICED or self.session.config.voice.barge_in == 'full_duplex': return False
        pcm = segment.pcm
        return any(pcm[index:index + FRAME_BYTES] == SILENT for index in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES))

    def _abandoned(self):
        return self.closed.is_set() or self.session.wake.calibrating or getattr(self.session.wake, 'testing', False)

    def _dispatch(self, text, segment, reply_to=None):
        try:
            if self._abandoned(): return
            # respond() waits for a running turn or initiative on this worker (never the ASR or microphone workers), so a
            # finished utterance is answered after it instead of being refused and lost. reply_to redoes a reply it cut.
            self.session.respond(text, record_user=reply_to is None, reply_to=reply_to, wait=self._abandoned,
                origin={'source':'microphone', 'message_id':segment.utterance_id})
        except TurnCancelled:
            pass
        except Exception as exc:
            event_bus.publish("voice.error", error=str(exc))

    def close(self):
        self.closed.set()
        self.responses.shutdown(wait=False, cancel_futures=True)
