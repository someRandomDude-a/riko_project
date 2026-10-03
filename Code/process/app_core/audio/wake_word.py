"""Few-shot EfficientWord-Net enrollment; no ASR runs while waiting for wake."""
from collections import deque
from ..runtime.workers import DaemonExecutor
from hashlib import sha256
import json
import os
import threading
import time

from ..events.bus import event_bus
from .wake_capture import WakeCapture, prepare_audio


def profile_key(phrase, device):
    identity = {"phrase": " ".join(phrase.casefold().split()), "device": device,
                "model": "efficientword-resnet50-arc", "schema": 2}
    return sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class WakeWord:
    def __init__(self, config):
        settings = config.raw.get("voice", {})
        self.mode = settings.get("mode", "wake_word")
        if self.mode not in {"wake_word", "continuous", "manual"}:
            raise ValueError("voice.mode must be wake_word, continuous or manual")
        self.phrase = settings.get("wake_word", config.character_name).strip()
        if not self.phrase or len(self.phrase.split()) != 1:
            raise ValueError("This enrollment detector requires a single short wake name")
        self.threshold = float(settings.get("wake_threshold", .9))
        self.config_threshold = self.threshold
        self.followup = float(settings.get("follow_up_seconds", 10))
        if not 0 < self.threshold < 1 or not 0 < self.followup <= 300:
            raise ValueError("Invalid wake threshold or follow-up duration")
        self.directory = config.root / "persistent_memories" / "wake_words"
        self.lock = threading.RLock()
        self.worker = DaemonExecutor(max_workers=1, thread_name_prefix="wake-detector", max_pending=2)
        self.model = None
        self.embeddings = []
        self.samples = []
        self.window = deque(maxlen=47)  # 1.5 seconds, 512-sample input blocks.
        self.device = None
        self.path = None
        self.job = None
        self.recording = None
        self.deadline = 0.0
        self.capture_boundary = None
        self.engaged = False
        self.calibrating = False
        self.closed = False
        self.testing = False
        self.error = ""
        self.counter = 0
        self.live_capture = WakeCapture()
        self.last_score = None
        self.pending_pcm = None
        self.expiry_timer = None
        self.expiry_revision = 0

    def bind_device(self, identity):
        with self.lock:
            if self.device == identity: return
            self.device = identity
            self.path = self.directory / (profile_key(self.phrase, identity) + '.json')
            self.embeddings, self.samples = [], []
            self.threshold = self.config_threshold
            self.error = ''
            self.testing = False
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding='utf8'))
                    if data.get('key') != self.path.stem: raise ValueError('Enrollment identity mismatch')
                    if len(data['embeddings']) < 6: raise ValueError('Incomplete enrollment')
                    saved_threshold = float(data.get('threshold', self.config_threshold))
                    if not 0 < saved_threshold < 1: raise ValueError('Invalid saved wake threshold')
                    self.embeddings = data['embeddings']
                    self.samples = list(self.embeddings)
                    self.threshold = saved_threshold
                except Exception as exc: self.error = str(exc)
            self.window.clear()
            self.live_capture = WakeCapture()
            self.pending_pcm = None
        self.publish()

    def status(self):
        with self.lock:
            return {"mode": self.mode, "wake_word": self.phrase, "device": self.device,
                    "enrolled": bool(self.embeddings), "samples": len(self.samples),
                    "required_samples": 6, "calibrating": self.calibrating,
                    "recording": self.recording is not None,
                    "processing": self.job is not None and not self.job.done(),
                    "active": self.active(), "error": self.error,
                    "last_score": self.last_score, "threshold": self.threshold,
                    "testing": getattr(self, 'testing', False),
                    "sample_max_seconds": 5, "speech_detected": bool(self.recording and self.recording.started)}

    def publish(self): event_bus.publish('voice.wake_status', **self.status())

    def active(self):
        with self.lock:
            return not self.calibrating and (self.mode == 'continuous' or self.engaged or time.monotonic() < self.deadline)

    def capture_state(self):
        # Read these together: detection can activate from a different worker.
        with self.lock:
            return self.active(), self.capture_boundary

    def activate(self, *, after_keyword=False):
        with self.lock:
            if after_keyword and (self.closed or self.calibrating or self.testing or self.active()): return
            now = time.monotonic()
            self.deadline = now + self.followup
            # Audio already queued at activation belongs to the detector, not ASR.
            self.capture_boundary = (now, after_keyword)
            self._schedule_expiry()
        event_bus.publish('voice.activated', wake_word=self.phrase, source='keyword' if after_keyword else 'manual')
        self.publish()

    def _schedule_expiry(self):
        if self.expiry_timer: self.expiry_timer.cancel()
        self.expiry_revision += 1
        revision = self.expiry_revision
        def expired():
            with self.lock:
                if self.closed or revision != self.expiry_revision or self.engaged: return
            self.publish()
            event_bus.publish('voice.waiting')
        self.expiry_timer = threading.Timer(max(0, self.deadline - time.monotonic()), expired)
        self.expiry_timer.daemon = True
        self.expiry_timer.start()

    def responding(self):
        with self.lock:
            self.engaged = True
            self.expiry_revision += 1
            if self.expiry_timer: self.expiry_timer.cancel()

    def response_finished(self):
        with self.lock:
            self.engaged = False
            self.deadline = time.monotonic() + self.followup
            self._schedule_expiry()
        event_bus.publish('voice.follow_up', seconds=self.followup)
        self.publish()

    def begin_calibration(self):
        with self.lock:
            if self.device is None: raise ValueError('Start microphone capture first')
            self.calibrating = True
            self.testing = False
            self.samples = list(self.embeddings)
            self.error = ''
            self.window.clear()
            self.live_capture = WakeCapture()
            self.pending_pcm = None
        self.publish()

    def record(self):
        with self.lock:
            if not self.calibrating: raise ValueError('Start calibration first')
            if self.recording is not None or (self.job and not self.job.done()):
                raise ValueError('Recording or processing already in progress')
            self.recording = WakeCapture()
            self.error = ''
        self.publish()

    def finish_calibration(self):
        with self.lock:
            if self.recording is not None or (self.job and not self.job.done()):
                raise ValueError('Wait until recording has finished')
            if len(self.samples) < 6: raise ValueError('Record all six samples first')
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'key': self.path.stem, 'embeddings': self.samples,
                                            'model_type': 'resnet_50_arc', 'threshold': self.threshold}), encoding='utf8')
            os.replace(temporary, self.path)
            self.embeddings = list(self.samples)
            self.calibrating = False
            self.deadline = 0
            self.window.clear()
        self.publish()

    def set_threshold(self, threshold):
        threshold = float(threshold)
        if not 0 < threshold < 1: raise ValueError('Threshold must be between 0 and 1')
        with self.lock:
            self.threshold = threshold
            if self.path and self.embeddings:
                data = json.loads(self.path.read_text(encoding='utf8'))
                data['threshold'] = threshold
                temporary = self.path.with_suffix('.tmp')
                temporary.write_text(json.dumps(data), encoding='utf8')
                os.replace(temporary, self.path)
        self.publish()

    def set_testing(self, enabled):
        with self.lock:
            if self.recording is not None: raise ValueError('Finish recording first')
            if enabled and not self.embeddings: raise ValueError('Save an enrollment first')
            if enabled and self.calibrating: raise ValueError('Save or cancel sample collection first')
            self.testing = bool(enabled)
            self.live_capture = WakeCapture()
            self.window.clear()
        self.publish()

    def cancel_calibration(self):
        with self.lock:
            if self.job and not self.job.done(): raise ValueError('Wait for sample processing to finish')
            self.calibrating = False
            self.recording = None
            self.samples = list(self.embeddings)
        self.publish()

    def feed(self, frame, speaking=False):
        with self.lock:
            if self.closed: return
            if self.recording is not None:
                outcome, pcm = self.recording.feed(frame, speaking)
                if outcome == 'timeout':
                    self.recording = None
                    self.error = 'Five-second limit reached; sample discarded. Click Record and try again.'
                    event_bus.publish('voice.calibration_discarded', error=self.error)
                    self.publish()
                elif outcome == 'complete':
                    self.recording = None
                    self.job = self.worker.submit(self._enroll, pcm)
                return
            testing = getattr(self, 'testing', False)
            if self.calibrating or (self.mode != 'wake_word' and not testing) or (self.active() and not testing):
                self.live_capture = WakeCapture()
                return
            outcome, pcm = self.live_capture.feed(frame, speaking)
            if outcome:
                self.live_capture = WakeCapture()
                if outcome == 'complete':
                    if self.job is not None and not self.job.done():
                        self.pending_pcm = pcm
                    else:
                        self._schedule_detection(pcm)
                    return
            self.window.append(frame)
            self.counter += 1
            if not self.embeddings or len(self.window) < 47 or self.counter % 8: return
            if self.job and not self.job.done(): return
            self._schedule_detection(b''.join(self.window))

    def _schedule_detection(self, pcm):
        self.job = self.worker.submit(self._detect, pcm)
        def completed(_future):
            with self.lock:
                pending, self.pending_pcm = self.pending_pcm, None
                if pending and not self.closed and not self.calibrating and (getattr(self, 'testing', False) or not self.active()):
                    self._schedule_detection(pending)
        self.job.add_done_callback(completed)

    def _backend(self):
        if self.model is None:
            from eff_word_net.audio_processing import Resnet50_Arc_loss
            self.model = Resnet50_Arc_loss()
        return self.model

    def _enroll(self, pcm):
        try:
            model = self._backend()
            audio = prepare_audio(pcm, model.window_frames)
            vector = model.audioToVector(audio).reshape(-1).tolist()
            with self.lock:
                if self.closed or not self.calibrating: return
                self.samples.append(vector)
            event_bus.publish('voice.calibration_sample', samples=len(self.samples))
        except Exception as exc:
            with self.lock: self.error = str(exc)
            event_bus.publish('voice.error', error=f'Wake enrollment: {exc}')
        finally: self.publish()

    def _detect(self, pcm):
        try:
            import numpy as np
            model = self._backend()
            try: audio = prepare_audio(pcm, model.window_frames)
            except ValueError: return  # Quiet or non-wake-length audio is normal while waiting.
            vector = model.audioToVector(audio)
            score = float(model.scoreVector(vector, np.array(self.embeddings, dtype='float32')))
            with self.lock:
                if self.closed or self.calibrating: return
                self.last_score = score
            event_bus.publish('voice.wake_score', score=score, threshold=self.threshold)
            event_bus.publish('voice.wake_test', score=score, threshold=self.threshold, matched=score >= self.threshold)
            if score >= self.threshold: self.activate(after_keyword=True)
        except Exception as exc:
            with self.lock:
                first_error = not self.error
                self.error = str(exc)
            if first_error: event_bus.publish('voice.error', error=f'Wake detector: {exc}')

    def close(self):
        with self.lock:
            self.closed = True
            self.expiry_revision += 1
            if self.expiry_timer: self.expiry_timer.cancel()
        self.worker.shutdown(wait=False, cancel_futures=True)
