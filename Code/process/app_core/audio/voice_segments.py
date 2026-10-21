"""Device-independent VAD segmentation with pre-roll and incremental ASR jobs."""
from collections import deque
from dataclasses import dataclass
import math
import uuid

from ..kernel.audio_config import FRAME_SECONDS


@dataclass
class Segment:
    utterance_id: str
    pcm: bytes
    anchor: object
    started_at: float
    ended_at: float
    speech_seconds: float
    final: bool
    provisional: bool = False


class VoiceSegments:
    def __init__(self, on_segment, on_start, on_activity, *, pre_roll=1.0,
                  gap=0.3, endpoint=1.0, max_segment=15.0, partial_interval=None):
        if not all(math.isfinite(value) and value > 0 for value in (pre_roll, gap, endpoint, max_segment)):
            raise ValueError("Voice timing settings must be positive finite seconds")
        if gap >= endpoint: raise ValueError("Transcription gap must be shorter than utterance endpoint")
        if partial_interval is not None and (not math.isfinite(partial_interval) or not .5 <= partial_interval <= 10): raise ValueError('Live transcript interval must be between .5 and 10 seconds')
        self.partial_interval = partial_interval
        self.on_segment, self.on_start, self.on_activity = on_segment, on_start, on_activity
        self.gap, self.endpoint, self.max_segment = gap, endpoint, max_segment
        self.pre_roll = deque(maxlen=max(1, math.ceil(pre_roll / FRAME_SECONDS)))
        self.capture_boundary = None
        self.wait_for_silence = False
        self.activation_silence = 0.0
        self.reset()

    def activate(self, boundary):
        """Separate wake/manual activation audio from the next request's pre-roll."""
        if boundary == self.capture_boundary: return
        self.reset()
        self.capture_boundary = boundary
        self.wait_for_silence = bool(boundary and boundary[1])
        self.activation_silence = 0.0

    def reset(self):
        self.pre_roll.clear()
        self.activation_silence = 0.0
        self.utterance_id = None
        self.audio = []
        self.silence = self.voiced = self.segment_seconds = 0.0
        self.partial_elapsed = 0.0
        self.anchor = None

    def feed(self, frame, speaking, timestamp):
        if self.capture_boundary and timestamp - FRAME_SECONDS < self.capture_boundary[0]:
            return  # Discard queued frames, including one straddling activation.
        if self.wait_for_silence:
            self.activation_silence = 0.0 if speaking else self.activation_silence + FRAME_SECONDS
            if self.activation_silence >= 0.32: self.wait_for_silence = False
            return  # A rolling wake match may finish before the keyword does.
        if self.utterance_id is None:
            if not speaking:
                self.pre_roll.append(frame)
                return
            self.utterance_id = str(uuid.uuid4())
            self.started_at = timestamp - FRAME_SECONDS
            self.anchor = self.on_start()
            self.audio = list(self.pre_roll)
            self.pre_roll.clear()
        self.audio.append(frame)
        self.segment_seconds += FRAME_SECONDS
        self.partial_elapsed += FRAME_SECONDS
        if speaking:
            self.voiced += FRAME_SECONDS
            self.silence = 0.0
            self.ended_at = timestamp
            self.on_activity(self.voiced, self.anchor)
        else:
            self.silence += FRAME_SECONDS
        if self.silence >= self.endpoint:
            self._emit(True)
            self.reset()
        elif self.segment_seconds >= self.max_segment or (self.silence >= self.gap and self.audio):
            self._emit(False)
        elif self.partial_interval and self.partial_elapsed >= self.partial_interval:
            self.on_segment(Segment(self.utterance_id, b''.join(self.audio), self.anchor,
                self.started_at, self.ended_at, self.voiced, False, provisional=True))
            self.partial_elapsed = 0.0 # Re-decode the uncommitted window, not duplicate committed text.

    def _emit(self, final):
        # Don't transcribe pure silence repeatedly between the gap and endpoint.
        pcm = b"".join(self.audio) if self.segment_seconds > 0 and self.audio else b""
        if pcm and self.silence >= self.gap and self.segment_seconds <= self.silence:
            pcm = b""
        if pcm or final:
            self.on_segment(Segment(self.utterance_id, pcm, self.anchor,
                                    self.started_at, self.ended_at, self.voiced, final))
        self.audio = []
        self.segment_seconds = 0.0
        self.partial_elapsed = 0.0
