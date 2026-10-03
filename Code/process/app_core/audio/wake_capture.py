"""Small VAD recorder shared by wake enrollment and live phrase matching."""
from collections import deque


class WakeCapture:
    def __init__(self, max_seconds=5.0, silence_seconds=0.32):
        self.max_seconds, self.silence_seconds = max_seconds, silence_seconds
        self.elapsed = self.silence = 0.0
        self.frames = []
        self.started = False
        self.pre_roll = deque(maxlen=4)

    def feed(self, frame, speaking):
        self.elapsed += len(frame) / 32000
        # Deadline takes precedence over endpoint: never enroll a timed-out clip.
        if self.elapsed >= self.max_seconds:
            self.frames.clear()
            return 'timeout', None
        if not self.started:
            if not speaking:
                self.pre_roll.append(frame)
                return None, None
            self.started = True
            self.frames.extend(self.pre_roll)
        self.frames.append(frame)
        self.silence = 0.0 if speaking else self.silence + len(frame) / 32000
        if self.silence >= self.silence_seconds:
            return 'complete', b''.join(self.frames)
        return None, None


def prepare_audio(pcm, window_frames):
    """Identical deterministic trim/centering for references AND live scoring."""
    import numpy as np
    audio = np.frombuffer(pcm, dtype='<i2').astype('float32') / 32768
    if not len(audio): raise ValueError('Empty wake recording')
    # Frame energy is less sensitive to a single spike than sample peak trimming.
    blocks = [float(np.sqrt(np.mean(audio[start:start + 160] ** 2))) for start in range(0, len(audio), 160)]
    maximum = max(blocks)
    if maximum < .005: raise ValueError('Recording too quiet; check microphone')
    voiced = np.flatnonzero(np.array(blocks) > max(.003, maximum * .15))
    start, end = max(0, int(voiced[0]) * 160 - 800), min(len(audio), (int(voiced[-1]) + 1) * 160 + 800)
    audio = audio[start:end]
    if len(audio) > window_frames:
        raise ValueError('Wake name exceeds the detector’s 1.5-second window; say it once, briefly')
    pad = window_frames - len(audio)
    return np.pad(audio, (pad // 2, pad - pad // 2)).astype('float32')
