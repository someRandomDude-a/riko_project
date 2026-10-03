"""Opt-in, per-speaker PCM buffers. Packet callbacks never run ASR/inference."""
import threading

MAX_CALL_SECONDS = 15
MAX_CALL_BYTES = 48000 * 2 * 2 * MAX_CALL_SECONDS


def downsample_call(pcm):
    import numpy as np
    from scipy.signal import resample_poly
    audio = np.frombuffer(pcm, dtype='<i2').reshape(-1, 2).astype('float32').mean(axis=1)
    return np.clip(resample_poly(audio, 1, 3), -32768, 32767).astype('<i2').tobytes()


class VoiceCapture:
    def __init__(self, loop, accepts, deliver, *, endpoint=.8, activity=lambda user, seconds: None):
        self.loop, self.accepts, self.deliver, self.endpoint = loop, accepts, deliver, endpoint
        self.consent, self.buffers, self.timers = set(), {}, {}
        self.closed = False
        self.activity = activity
        self.pending = threading.BoundedSemaphore(128)

    def allow(self, user_id):
        if len(self.consent) >= 8 and user_id not in self.consent: raise ValueError('At most eight consenting speakers per call')
        self.consent.add(user_id)

    def revoke(self, user_id):
        self.consent.discard(user_id)
        self.buffers.pop(user_id, None)
        timer = self.timers.pop(user_id, None)
        if timer: timer.cancel()

    def write(self, user, data):
        if self.closed or user is None or user.bot or not data or len(data) > 38400 or len(data) % 4: return
        if not self.pending.acquire(blocking=False): return
        try: self.loop.call_soon_threadsafe(self._packet, user, bytes(data))
        except RuntimeError: self.pending.release()

    def _packet(self, user, data):
        self.pending.release()
        if self.closed or user.id not in self.consent or not self.accepts(user): return
        entry = self.buffers.setdefault(user.id, (user, bytearray()))
        pcm = entry[1]
        if len(pcm) + len(data) > MAX_CALL_BYTES:
            self.flush(user.id)
            entry = self.buffers.setdefault(user.id, (user, bytearray()))
        entry[1].extend(data)
        self.activity(user, len(entry[1]) / (48000 * 4))
        old = self.timers.pop(user.id, None)
        if old: old.cancel()
        self.timers[user.id] = self.loop.call_later(self.endpoint, self.flush, user.id)

    def flush(self, user_id):
        timer = self.timers.pop(user_id, None)
        if timer: timer.cancel()
        entry = self.buffers.pop(user_id, None)
        if self.closed or user_id not in self.consent or not entry or not self.accepts(entry[0]): return
        if len(entry[1]) >= 48000 * 4 * .15: self.deliver(entry[0], bytes(entry[1]))

    def close(self):
        self.closed = True
        for timer in self.timers.values(): timer.cancel()
        self.timers.clear(); self.buffers.clear(); self.consent.clear()


def receive_extension():
    try:
        from discord.ext import voice_recv
    except ImportError as exc:
        raise ValueError('Call transcription requires a DAVE-compatible discord-ext-voice-recv build; uploaded-audio STT still works') from exc
    # Older builds decode encrypted DAVE packets as Opus noise. Fail closed,
    # rather than recording/feeding corrupted audio or downgrading encryption.
    import inspect
    from discord.ext.voice_recv import opus
    try: source = inspect.getsource(opus)
    except (OSError, TypeError): source = ''
    if 'dave_session' not in source:
        raise ValueError('Installed voice-receive extension lacks DAVE decryption; update it before enabling call STT')
    return voice_recv
