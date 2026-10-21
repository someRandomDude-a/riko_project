"""The avatar's switches and expression: PlaybackFlags (microphone, audio, volume, sleep) and Presence (the speech bubble,
which clears itself when it expires, and the latest emotion)."""
from __future__ import annotations

import math
import threading
import time


class PlaybackFlags:
    """Emits 'mic', 'audio', 'audio_volume' and 'sleep'."""
    __slots__ = ('events', '_lock', '_mic', '_audio', '_volume', '_sleep')

    def __init__(self, events):
        self.events, self._lock = events, threading.Lock()
        self._mic, self._audio, self._volume, self._sleep = True, True, 1.0, False

    mic_enabled = property(lambda self: self._mic)
    audio_enabled = property(lambda self: self._audio)
    audio_volume = property(lambda self: self._volume)
    sleep_mode = property(lambda self: self._sleep)

    def snapshot(self):
        with self._lock: return {"mic": self._mic, "audio": self._audio, "audio_volume": self._volume, "sleep": self._sleep}

    def toggle_mic(self):
        with self._lock: self._mic = enabled = not self._mic
        self.events.emit("mic", enabled)
        return enabled

    def set_mic(self, enabled):
        """Turn the microphone switch on or off; reported only when it changes."""
        with self._lock: changed, self._mic = self._mic != enabled, enabled
        if changed: self.events.emit("mic", enabled)
        return enabled

    def toggle_audio(self):
        with self._lock: self._audio = enabled = not self._audio
        self.events.emit("audio", enabled)
        return enabled

    def set_audio_volume(self, volume):
        if type(volume) not in (int, float) or not math.isfinite(volume) or not 0 <= volume <= 1:
            raise ValueError('Audio volume must be a finite number between 0 and 1')
        with self._lock: self._volume = volume = float(volume)
        self.events.emit('audio_volume', volume)
        return volume

    def set_sleep(self, enabled=True):
        with self._lock: self._sleep = enabled
        self.events.emit("sleep", enabled)
        return enabled


class Presence:
    """Emits 'speech' (also when a bubble expires) and 'emotion'."""
    __slots__ = ('events', '_lock', '_speech', '_until', '_timer', '_emotion')

    def __init__(self, events):
        self.events, self._lock = events, threading.Lock()
        self._speech, self._until, self._timer, self._emotion = '', 0.0, None, None

    def snapshot(self):
        with self._lock: return {"emotion": self._emotion.as_dict() if self._emotion else None, "speech": self._speech if self._until > time.time() else ""}

    def emotion_snapshot(self):
        with self._lock: return self._emotion.as_dict() if self._emotion else None

    def set_emotion(self, emotion):
        with self._lock: self._emotion = emotion
        self.events.emit("emotion", emotion)

    def set_speech(self, text: str, seconds: float = 12.0):
        with self._lock:
            if self._timer: self._timer.cancel()
            self._speech, self._until = text, time.time() + seconds
            self._timer = None
            if text and seconds > 0:
                self._timer = threading.Timer(seconds, self._expire, args=(self._until,))
                self._timer.daemon = True
                self._timer.start()
        self.events.emit("speech", text)

    def _expire(self, deadline):
        with self._lock:
            if self._until != deadline: return  # a newer bubble replaced it
            self._speech = ''
            self._timer = None
        self.events.emit('speech', '')
