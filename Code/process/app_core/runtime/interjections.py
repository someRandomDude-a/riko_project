"""Voice activity policy; timestamps are monotonic seconds, not ASR latency."""
import math
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass

from ..conversation.messages import ChatMessage


@dataclass
class Interjection:
    text: str
    offset: int
    ended_at: float
    timestamp: str | None = None


class Interjections:
    def __init__(self, interrupt, threshold=1.5, debounce=1.0):
        if not all(math.isfinite(v) and v > 0 for v in (threshold, debounce)):
            raise ValueError("Voice threshold and debounce must be positive finite seconds")
        self.interrupt, self.threshold, self.debounce = interrupt, threshold, debounce
        self.lock = threading.RLock()
        self.items = []
        self.interrupted = False
        self.assistant_timestamp = datetime.now().astimezone().isoformat()

    def activity(self, speech_seconds):
        with self.lock:
            if speech_seconds >= self.threshold and not self.interrupted:
                self.interrupted = True
                self.interrupt()

    def transcript(self, text, offset, started_at, ended_at):
        if not text.strip(): return
        with self.lock:
            if self.items and 0 <= started_at - self.items[-1].ended_at <= self.debounce:
                item = self.items[-1]
                item.text += " " + text.strip()
                item.ended_at = ended_at
            else:
                timestamp = datetime.fromtimestamp(time.time() - time.monotonic() + started_at, timezone.utc).astimezone().isoformat()
                self.items.append(Interjection(text.strip(), max(0, offset), ended_at, timestamp))

    def messages(self, response):
        with self.lock:
            if self.assistant_timestamp is None:
                self.assistant_timestamp = datetime.now().astimezone().isoformat()
            result, cursor = [], 0
            assistant_timestamp = self.assistant_timestamp
            for item in self.items:
                offset = max(cursor, min(len(response), item.offset))
                if response[cursor:offset]:
                    result.append(ChatMessage("assistant", response[cursor:offset], timestamp=assistant_timestamp))
                result.append(ChatMessage("user", "[speaking over you] " + item.text, timestamp=item.timestamp))
                cursor = offset
                assistant_timestamp = item.timestamp
            if response[cursor:]:
                timestamp = self.items[-1].timestamp if self.items else self.assistant_timestamp
                result.append(ChatMessage("assistant", response[cursor:], timestamp=timestamp))
            return result
