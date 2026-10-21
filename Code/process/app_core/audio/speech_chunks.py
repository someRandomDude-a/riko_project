"""Application-owned, punctuation-prioritized soft word limits for speech."""
import re

from ..kernel.audio_config import SpeechConfig

WORDS = re.compile(r"\b\w+(?:['’]\w+)*\b")


class SpeechChunks:
    def __init__(self, settings=SpeechConfig()):  # the speech section, which load_config has checked (SpeechConfig.from_raw)
        self.settings = settings
        self.pending = ''

    def feed(self, delta, final=False):
        self.pending += delta
        result = []
        while True:
            words = list(WORDS.finditer(self.pending))
            limit = self.settings.max_words
            if len(words) <= limit:
                break
            lower = max(1, limit - self.settings.split_window_words)
            # Prefer punctuation within the look-back window ending at the
            # word threshold. If absent, wait for the first future boundary.
            candidates = []
            count = 0
            for position, character in enumerate(self.pending):
                while count < len(words) and words[count].end() <= position:
                    count += 1
                if count < lower: continue
                for rank, group in enumerate(self.settings.split_priority):
                    if character in group:
                        # Ignore decimal points and intra-word apostrophe-like
                        # punctuation: a boundary needs whitespace/end/quotes.
                        end = position + 1
                        while end < len(self.pending) and self.pending[end] in '.!?。！？"”’\')]}':
                            end += 1
                        if end < len(self.pending) and not self.pending[end].isspace():
                            break
                        candidates.append((count, rank, end))
                        break
            nearby = [c for c in candidates if c[0] <= limit]
            if nearby:
                boundary = min(nearby, key=lambda c: (c[1], limit-c[0], -c[2]))[2]
            elif candidates:
                boundary = min(candidates, key=lambda c: c[2])[2]
            else:
                break
            text = self.pending[:boundary].strip()
            self.pending = self.pending[boundary:].lstrip()
            if text: result.append(text)
        if final:
            tail = self.pending.strip()
            if tail: result.append(tail)
            self.pending = ''
        return result
