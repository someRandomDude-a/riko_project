from __future__ import annotations

import re
from typing import Callable, Optional


class SentenceStreamer:
    """
    Buffer tokens until a terminator (". ! ? 。 ！ ？ \\n\\n") lands, then
    flush the complete sentence to `on_sentence(sentence)`.

    Force-flush rules (so the user isn't left waiting on a fragment):
      * buffer crosses `max_buffer_chars`
      * stream ends (`close()`)
    """

    def __init__(
        self,
        on_sentence: Callable[[str], None],
        terminators: Optional[list[str]] = None,
        max_buffer_chars: int = 240,
        min_chars_before_flush: int = 12,
    ) -> None:
        self.on_sentence = on_sentence
        self.max_buffer_chars = max_buffer_chars
        self.min_chars_before_flush = min_chars_before_flush

        # Bigger terminators must be matched before smaller prefixes
        terms = sorted(set(terminators or [". ", "! ", "? ", "。", "！", "？", "\n\n"]),
                       key=len, reverse=True)
        self.terminators = terms

        # Compile the regex once
        pattern = "(" + "|".join(re.escape(t) for t in self.terminators) + ")"
        self._split_re = re.compile(pattern)

        self._buf = ""
        self._closed = False

    def feed(self, token: str) -> None:
        """Append a token (or delta) and flush any complete sentences."""
        if not token:
            return
        self._buf += token

        # Hard cap — don't let the buffer grow unbounded
        if len(self._buf) >= self.max_buffer_chars:
            self._flush(force=True)
            return

        # Split-and-flush any newly-completed sentences
        while True:
            m = self._split_re.search(self._buf)
            if not m:
                break
            end = m.end()
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if len(sentence) >= self.min_chars_before_flush:
                self.on_sentence(sentence)
            # loop: there may be multiple terminators in the buffer

    def close(self) -> None:
        """Flush any remaining buffer (e.g., final fragment with no terminator)."""
        if self._closed:
            return
        self._closed = True
        if self._buf.strip():
            self.on_sentence(self._buf.strip())
        self._buf = ""

    def _flush(self, force: bool = False) -> None:
        if not self._buf.strip():
            return
        # On force flush (oversize buffer), split at the last whitespace near cap
        if force and len(self._buf) > self.max_buffer_chars:
            cut = self._buf.rfind(" ", 0, self.max_buffer_chars)
            if cut < self.min_chars_before_flush:
                cut = self.max_buffer_chars
            head, self._buf = self._buf[:cut], self._buf[cut:]
            if head.strip():
                self.on_sentence(head.strip())


def token_callback_for_streamer(
    streamer: SentenceStreamer,
) -> Callable[[str], None]:
    """Wrap a SentenceStreamer in a callable suitable for `on_token=...`."""
    def _cb(delta: str) -> None:
        streamer.feed(delta)
    return _cb
