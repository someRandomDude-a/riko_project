"""Remove echoed application time metadata before display, speech and history.

Only full date+time stamps and explicit section markers are removed. Ordinary
dates, clock times, Markdown, numbers and tool arguments are not rewritten.
Candidate prefixes are held across stream chunks so TTS never receives half a
timestamp. The held metadata suffix is bounded, not a whole-reply buffer.
"""
import re

from .streaming import WordDeltas

_ISO = r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+-]\d{2}:?\d{2})?'
_STAMP = re.compile(_ISO, re.I)
_MARKER = re.compile(r'(?:' + _ISO + r'|timestamp unavailable|YYYY-MM-DDTHH:MM(?::SS)?)', re.I)
_BASE = 'dddd-dd-ddTdd:dd'


def _iso_prefix(value):
    for index, char in enumerate(value[:len(_BASE)]):
        expected = _BASE[index]
        if expected == 'd':
            if char not in '0123456789': return False
        elif expected == 'T':
            if char not in 'Tt ': return False
        elif char != expected: return False
    if len(value) <= len(_BASE): return True
    return bool(re.fullmatch(r'(?::\d{0,2}(?:\.\d{0,9})?)?(?:[Zz]|[+-]\d{0,2}(?::?\d{0,2})?)?', value[len(_BASE):]))


def _marker_prefix(value):
    return (_iso_prefix(value) or 'timestamp unavailable'.startswith(value.lower())
            or 'yyyy-mm-ddthh:mm:ss'.startswith(value.lower()))


class OutputFilter:
    def __init__(self, emit):
        self.pending = ''
        self.words = WordDeltas(emit)

    def feed(self, text):
        self.pending += text
        self._drain(final=False)

    def finish(self):
        self._drain(final=True)
        self.words.finish()

    def _drain(self, final):
        output = []
        while self.pending:
            text = self.pending
            if text.startswith('['):
                end = text.find(']')
                if end >= 0 and _MARKER.fullmatch(text[1:end]):
                    self.pending = text[end + 1:]
                    continue
                if end < 0 and len(text) <= 80 and _marker_prefix(text[1:]):
                    if not final: break
                    # A token-limit cutoff may leave an unmistakable metadata
                    # stamp unfinished. Short ambiguous text such as '[2026'
                    # is retained rather than mistaken for a complete stamp.
                    if len(text) >= 12:
                        self.pending = ''
                        continue
            if text[0] in '0123456789':
                if len(text) <= 80 and _iso_prefix(text):
                    if not final: break
                    if len(text) >= 16:
                        self.pending = ''
                        continue
                match = _STAMP.match(text)
                if match:
                    self.pending = text[match.end():]
                    continue
            output.append(text[0])
            self.pending = text[1:]
        if output: self.words.feed(''.join(output))


def clean_output(text):
    chunks = []
    stream = OutputFilter(chunks.append)
    stream.feed(text)
    stream.finish()
    return ''.join(chunks)
