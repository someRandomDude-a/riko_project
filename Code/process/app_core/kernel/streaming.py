"""Provider-independent text-stream utilities; no SDK or model imports."""
import re


class WordDeltas:
    """Preserve text exactly, delivering only whitespace-ended chunks until EOF."""
    def __init__(self, emit):
        self.emit, self.pending = emit, ""

    def feed(self, text):
        self.pending += text
        end = 0
        for boundary in re.finditer(r'\s+', self.pending):
            end = boundary.end()
        if end:
            chunk, self.pending = self.pending[:end], self.pending[end:]
            self.emit(chunk)

    def finish(self):
        if self.pending:
            tail, self.pending = self.pending, ""
            self.emit(tail)
