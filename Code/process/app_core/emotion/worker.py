"""Coalesce pending text so CPU interpretation never blocks token delivery."""
import logging
import threading

logger = logging.getLogger(__name__)


class EmotionWorker:
    def __init__(self, engine):
        self.engine = engine
        self.condition = threading.Condition()
        self.pending = []
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="emotion-stream", daemon=True)
        self.thread.start()

    def submit(self, kind, text="", final=False):
        with self.condition:
            if self.closed: return
            if kind == "start":
                # A new turn supersedes work that has not started yet.
                self.pending.clear()
            if self.pending and self.pending[-1][0] == kind and not self.pending[-1][2]:
                previous = self.pending.pop()
                text = previous[1] + text
            self.pending.append((kind, text, final))
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending)
                if self.closed: return
                kind, text, final = self.pending.pop(0)
            try:
                if kind == "start": self.engine.start_turn(text or None)
                elif kind == "input": self.engine.observe_input(text, final=final)
                else: self.engine.observe_output(text, final=final)
            except Exception:
                logger.exception("Emotion stream analysis failed")

    def close(self):
        with self.condition:
            self.closed = True
            self.pending.clear()
            self.condition.notify_all()
