"""Coalesce pending text so CPU interpretation never blocks token delivery."""
import logging
import threading
import re

logger = logging.getLogger(__name__)


class EmotionWorker:
    def __init__(self, engine):
        self.engine = engine
        self.condition = threading.Condition()
        self.pending = []
        self.closed = False
        self.revision = 0
        self.generated = ''
        self.playback = False
        self.thread = threading.Thread(target=self._run, name="emotion-stream", daemon=True)
        self.thread.start()

    def submit(self, kind, text="", final=False):
        with self.condition:
            if self.closed: return
            if kind == "start":
                # A new turn supersedes work that has not started yet.
                self.pending.clear()
                self.revision += 1
                self.generated = ''
                self.playback = False
                self.engine.playback_active = False
            if kind == 'speech':
                self.revision += 1
                self.playback = True
                self.engine.playback_active = True
                self.generated = ''
                self.pending = [job for job in self.pending if job[0] not in {'output','speech'}]
            if kind == 'output' and self.playback: return
            if self.pending and self.pending[-1][0] == kind and not self.pending[-1][2]:
                previous = self.pending.pop()
                text = previous[1] + text
            self.pending.append((kind, text, final))
            self.condition.notify()

    def generation(self, delta, final=False):
        with self.condition:
            if self.playback or self.closed: return
            self.generated += delta
            while True:
                boundary = re.search(r'[.!?。！？]+["”’\')\]]*(?=\s)|\n', self.generated)
                if not boundary: break
                sentence = self.generated[:boundary.end()].strip()
                self.generated = self.generated[boundary.end():]
                if sentence: self.submit_sentence(sentence)
            if final and self.generated.strip(): self.submit_sentence(self.generated.strip())
            if final: self.generated = ''

    def transcript(self, text, utterance_id):
        if not text.strip(): return
        with self.condition:
            if self.closed:return
            self.revision+=1
            self.pending=[job for job in self.pending if job[0]!='transcript']
            self.pending.append(('transcript',(utterance_id,text),True))
            self.condition.notify_all()

    def submit_sentence(self, text):
        # Bound backlog; do not replay old expressions after slow analysis.
        self.pending = [job for job in self.pending if job[0] != 'output']
        self.pending.append(('output', text, True))
        self.condition.notify()

    def _run(self):

        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending)
                if self.closed: return
                kind, text, final = self.pending.pop(0)
                revision = self.revision
            try:
                if kind == "start": self.engine.start_turn(text or None)
                elif kind=='transcript' and hasattr(self.engine,'observe_transcript'):
                    self.engine.observe_transcript(text[1],text[0],current=lambda:not self.closed and revision==self.revision)
                elif hasattr(self.engine, 'observe_segment'):
                    self.engine.observe_segment('user' if kind=='input' else 'speech' if kind=='speech' else 'assistant', text,
                        current=lambda: not self.closed and revision==self.revision)
                elif kind == "input": self.engine.observe_input(text, final=final)
                else: self.engine.observe_output(text, final=final)
            except Exception:
                logger.exception("Emotion stream analysis failed")

    def invalidate(self):
        with self.condition:
            self.revision += 1
            self.generated = ''
            self.pending = [job for job in self.pending if job[0] not in {'output','speech'}]


    def close(self):
        with self.condition:
            self.closed = True
            self.pending.clear()
            self.condition.notify_all()
