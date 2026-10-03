"""Bounded, abstaining choice repair before approvals; never repair file permissions."""
from difflib import SequenceMatcher
import math
import re
import threading

from ..runtime.workers import DaemonExecutor


def normalized(value): return re.sub(r'[\s_-]+', ' ', value.strip().casefold())


class ChoiceResolver:
    def __init__(self, selector=None, *, enabled=True, timeout=.4, confidence=.85):
        self.selector, self.enabled, self.timeout, self.confidence = selector, enabled, timeout, confidence
        self.worker = DaemonExecutor(max_workers=1, max_pending=1, thread_name_prefix='tool-choice')
        self.lock = threading.Lock()
        self.pending = None

    def choose(self, tool, parameter, value, options):
        if value in options: return value
        if not self.enabled or not isinstance(value, str) or not value.strip() or len(value) > 200: return None
        # URI/path/traversal tokens are not cosmetic typos or choice synonyms.
        if any(token in value for token in ('/', '\\', ':', '..')): return None
        exact = [option for option in options if normalized(option) == normalized(value)]
        if len(exact) == 1: return exact[0]
        if self.selector:
            with self.lock:
                if self.pending is None or self.pending.done():
                    try: self.pending = self.worker.submit(self.selector, tool, parameter, value,
                        [{'id':f'choice-{i}', 'description':option} for i, option in enumerate(options[:64])])
                    except RuntimeError: return None
                    future = self.pending
                else: future = None
            if future:
                try:
                    result = future.result(timeout=self.timeout)
                    choices = {f'choice-{i}':option for i, option in enumerate(options[:64])}
                    confidence = float(result.get('confidence', 0)) if isinstance(result, dict) else 0
                    if math.isfinite(confidence) and confidence >= self.confidence and result.get('choice') in choices:
                        return choices[result['choice']]
                except Exception: pass # Unavailable/native Julia cannot break the tool pipeline.
        # Only unique close spellings; unrelated semantic inputs fail closed.
        ranked = sorted(((SequenceMatcher(None, normalized(value), normalized(option)).ratio(), option) for option in options), reverse=True)
        if ranked and ranked[0][0] >= .82 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= .1): return ranked[0][1]
        return None

    def normalize(self, tool, arguments, choices):
        corrections = []
        result = dict(arguments)
        for parameter, options in choices.items():
            if parameter not in result or not options: continue
            original = result[parameter]
            if original == '' and parameter in {'name','asset_id'}: continue # Optional fields may be filled with empty defaults.
            if original in options: continue
            selected = self.choose(tool, parameter, original, list(options))
            if selected is None: raise ValueError(f'Invalid {parameter} for {tool}. Available choices: ' + ', '.join(options[:64]))
            result[parameter] = selected
            corrections.append({'parameter':parameter, 'from':original, 'to':selected})
        return result, corrections

    def close(self): self.worker.shutdown(wait=False, cancel_futures=True)
