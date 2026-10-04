"""Low-cost stream timing; never tokenizes or stores conversation text."""
import logging
import time

logger = logging.getLogger(__name__)


class InferenceMetrics:
    def __init__(self, publish=lambda value: None, clock=time.perf_counter):
        self.publish, self.clock = publish, clock
        self.started = clock()
        self.first = None
        self.characters = 0
        self.last = self.started
        self.native_timings = None

    def native(self, timings):
        if not isinstance(timings, dict) or type(timings.get('predicted_n')) not in (int,float): return
        self.native_timings = dict(timings)
        now = self.clock()
        if self.first is None and timings['predicted_n'] > 0: self.first = now
        if now - self.last >= .25:
            self.last = now
            self.publish(self.snapshot(now))

    def delta(self, text):
        if not text: return
        now = self.clock()
        if self.first is None: self.first = now
        self.characters += len(text)
        if now - self.last >= .25:
            self.last = now
            self.publish(self.snapshot(now))

    def snapshot(self, now=None, usage=None, final=False):
        now = self.clock() if now is None else now
        usage = usage or {}
        native = self.native_timings or {}
        tokens = native.get('predicted_n', usage.get('output_tokens', usage.get('completion_tokens')))
        exact = type(tokens) in (int,float) and tokens >= 0
        tokens = tokens if exact else self.characters / 4
        duration = max(0, now - self.first) if self.first is not None else 0
        return {'output_tokens': round(tokens, 1), 'estimated': not exact,
            'tokens_per_second': native.get('predicted_per_second', round(tokens / duration, 2) if duration >= .1 else None),
            'timing_source': 'llama.cpp' if native else 'provider_usage' if exact else 'text_estimate',
            'first_token_seconds': round(self.first - self.started, 3) if self.first is not None else None,
            'provider_seconds': round(now - self.started, 3), 'final': final}

    def finish(self, usage=None):
        value = self.snapshot(usage=usage, final=True)
        self.publish(value)
        logger.info('Inference finished duration_s=%s first_token_s=%s output_tokens=%s tokens_per_s=%s estimated=%s',
            value['provider_seconds'], value['first_token_seconds'], value['output_tokens'], value['tokens_per_second'], value['estimated'])
        return value
