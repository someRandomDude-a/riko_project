from process.app_core.inference.metrics import InferenceMetrics
from process.app_core.configuration.debug_logging import SafeFormatter
import logging


def test_live_metrics_throttle_and_exact_final_usage():
    now = [0.0]
    updates = []
    metrics = InferenceMetrics(updates.append, lambda: now[0])
    now[0] = 1
    metrics.delta('abcd')
    now[0] = 1.1
    metrics.delta('abcd')
    assert len(updates) == 1
    now[0] = 2
    value = metrics.finish({'output_tokens':10})
    assert value['tokens_per_second'] == 10
    assert not value['estimated']
    assert value['first_token_seconds'] == 1


def test_log_redacts_known_secrets_and_auth_fields():
    record = logging.LogRecord('test',logging.ERROR,'',1,'secret-value Bearer abc token=xyz',(),None)
    output = SafeFormatter(['secret-value']).format(record)
    assert 'secret-value' not in output and 'abc' not in output and 'xyz' not in output


def test_native_rate_uses_slot_timings_not_character_estimates():
    now = [0.0]
    metrics = InferenceMetrics(clock=lambda:now[0])
    now[0]=1
    metrics.native({'predicted_n':32,'predicted_per_second':45.6})
    metrics.delta('not a tokenizer count')
    value=metrics.finish({'output_tokens':33})
    assert value['tokens_per_second']==45.6
    assert value['output_tokens']==32
    assert value['timing_source']=='llama.cpp'
    assert not value['estimated']
