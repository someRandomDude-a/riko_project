from pathlib import Path
import struct

import pytest

from process.app_core.configuration.config import AppConfig, RuntimeConfig
from process.app_core.resources.gpu_memory import GPUMonitor, reconcile
from process.app_core.inference.llama_context import context_capacity, native_arguments
from process.app_core.resources.vram_estimate import estimate, read_gguf


def config(tmp_path, **options):
    return AppConfig(tmp_path, runtime=RuntimeConfig(provider='llama_cpp', model_path=Path('test.gguf'), **options),
        raw={'voice': {'asr_device': 'cpu'}})


def metadata():
    return {'layers': 32, 'embedding': 4096, 'heads': 32, 'kv_heads': 8,
        'weight_bytes': 3 * 1024 ** 3, 'context_length': 65536, 'source': 'fixture'}


def gpu(other=1000):
    return {'gpus': [{'index': 0, 'total_mib': 16000, 'used_mib': 4000, 'other_mib': other}]}


def test_pool_uses_worst_concurrent_task_budgets(tmp_path):
    c = config(tmp_path, parallel_slots=4, n_ctx=16384)
    c.runtime.initiative_n_ctx, c.runtime.reflection_n_ctx = 2048, 4096
    assert context_capacity(c.runtime) == 16384 + 3 * 4096
    c.runtime.initiative_n_ctx = 8192
    assert context_capacity(c.runtime) == 16384 + 8192 + 2 * 4096
    c.runtime.kv_unified = False
    assert context_capacity(c.runtime) == 16384 * 4
    assert '--no-kv-unified' in native_arguments(c.runtime, Path('test.gguf'))


def test_manual_pool_overrides_allocation_but_cannot_undersize(tmp_path):
    c = config(tmp_path, kv_pool_auto=False, kv_pool_tokens=32768)
    assert context_capacity(c.runtime) == 32768
    args = native_arguments(c.runtime, Path('test.gguf'))
    assert args[args.index('--ctx-size')+1] == '32768'
    c.runtime.kv_pool_tokens = 1024
    with pytest.raises(ValueError, match='configured concurrent'): context_capacity(c.runtime)
    c.runtime.kv_pool_auto = True
    assert context_capacity(c.runtime) == 12288


def test_precision_and_task_context_change_estimates_without_changing_settings(tmp_path):
    c = config(tmp_path)
    before = estimate(c, gpu(), metadata())
    assert before['confidence'] == 'low'
    old = c.runtime.n_ctx
    c.runtime.type_k = c.runtime.type_v = 'q4_0'
    quantized = estimate(c, gpu(), metadata())
    get = lambda result, key: next(row for row in result['components'] if row['id'] == key)
    assert get(quantized, 'kv')['high_mib'] < get(before, 'kv')['high_mib']
    c.runtime.reflection_n_ctx = 16384
    bigger = estimate(c, gpu(), metadata())
    assert get(bigger, 'kv')['high_mib'] > get(quantized, 'kv')['high_mib']
    assert c.runtime.n_ctx == old
    assert bigger['suggestion']['live_context_tokens'] != old


def test_cpu_offload_and_external_provider_exclusions(tmp_path):
    c = config(tmp_path, n_gpu_layers=0)
    result = estimate(c, gpu(), metadata())
    assert next(row for row in result['components'] if row['id'] == 'llm_weights')['high_mib'] == 0
    assert next(row for row in result['components'] if row['id'] == 'asr')['high_mib'] == 0
    c.runtime.provider = 'openai'
    result = estimate(c, gpu(), metadata())
    assert not result['managed']
    assert result['suggestion'] is None
    assert next(row for row in result['components'] if row['id'] == 'tts')['high_mib'] == 0


def test_asr_model_precision_and_batches_affect_estimate(tmp_path):
    c = config(tmp_path)
    c.raw['voice'] = {'asr_device': 'cuda', 'asr_model': 'small', 'asr_compute_type': 'int8_float16'}
    compact = estimate(c, gpu(), metadata())
    c.raw['voice']['asr_model'] = 'large-v3'
    c.raw['voice']['asr_compute_type'] = 'float32'
    c.runtime.n_batch = c.runtime.n_ubatch = 2048
    large = estimate(c, gpu(), metadata())
    assert large['high_mib'] > compact['high_mib']


def test_unknown_metadata_suppresses_false_total_and_suggestion(tmp_path):
    result = estimate(config(tmp_path), gpu(), {'source': 'Unavailable'})
    assert not result['complete']
    assert result['unknown_components']
    assert result['suggestion'] is None


def test_hybrid_attention_metadata_does_not_charge_every_layer_dense_kv(tmp_path):
    c = config(tmp_path)
    dense = estimate(c, gpu(), metadata())
    hybrid = estimate(c, gpu(), {**metadata(), 'full_attention_interval': 4})
    kv = lambda result: next(row for row in result['components'] if row['id'] == 'kv')['low_mib']
    assert kv(hybrid) == kv(dense) / 4
    assert any(row['id'] == 'recurrent' for row in hybrid['components'])


def test_reconcile_never_invents_other_from_missing_process_memory():
    gpus = [{'uuid': 'gpu', 'used_mib': 4000}]
    processes = [{'gpu_uuid': 'gpu', 'pid': 10, 'used_mib': None}]
    result = reconcile(gpus, processes, {10: 'Python'}, complete=True)[0]
    assert result['owned_mib'] is None and result['other_mib'] is None
    processes[0]['used_mib'] = 1000
    result = reconcile(gpus, processes, {10: 'Python'}, complete=True)[0]
    assert result['owned_mib'] == 1000 and result['other_mib'] == 3000
    assert reconcile(gpus, processes, {10: 'Python'}, complete=False)[0]['other_mib'] is None
    processes[0]['used_mib'] = 30000
    assert reconcile(gpus, processes, {10: 'Python'}, complete=True)[0]['other_mib'] is None


def test_unattributed_usage_produces_explicitly_conservative_suggestion(tmp_path):
    result = estimate(config(tmp_path), gpu(None), metadata())
    assert 'double-count' in result['suggestion']['basis']


def test_monitor_missing_driver_fails_cleanly_and_cache_bounds_polling(monkeypatch):
    monitor = GPUMonitor()
    calls = []
    monkeypatch.setattr('process.app_core.resources.gpu_memory.shutil.which', lambda name: calls.append(name) or None)
    assert not monitor.sample()['available']
    assert not monitor.sample()['available']
    assert len(calls) == 1
    with pytest.raises(ValueError): monitor.register_electron([{'pid': -1, 'kind': 'GPU'}])


def test_gpu_sampling_is_shared_and_stops_with_last_subscriber(monkeypatch):
    import threading
    from process.app_core.events.bus import event_bus
    monitor = GPUMonitor()
    sampled = threading.Event()
    calls = []
    monkeypatch.setattr(monitor, 'sample', lambda provider: calls.append(provider) or {'available': False, 'gpus': []})
    listener = event_bus.subscribe(lambda event: sampled.set() if event.type == 'resource.gpu' else None)
    first = monitor.subscribe(lambda: 'provider')
    second = monitor.subscribe(lambda: 'provider')
    try:
        assert sampled.wait(1)
        assert len(calls) == 1
        stop = monitor.watch_stop
        first()
        assert not stop.is_set()
        second()
        assert stop.is_set()
        assert monitor.watchers == 0
    finally: listener()


def test_gguf_reader_skips_tokenizer_and_reads_architecture(tmp_path):
    def text(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data
    entries = [text('general.architecture') + struct.pack('<I', 8) + text('llama'),
        text('llama.block_count') + struct.pack('<II', 4, 32),
        text('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 2) + text('a') + text('b')]
    path = tmp_path / 'model.gguf'
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, len(entries)) + b''.join(entries))
    raw = read_gguf(path)
    assert raw == {'general.architecture': 'llama', 'llama.block_count': 32}
    path.write_bytes(b'broken')
    with pytest.raises(ValueError): read_gguf(path)
