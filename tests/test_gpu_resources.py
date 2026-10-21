from pathlib import Path
import struct

import pytest

from process.app_core.configuration.config import AppConfig, RuntimeConfig
from process.app_core.resources.gpu_memory import GPUMonitor, merge, reconcile
from process.app_core.resources import gpu_sources
from process.app_core.inference.llama_context import context_capacity, native_arguments
from process.app_core.inference.gguf import read_gguf
from process.app_core.resources.vram_estimate import estimate


def config(tmp_path, **options):
    return AppConfig(tmp_path, runtime=RuntimeConfig(provider='llama_cpp', model_path=Path('test.gguf'), **options),
        raw={'voice': {'asr_device': 'cpu'}})


def metadata():
    return {'layers': 32, 'embedding': 4096, 'heads': 32, 'kv_heads': 8,
        'weight_bytes': 3 * 1024 ** 3, 'context_length': 65536, 'source': 'fixture'}


def gpu(other=1000):
    return {'gpus': [{'key': 'pci:0000:01:00.0', 'index': 0, 'pci_bus_id': '0000:01:00.0', 'llm': True, 'memory_kind': 'dedicated',
        'memory_scope': 'device', 'total_mib': 16000, 'used_mib': 4000, 'other_mib': other}]}


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


def ctranslate2(monkeypatch, cuda):
    from process.app_core.audio import asr
    types = {'cpu': frozenset({'float32', 'int8', 'int8_float32'})}
    if cuda: types['cuda'] = frozenset({'float32', 'float16', 'int8', 'int8_float32', 'int8_float16'})
    monkeypatch.setattr(asr, 'supported_types', lambda: types)


def test_asr_model_precision_and_batches_affect_estimate(tmp_path, monkeypatch):
    ctranslate2(monkeypatch, cuda=True)
    c = config(tmp_path)
    c.raw['voice'] = {'asr_device': 'cuda', 'asr_model': 'small', 'asr_compute_type': 'int8_float16'}
    compact = estimate(c, gpu(), metadata())
    c.raw['voice']['asr_model'] = 'large-v3'
    c.raw['voice']['asr_compute_type'] = 'float32'
    c.runtime.n_batch = c.runtime.n_ubatch = 2048
    large = estimate(c, gpu(), metadata())
    asr = lambda result: next(row for row in result['components'] if row['id'] == 'asr')['high_mib']
    assert large['high_mib'] > compact['high_mib'] and asr(large) > asr(compact) > 0


def test_asr_estimate_uses_the_device_it_will_actually_run_on(tmp_path, monkeypatch):
    ctranslate2(monkeypatch, cuda=False)
    c = config(tmp_path)
    c.raw['voice'] = {'asr_device': 'cuda', 'asr_model': 'small', 'asr_compute_type': 'int8_float16'}
    result = estimate(c, gpu(), metadata())
    row = lambda key: next(row for row in result['components'] if row['id'] == key)
    assert row('asr')['high_mib'] == 0 and row('gpu_runtime')['high_mib'] == 0 and row('asr')['basis'] == 'CPU (int8)'
    assert any('runs as cpu/int8' in warning for warning in result['warnings'])


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
    gpus = [{'key': 'gpu', 'used_mib': 4000}]
    processes = [{'gpu': 'gpu', 'pid': 10, 'used_mib': None}]
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
    calls = []
    monitor = GPUMonitor(sources=[gpu_sources.NvidiaSmi(which=lambda name: calls.append(name) or None)])
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


# Backend-neutral telemetry (resources/gpu_sources.py, gpu_memory.merge) and the estimate's device mapping.
def ggml(index, name, backend, kind, device_id=None, total=24564, free=20564, description='Device', model=False, main=False):
    return {'index': index, 'name': name, 'description': description, 'backend': backend, 'type': kind, 'device_id': device_id,
        'memory_free': free * 1024 ** 2, 'memory_total': total * 1024 ** 2, 'model': model, 'main': main}


class Provider:  # what InProcessLlamaProvider.devices() answers; GPUMonitor calls it directly
    def __init__(self, devices): self.listed, self.calls = devices, 0
    def devices(self):
        self.calls += 1
        return self.listed


def smi(gpus, apps=''):
    def run(command):
        if command[1].startswith('--query-gpu='):
            assert command[1].endswith(',pci.bus_id')
            return gpus
        return apps
    return gpu_sources.NvidiaSmi(which=lambda name: '/usr/bin/nvidia-smi', run=run)


SMI = '0, GPU-a, NVIDIA GeForce RTX 3090, 24576, 2000, 22576, 00000000:01:00.0\n1, GPU-b, NVIDIA GeForce RTX 4090, 24564, 5000, 19564, 00000000:02:00.0\n'
# CUDA lists the fastest GPU first: CUDA0 is nvidia-smi's GPU 1 (PCI 02), CUDA1 its GPU 0 (PCI 01).
CUDA = [ggml(0, 'CUDA0', 'CUDA', 'gpu', '0000:02:00.0', description='NVIDIA GeForce RTX 4090', model=True, main=True),
        ggml(1, 'CUDA1', 'CUDA', 'gpu', '0000:01:00.0', description='NVIDIA GeForce RTX 3090'), ggml(2, 'CPU', 'CPU', 'cpu', total=65536, free=65536)]


def test_pci_addresses_normalise_across_nvidia_smi_ggml_sysfs_and_d3dkmt():
    assert gpu_sources.pci_address('00000000:01:00.0') == gpu_sources.pci_address('0000:01:00.0') == '0000:01:00.0'
    assert gpu_sources.pci_address('0000:0A:1F.7') == '0000:0a:1f.7' and gpu_sources.pci_address('03:00.0') == '0000:03:00.0'
    assert gpu_sources.pci_address('GPU-1234') is None and gpu_sources.pci_address(None) is None
    assert gpu_sources.adapter_pci(1, 0, 0) == '0000:01:00.0' and gpu_sources.adapter_pci(0x1c3, 0x21, 9) == '0000:c3:01.1'


def test_merge_joins_nvidia_smi_and_the_bridge_by_pci_and_names_the_model_gpu():
    payload = GPUMonitor(sources=[smi(SMI, 'GPU-b, 4242, 4000\n')]).sample(Provider(CUDA))
    assert payload['available'] and payload['errors'] == {} and [gpu['key'] for gpu in payload['gpus']] == ['pci:0000:01:00.0', 'pci:0000:02:00.0']
    first, second = payload['gpus']
    assert (first['index'], first['ggml_device'], first['ggml_index'], first['llm'], first['uuid']) == (0, 'CUDA1', 1, False, 'GPU-a')
    assert (second['index'], second['ggml_device'], second['ggml_index'], second['llm'], second['main']) == (1, 'CUDA0', 0, True, True)
    assert second['used_mib'] == 5000 and second['backend'] == 'CUDA' and second['memory_kind'] == 'dedicated' and second['memory_scope'] == 'device'
    assert second['sources'] == ['nvidia-smi (compute processes only)', 'llama.cpp devices (riko_devices)']
    assert payload['main_key'] == 'pci:0000:02:00.0' and payload['ggml_devices'][2]['name'] == 'CPU'
    assert second['owned_mib'] is None and second['attribution_complete'] is False  # nvidia-smi lists compute processes only


def test_metal_row_is_unified_and_counts_only_this_process():
    import os
    metal = [ggml(0, 'MTL0', 'MTL', 'gpu', total=38339, free=37791, description='Apple M5 Pro', model=True, main=True),
             ggml(1, 'BLAS', 'BLAS', 'accel', total=0, free=0), ggml(2, 'CPU', 'CPU', 'cpu', total=49152, free=49152)]
    monitor = GPUMonitor(sources=[])
    monitor.electron, monitor.electron_identity = {999999: 'Electron GPU'}, {999999: None}
    payload = monitor.sample(Provider(metal))
    [row] = payload['gpus']
    assert row['key'] == 'ggml:MTL0' and row['name'] == 'Apple M5 Pro' and (row['memory_kind'], row['memory_scope']) == ('unified', 'process')
    assert row['total_mib'] == 38339 and row['used_mib'] == 548 and row['owned_mib'] == 548 and row['other_mib'] is None
    assert row['owned_processes'] == [{'pid': os.getpid(), 'used_mib': 548, 'category': 'Python (LLM, ASR and runtime)'}]
    assert payload['main_key'] == 'ggml:MTL0' and payload['source'] == 'llama.cpp devices (riko_devices)'


def test_no_source_at_all_says_why_and_a_failing_source_is_shown_without_hiding_the_others():
    payload = GPUMonitor(sources=[smi('')]).sample(Provider(None))  # an older bridge: devices() is None
    assert not payload['available'] and 'riko_devices' in payload['error']
    def broken(command): raise gpu_sources.subprocess.CalledProcessError(9, command, stderr='driver mismatch')
    failing = gpu_sources.NvidiaSmi(which=lambda name: 'nvidia-smi', run=broken)
    payload = GPUMonitor(sources=[failing]).sample(Provider(CUDA))
    assert payload['available'] and 'nvidia-smi' in payload['errors'] and [gpu['ggml_device'] for gpu in payload['gpus']] == ['CUDA0', 'CUDA1']
    vulkan = [ggml(0, 'Vulkan0', 'Vulkan', 'gpu', '0000:03:00.0', total=16368, free=16368)]  # no memory-budget extension: free == total
    [row] = GPUMonitor(sources=[]).sample(Provider(vulkan))['gpus']
    assert row['total_mib'] == 16368 and row['used_mib'] is None and row['free_mib'] is None


def test_windows_counters_attribute_every_process_per_adapter():
    adapter_values = [('luid_0x00000000_0x0000D1B4_phys_0', 6 * 1024 ** 3), ('luid_0x00000000_0x0000E000_phys_0', 64 * 1024 ** 2)]
    process_values = [('pid_4242_luid_0x00000000_0x0000D1B4_phys_0', 4 * 1024 ** 3), ('pid_4242_luid_0x00000000_0x0000D1B4_phys_1', 1024 ** 3),
        ('pid_77_luid_0x00000000_0x0000D1B4_phys_0', 512 * 1024 ** 2), ('not an instance', 1), ('pid_5_luid_0x0_0x1_phys_0', float('nan'))]
    adapter, process = gpu_sources.counter_totals(adapter_values, process_values)
    assert adapter == {'00000000:0000d1b4': 6 * 1024 ** 3, '00000000:0000e000': 64 * 1024 ** 2}
    assert process == {(4242, '00000000:0000d1b4'): 5 * 1024 ** 3, (77, '00000000:0000d1b4'): 512 * 1024 ** 2}
    source = gpu_sources.WindowsCounters(adapters=lambda: [
        {'name': 'AMD Radeon RX 7900 XTX', 'vendor': 0x1002, 'dedicated': 24 * 1024 ** 3, 'luid': '00000000:0000d1b4', 'pci_bus_id': '0000:03:00.0'}])
    source.sample = lambda: (adapter, process)
    payload = merge([source.read({4242: 'Python'})], {4242: 'Python'})
    [row] = payload['gpus']
    assert row['key'] == 'pci:0000:03:00.0' and row['total_mib'] == 24576 and row['used_mib'] == 6144 and row['attribution_complete']
    assert row['owned_mib'] == 5120 and row['other_mib'] == 1024 and row['owned_processes'][0]['pid'] == 4242
    source.sample = lambda: (adapter, None)  # the process counter failed this sample: usage, never "all of it is Other"
    [row] = merge([source.read({4242: 'Python'})], {4242: 'Python'})['gpus']
    assert row['used_mib'] == 6144 and not row['attribution_complete'] and row['owned_mib'] is None and row['other_mib'] is None


def test_amdgpu_sysfs_and_drm_fdinfo_read_vram_and_owned_process_memory(tmp_path):
    card = tmp_path / 'sys/class/drm/card1/device'
    card.mkdir(parents=True)
    (tmp_path / 'sys/class/drm/card1-DP-1').mkdir()  # a connector, not a card
    for name, value in {'vendor': '0x1002', 'mem_info_vram_total': str(16 * 1024 ** 3), 'mem_info_vram_used': str(3 * 1024 ** 3),
            'product_name': 'AMD Radeon RX 7800 XT', 'uevent': 'DRIVER=amdgpu\nPCI_SLOT_NAME=0000:03:00.0\n'}.items():
        (card / name).write_text(value)
    nvidia = tmp_path / 'sys/class/drm/card0/device'
    nvidia.mkdir(parents=True)
    (nvidia / 'vendor').write_text('0x10de')
    reading = gpu_sources.AmdgpuSysfs(tmp_path / 'sys').read({})
    assert [(d['id'], d['name'], d['pci_bus_id'], d['total_mib'], d['used_mib']) for d in reading.devices] == [('card1', 'AMD Radeon RX 7800 XT', '0000:03:00.0', 16384, 3072)]
    fdinfo = tmp_path / 'proc/4242/fdinfo'
    fdinfo.mkdir(parents=True)
    client = 'drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t{}\ndrm-memory-vram:\t{} KiB\n'
    (fdinfo / '5').write_text(client.format(7, 1024 * 1024))
    (fdinfo / '6').write_text(client.format(7, 1024 * 1024))  # a dup of the same DRM client: counted once
    (fdinfo / '7').write_text(client.format(8, 512 * 1024))
    (fdinfo / '1').write_text('pos:\t0\nflags:\t02\n')  # not a DRM file
    owned = {4242: 'Python'}
    drm = gpu_sources.DrmFdinfo(tmp_path / 'proc').read(owned)
    assert drm.processes == [{'device': '0000:03:00.0', 'pid': 4242, 'used_mib': 1536}] and drm.complete == {'0000:03:00.0'}
    [row] = merge([reading, drm], owned)['gpus']
    assert row['owned_mib'] == 1536 and row['other_mib'] == 1536 and row['attribution_complete']
    drm = gpu_sources.DrmFdinfo(tmp_path / 'proc').read({**owned, 77: 'Electron GPU'})  # a sandboxed Chromium helper we cannot read
    assert drm.complete == set()
    assert merge([reading, drm], {**owned, 77: 'Electron GPU'})['gpus'][0]['other_mib'] is None


def test_platform_sources_and_process_identity_per_platform(monkeypatch):
    import os
    import sys
    names = lambda platform: [source.name for source in gpu_sources.platform_sources(platform)]
    assert names('darwin') == ['nvidia-smi'] and names('linux') == ['nvidia-smi', 'amdgpu', 'drm-fdinfo'] and names('win32') == ['nvidia-smi', 'windows']
    assert gpu_sources.process_identity(os.getpid(), platform='sunos5') is None
    if sys.platform != 'darwin': assert gpu_sources.process_identity(os.getpid(), platform='darwin') is None  # no libproc here
    identity = gpu_sources.process_identity(os.getpid())
    assert identity is not None and identity == gpu_sources.process_identity(os.getpid())  # this process, on every CI OS
    assert gpu_sources.process_identity(2 ** 22 + 12345) is None  # no such process


@pytest.mark.skipif(__import__('sys').platform != 'win32', reason='DXGI and PDH exist only on Windows')
def test_windows_counters_run_for_real_without_raising():
    import ctypes
    import os
    reading = gpu_sources.WindowsCounters().read({os.getpid(): 'Python'})  # a runner without a GPU: no adapters or no counters
    assert reading is None or all(device['luid'] for device in reading.devices)
    # A CI VM lists only software adapters, so read() stops before PDH: run PDH itself, twice (the kept query), for real.
    fake = {'name': 'Fake', 'vendor': 0x10de, 'dedicated': 1024 ** 3, 'luid': '00000000:00000001', 'pci_bus_id': None}
    source = gpu_sources.WindowsCounters(adapters=lambda: [fake])
    for _ in range(2):
        values = source.sample()
        assert values is None or (isinstance(values[0], dict) and (values[1] is None or isinstance(values[1], dict)))
    assert source.read({os.getpid(): 'Python'}).devices[0]['luid'] == fake['luid']
    class LUID(ctypes.Structure): _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_int32)]
    assert gpu_sources.adapter_address(LUID(0, 0)) is None  # D3DKMT refuses a LUID no adapter has


def telemetry_for(devices, sources=()):
    return GPUMonitor(sources=list(sources)).sample(Provider(devices))


def test_estimate_compares_against_the_device_llama_cpp_picks_not_the_driver_numbering(tmp_path):
    c = config(tmp_path, split_mode='none', main_gpu=1)
    telemetry = telemetry_for(CUDA, [smi(SMI)])
    result = estimate(c, telemetry, metadata())
    assert result['gpu_key'] == 'pci:0000:01:00.0' and result['suggestion'] is not None  # CUDA1 is nvidia-smi's GPU 0
    assert not any('assumes' in warning for warning in result['warnings'])
    assert 'includes CUDA graphs' in next(row for row in result['components'] if row['id'] == 'llm_compute')['basis']
    c.runtime.main_gpu = 2
    result = estimate(c, telemetry, metadata())
    assert result['projection'] is None and result['suggestion'] is None and any('refuse to load' in warning for warning in result['warnings'])


def test_estimate_sums_a_model_spread_over_gpus_and_suppresses_the_suggestion(tmp_path):
    result = estimate(config(tmp_path, split_mode='layer'), telemetry_for(CUDA, [smi(SMI)]), metadata())
    assert result['suggestion'] is None and any('spans 2 GPUs' in warning for warning in result['warnings'])
    assert result['gpu_key'] == 'pci:0000:02:00.0+pci:0000:01:00.0'
    assert result['projection'] is None  # nvidia-smi cannot attribute: no Other to add


def test_estimate_without_bridge_devices_says_main_gpu_is_a_guess(tmp_path):
    telemetry = GPUMonitor(sources=[smi(SMI)]).sample(None)
    result = estimate(config(tmp_path, main_gpu=1), telemetry, metadata())
    assert any('without riko_devices' in warning for warning in result['warnings']) and result['suggestion'] is not None
    one = estimate(config(tmp_path), gpu(), metadata())  # one GPU: nothing to guess
    assert not any('without riko_devices' in warning for warning in one['warnings']) and one['projection']['gpu_key'] == 'pci:0000:01:00.0'


def test_unified_memory_projection_uses_the_working_set_and_leaves_electron_out(tmp_path):
    metal = [ggml(0, 'MTL0', 'MTL', 'gpu', total=38339, free=37791, description='Apple M5 Pro', model=True, main=True),
             ggml(1, 'BLAS', 'BLAS', 'accel', total=0, free=0), ggml(2, 'CPU', 'CPU', 'cpu', total=49152, free=49152)]
    result = estimate(config(tmp_path), telemetry_for(metal), metadata())
    get = lambda key: next(row for row in result['components'] if row['id'] == key)
    projection = result['projection']
    assert projection['unified'] and projection['scope'] == 'process' and projection['total_mib'] == 38339
    assert projection['high_mib'] == result['high_mib'] - get('electron')['high_mib']  # no reserve: Electron and other apps are other processes
    assert any(w.startswith("Apple unified memory: the budget is Metal's recommended working set for this process (38,339 MiB of 49,152 MiB RAM)") for w in result['warnings'])
    assert 'CUDA graphs' not in get('llm_compute')['basis'] and 'counts this process only' in result['suggestion']['basis']
    assert 'double-count' not in result['suggestion']['basis'] and result['suggestion']['live_context_tokens'] > 0


def test_torch_model_rows_follow_their_device_settings(tmp_path, monkeypatch):
    import sys
    from process.app_core.resources import vram_estimate
    monkeypatch.setattr(vram_estimate, 'weights_mib', lambda model_id, cache_dir=None, path=None, revision=None: {'SupersonicLabs/Julia-1': 1000, 'sentence-transformers/all-MiniLM-L6-v2': 100}.get(model_id))
    c = config(tmp_path)
    row = lambda result, key: next(row for row in result['components'] if row['id'] == key)
    result = estimate(c, gpu(), metadata())  # the defaults: emotion off, memory models on the CPU
    assert row(result, 'emotion_model')['high_mib'] == 0 and row(result, 'memory_models')['high_mib'] == 0 and row(result, 'gpu_runtime')['high_mib'] == 0
    c.emotion.enabled, c.emotion.device, c.memory.device = True, 'cuda', 'cuda'
    result = estimate(c, gpu(), metadata())
    assert (row(result, 'emotion_model')['low_mib'], row(result, 'emotion_model')['high_mib']) == (1128, 1428)
    assert (row(result, 'memory_models')['low_mib'], row(result, 'memory_models')['high_mib']) == (1128 + 228, 1428 + 258)
    assert (row(result, 'gpu_runtime')['low_mib'], row(result, 'gpu_runtime')['high_mib']) == (128, 384)
    c.emotion.device = c.memory.device = 'mps'  # Julia maps MPS to the CPU; the sentence embedder runs on MPS
    result = estimate(c, gpu(), metadata())
    assert row(result, 'emotion_model')['high_mib'] == 0 and row(result, 'memory_models')['high_mib'] == 258
    assert row(result, 'gpu_runtime')['high_mib'] == 0 and 'Metal (MPS)' in row(result, 'gpu_runtime')['basis']
    monkeypatch.delitem(sys.modules, 'torch', raising=False)  # auto is resolved only once torch is loaded: a range until then
    monkeypatch.setattr(vram_estimate, 'host_platform', lambda: 'linux')
    c.emotion.device = 'auto'
    result = estimate(c, gpu(), metadata())
    assert (row(result, 'emotion_model')['low_mib'], row(result, 'emotion_model')['high_mib']) == (0, 1428) and row(result, 'gpu_runtime')['high_mib'] == 384
    monkeypatch.setattr(vram_estimate, 'host_platform', lambda: 'darwin')  # on a Mac auto is MPS or the CPU: never a CUDA context
    c.memory.device = 'auto'
    result = estimate(c, gpu(), metadata())
    assert row(result, 'emotion_model')['high_mib'] == 0 and 'on the CPU' in row(result, 'emotion_model')['basis']  # Julia maps MPS to the CPU
    assert (row(result, 'memory_models')['low_mib'], row(result, 'memory_models')['high_mib']) == (0, 258) and 'on mps (auto' in row(result, 'memory_models')['basis']
    assert row(result, 'gpu_runtime')['high_mib'] == 0 and 'Metal (MPS)' in row(result, 'gpu_runtime')['basis']
    monkeypatch.setattr(vram_estimate, 'host_platform', lambda: 'linux')
    c.emotion.model_id = 'not/downloaded'
    c.emotion.device = 'cuda'
    result = estimate(c, gpu(), metadata())
    assert row(result, 'emotion_model')['high_mib'] is None and not result['complete'] and result['suggestion'] is None


def test_weights_come_from_a_local_folder_or_the_cache_and_are_never_downloaded(tmp_path, monkeypatch):
    from process.app_core.resources.vram_estimate import weights_mib
    folder = tmp_path / 'julia'
    folder.mkdir()
    (folder / 'model.safetensors').write_bytes(b'0' * 2 * 1024 ** 2)
    (folder / 'tokenizer.json').write_text('{}')
    assert weights_mib('ignored', path=folder) == 2
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    assert weights_mib('nobody/not-cached-anywhere', cache_dir=tmp_path / 'empty-cache') is None


def test_a_pinned_snapshots_weights_are_found_at_its_commit(tmp_path, monkeypatch):
    # snapshot_download(revision=<commit>) writes no refs/main, so the estimate looks the weights up at that commit.
    from process.app_core.resources.vram_estimate import weights_mib
    commit = 'a' * 40
    snapshot = tmp_path / 'models--someone--julia' / 'snapshots' / commit
    snapshot.mkdir(parents=True)
    (snapshot / 'model.safetensors').write_bytes(b'0' * 3 * 1024 ** 2)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    assert weights_mib('someone/julia', cache_dir=tmp_path, revision=commit) == 3
    assert weights_mib('someone/julia', cache_dir=tmp_path) is None  # main: not downloaded as far as the cache knows


def test_an_integrated_gpu_takes_ggml_shared_pool_not_the_vendor_carve_out(tmp_path):
    """amdgpu's mem_info_vram_* and DXGI's DedicatedVideoMemory are an APU's small carve-out; llama.cpp budgets the shared
    pool ggml reports (Vulkan sums every heap of an iGPU), so the row and the estimate use that."""
    card = tmp_path / 'sys/class/drm/card0/device'
    card.mkdir(parents=True)
    for name, value in {'vendor': '0x1002', 'mem_info_vram_total': str(512 * 1024 ** 2), 'mem_info_vram_used': str(300 * 1024 ** 2),
            'product_name': 'AMD Radeon 780M', 'uevent': 'PCI_SLOT_NAME=0000:c4:00.0\n'}.items():
        (card / name).write_text(value)
    apu = [ggml(0, 'Vulkan0', 'Vulkan', 'igpu', '0000:c4:00.0', total=30720, free=24576, description='AMD Radeon 780M', model=True, main=True),
           ggml(1, 'CPU', 'CPU', 'cpu', total=32768, free=32768)]
    fdinfo = tmp_path / 'proc' / str(__import__('os').getpid()) / 'fdinfo'
    fdinfo.mkdir(parents=True)
    (fdinfo / '5').write_text('drm-pdev:\t0000:c4:00.0\ndrm-client-id:\t3\ndrm-memory-vram:\t102400 KiB\n')
    telemetry = GPUMonitor(sources=[gpu_sources.AmdgpuSysfs(tmp_path / 'sys'), gpu_sources.DrmFdinfo(tmp_path / 'proc')]).sample(Provider(apu))
    [row] = telemetry['gpus']
    assert (row['key'], row['memory_kind'], row['total_mib'], row['used_mib'], row['free_mib']) == ('pci:0000:c4:00.0', 'unified', 30720, 6144, 24576)
    assert not row['attribution_complete'] and row['other_mib'] is None  # fdinfo's VRAM is the carve-out, not the pool
    result = estimate(config(tmp_path), telemetry, metadata())
    assert result['gpu_key'] == 'pci:0000:c4:00.0' and result['suggestion']['live_context_tokens'] > 0
    assert any(w.startswith('Integrated GPU') for w in result['warnings']) and not any('-1' in w for w in result['warnings'])
    source = gpu_sources.WindowsCounters(adapters=lambda: [{'name': 'Intel(R) Arc(TM) Graphics', 'vendor': 0x8086, 'dedicated': 128 * 1024 ** 2,
        'luid': '00000000:0000a001', 'pci_bus_id': '0000:00:02.0'}])
    source.sample = lambda: ({'00000000:0000a001': 100 * 1024 ** 2}, {})
    arc = [ggml(0, 'Vulkan0', 'Vulkan', 'igpu', '0000:00:02.0', total=16384, free=12288, description='Intel(R) Arc(TM) Graphics')]
    [row] = GPUMonitor(sources=[source]).sample(Provider(arc))['gpus']
    assert (row['total_mib'], row['used_mib'], row['memory_kind'], row['attribution_complete']) == (16384, 4096, 'unified', False)


def test_ggml_free_above_total_is_unknown_not_a_negative_used():
    over = [ggml(0, 'Vulkan0', 'Vulkan', 'gpu', '0000:03:00.0', total=16368, free=0)]
    over[0]['memory_free'] = 2 ** 64 - 5 * 1024 ** 2  # Vulkan's unsigned budget - usage wrapped: a heap over budget
    [row] = GPUMonitor(sources=[]).sample(Provider(over))['gpus']
    assert row['total_mib'] == 16368 and row['used_mib'] is None and row['free_mib'] is None


def test_a_draft_on_another_native_library_does_not_use_the_running_builds_device_numbering(tmp_path):
    from process.app_core.resources.vram_estimate import draft_telemetry
    telemetry = telemetry_for(CUDA, [smi(SMI)])
    running, draft = config(tmp_path, split_mode='none', main_gpu=1), config(tmp_path, split_mode='none', main_gpu=1)
    running.runtime.native_library = draft.runtime.native_library = tmp_path / 'native/cuda/libriko-native.so'
    assert draft_telemetry(telemetry, running.runtime, draft.runtime) is telemetry  # the same library: its numbering holds
    draft.runtime.native_library = tmp_path / 'native/vulkan/libriko-native.so'
    other = draft_telemetry(telemetry, running.runtime, draft.runtime)
    assert other['ggml_devices'] is None and telemetry['ggml_devices'] is not None
    result = estimate(draft, other, metadata())
    assert any('without riko_devices (the draft loads another native library' in w for w in result['warnings'])
    assert result['gpu_key'] == 'pci:0000:02:00.0'  # nvidia-smi's GPU 1, as the warning says: a guess


def test_unified_memory_without_gpu_layers_shows_no_projection_and_the_suggestion_leaves_ram_to_others(tmp_path):
    def metal(ram): return [ggml(0, 'MTL0', 'MTL', 'gpu', total=38339, free=37791, description='Apple M5 Pro', model=True, main=True),
                            ggml(1, 'CPU', 'CPU', 'cpu', total=ram, free=ram)]
    result = estimate(config(tmp_path, n_gpu_layers=0), telemetry_for(metal(49152)), metadata())
    assert result['projection'] is None and any('offload 0 the model runs on the CPU' in w for w in result['warnings'])
    capped = estimate(config(tmp_path), telemetry_for(metal(49152)), metadata())['suggestion']
    roomy = estimate(config(tmp_path), telemetry_for(metal(10 ** 6)), metadata())['suggestion']  # RAM never binds: the working set does
    assert 0 < capped['maximum_pool_tokens'] < roomy['maximum_pool_tokens'] and '75% of RAM' in capped['basis']
