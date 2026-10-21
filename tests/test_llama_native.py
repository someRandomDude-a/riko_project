import ctypes
import json
import threading
from types import SimpleNamespace

import pytest

from process.app_core.inference.llama_native import NativeClient, NativeRuntime
from process.app_core.configuration.config import load_config
from process.app_core.configuration.config import RuntimeConfig
from process.app_core.kernel.messages import ChatMessage
from process.app_core.kernel.turns import TurnContext
from process.app_core.inference.llama_native import InProcessLlamaProvider
from process.app_core.configuration.settings_store import SettingsStore, field


def fake_runtime(chunks):
    runtime = NativeRuntime.__new__(NativeRuntime)
    runtime.handle, runtime.closed = 123, False
    runtime.lock, runtime.requests = threading.Lock(), set()
    runtime.directory = None
    def request(handle, path, body, output, cancel, user):
        for status, data in chunks:
            buffer = ctypes.create_string_buffer(data)
            if not output(status, ctypes.cast(buffer, ctypes.c_void_p), len(data), user): break
        return 0
    runtime.dll = SimpleNamespace(riko_request=request)
    return runtime


def test_in_process_stream_has_no_socket_and_preserves_split_utf8():
    data = 'data: {"type":"response.output_text.delta","delta":"café"}\n\n'.encode()
    split = data.index('é'.encode()) + 1
    runtime = fake_runtime([(200, b''), (200, data[:split]), (200, data[split:])])
    with NativeClient(runtime) as client:
        with client.stream('POST', '/v1/responses', {}) as response:
            assert list(response.iter_lines()) == ['data: {"type":"response.output_text.delta","delta":"café"}', '']


def test_in_process_nonstream_body_and_repeated_read():
    runtime = fake_runtime([(200, b'{"tokens":[1,2]}')])
    response = NativeClient(runtime).post('/tokenize', json={'content': 'hi'})
    assert response.is_success
    assert response.json() == {'tokens': [1, 2]}
    assert response.read() == b'{"tokens":[1,2]}'


def test_native_error_is_visible_without_hanging():
    runtime = fake_runtime([(500, b'{"error":{"message":"failed"}}')])
    response = NativeClient(runtime).get('/props')
    assert not response.is_success
    assert response.json()['error']['message'] == 'failed'


def test_native_client_rejects_requests_after_cancellation():
    client = NativeClient(fake_runtime([]))
    client.close()
    with pytest.raises(RuntimeError, match='cancelled'): client.stream('POST', '/v1/responses')


def test_interval_is_ui_slider_and_live_only_change(tmp_path):
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: openai\nemotion:\n  enabled: false\n', encoding='utf-8')
    store = SettingsStore(path)
    snapshot = store.snapshot()
    spec = next(item for item in snapshot['fields'] if item['path'] == 'emotion.probe.interval_tokens')
    assert spec['kind'] == 'number' and spec['integer']
    assert spec['min'] == 1 and spec['max'] == 512 and spec['restart'] is False
    assert snapshot['values']['emotion.probe.interval_tokens'] == 32
    result = store.save({'emotion.probe.interval_tokens': 64}, snapshot['revision'])
    assert result['saved'] and not result['restart_required']
    config = load_config(path)
    assert config.emotion.probe == {'interval_tokens': 64}
    assert config.runtime.provider == 'openai' and config.emotion.enabled is False


def test_julia_gpu_option_does_not_enable_gpu_probe():
    assert field('emotion.device', 'cpu')['options'] == ['cpu', 'auto', 'cuda', 'cuda:0']
    assert field('memory.device', 'cpu')['options'] == ['cpu', 'auto', 'cuda', 'cuda:0', 'mps']
    from process.app_core.emotion.probe import ProbeConfig
    assert not hasattr(ProbeConfig.from_raw({'device': 'cuda'}), 'device')  # ignored (load_config reports it), never a GPU probe


def test_probe_cannot_silently_use_stock_http_server(tmp_path):
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: llama_cpp\n  model_path: test.gguf\nemotion:\n  enabled: true\n  probe:\n    enabled: true\n', encoding='utf-8')
    with pytest.raises(ValueError, match='native_library'): load_config(path)


def test_token_count_response_helpers_and_no_retained_bodies():
    client = NativeClient(fake_runtime([(200, b'{"tokens":[1,2]}')]))
    for _ in range(100):
        response = client.post('/tokenize')
        response.raise_for_status()
        assert response.text == '{"tokens":[1,2]}'
        assert not client.responses
    client.close()


def test_non_json_native_errors_have_readable_details():
    response = NativeClient(fake_runtime([(500, b'failed')])).get('/props')
    from process.app_core.inference.llama_context import LlamaContextProvider
    with pytest.raises(RuntimeError, match='failed'): LlamaContextProvider._check_response(response)
    with pytest.raises(RuntimeError, match='failed'): response.raise_for_status()


def test_request_thread_failure_reaches_caller():
    runtime = fake_runtime([])
    def broken(*args): raise OSError('DLL failure')
    runtime.dll.riko_request = broken
    with NativeClient(runtime) as client:
        with pytest.raises(OSError, match='DLL failure'): client.post('/props')


def test_cancellation_unblocks_wait_for_first_response():
    runtime = fake_runtime([])
    entered, release = threading.Event(), threading.Event()
    def stalled(*args):
        entered.set()
        assert release.wait(5)
        return 1
    runtime.dll.riko_request = stalled
    client = NativeClient(runtime)
    response = client.stream('POST', '/v1/responses')
    try:
        assert entered.wait(5)
        client.close()
        with pytest.raises(RuntimeError, match='cancelled'): response.__enter__()
    finally:
        release.set()
        response.thread.join(5)
    assert not response.thread.is_alive() and not runtime.requests


def test_native_idle_timeout_cancels_without_freeing_active_context(monkeypatch):
    runtime = fake_runtime([])
    entered, release = threading.Event(), threading.Event()
    def stalled(*args):
        entered.set()
        assert release.wait(5)
        return 1
    runtime.dll.riko_request = stalled
    client = NativeClient(runtime, timeout=1)
    response = client.stream('POST', '/v1/responses')
    try:
        assert entered.wait(5)
        clock = iter([0, 2])
        monkeypatch.setattr('process.app_core.inference.llama_native.time.monotonic', lambda: next(clock))
        with pytest.raises(TimeoutError): response.__enter__()
        assert response.stop.is_set() and runtime.handle == 123
    finally:
        release.set()
        response.thread.join(5)
        client.close()


def test_runtime_close_cancels_and_joins_before_destroy():
    runtime = fake_runtime([])
    entered, release = threading.Event(), threading.Event()
    destroyed = []
    def stalled(*args):
        entered.set()
        assert release.wait(5)
        return 1
    runtime.dll = SimpleNamespace(riko_request=stalled, riko_stop=lambda _: release.set(),
        riko_destroy=lambda handle: destroyed.append(handle))
    response = runtime.request('/v1/responses', {})
    assert entered.wait(5)
    runtime.close()
    assert response.stop.is_set() and not response.thread.is_alive()
    assert destroyed == [123] and runtime.handle is None
    runtime.close()
    assert destroyed == [123]


def test_provider_native_routes_startup_tools_timings_and_slots(tmp_path, monkeypatch):
    """Exercise the provider through real NativeClient/ctypes callbacks, not HTTP mocks."""
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'test-model')
    (tmp_path / 'riko-native.dll').write_bytes(b'test-library')
    runtime = fake_runtime([])
    requests, destroyed = [], []
    def request(handle, path, body, output, cancel, user):
        path, payload = path.decode(), json.loads(body)
        requests.append((path, payload))
        if path == '/slots': data = [{'n_ctx': 8192}, {'n_ctx': 8192}]
        elif path == '/apply-template': data = {'prompt': 'rendered'}
        elif path == '/tokenize': data = {'tokens': [1, 2, 3]}
        elif path == '/v1/responses':
            events = [
                {'type': 'response.output_text.delta', 'delta': 'Hello ',
                    'timings': {'predicted_n': 2, 'predicted_per_second': 20}},
                {'type': 'response.output_text.delta', 'delta': 'world'},
                {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Hello world'}]},
                    {'type': 'function_call', 'call_id': 'call-1', 'name': 'lookup', 'arguments': '{"query":"color"}'}]}}]
            data = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events).encode()
        else: raise AssertionError(path)
        if not isinstance(data, bytes): data = json.dumps(data).encode()
        buffer = ctypes.create_string_buffer(data)
        output(200, ctypes.cast(buffer, ctypes.c_void_p), len(data), user)
        return 0
    runtime.dll = SimpleNamespace(riko_request=request, riko_stop=lambda _: None,
        riko_destroy=lambda handle: destroyed.append(handle))
    def create(library, args, interval):
        assert args[0] == 'riko-native'
        assert '--host' not in args and '--port' not in args
        assert interval == 0
        return runtime
    monkeypatch.setattr('process.app_core.inference.llama_native.NativeRuntime', create)
    provider = InProcessLlamaProvider(RuntimeConfig(provider='llama_cpp', model_path=model,
        native_library=tmp_path / 'riko-native.dll'))
    text, timing = [], []
    try:
        result = provider.generate([ChatMessage('user', 'Hi')], on_delta=text.append,
            on_metrics=timing.append, tools=[{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}])
        assert result.message.content == 'Hello world'
        assert result.message.tool_calls[0].id == 'call-1'
        assert result.message.tool_calls[0].arguments == {'query': 'color'}
        assert ''.join(text) == 'Hello world'
        assert timing == [{'predicted_n': 2, 'predicted_per_second': 20}]
        assert provider.count_tokens([ChatMessage('user', 'Hi')]) == 3
        provider.reflection.generate([ChatMessage('user', 'Reflect')])
        provider.initiative.generate([ChatMessage('user', 'Consider')])
        payloads = [body for path, body in requests if path == '/v1/responses']
        assert [body['id_slot'] for body in payloads] == [0, 1, 1]
        assert all(body['cache_prompt'] and body['timings_per_token'] for body in payloads)
        assert payloads[0]['tools'][0]['name'] == 'lookup'
    finally:
        provider.close()
    assert destroyed == [123]


def test_llama_cpp_never_selects_external_server_without_library():
    from process.app_core.inference.providers import create_provider
    provider = create_provider(RuntimeConfig(provider='llama_cpp', model_path='unused.gguf'))
    try:
        assert isinstance(provider, InProcessLlamaProvider)
        with pytest.raises(RuntimeError, match='native_library'): provider.warmup()
    finally:
        provider.close()


def test_settings_hide_legacy_server_path(tmp_path):
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: openai\n  server_path: old-server\n', encoding='utf-8')
    snapshot = SettingsStore(path).snapshot()
    assert all(item['path'] != 'runtime.server_path' for item in snapshot['fields'])
    assert 'Required for llama_cpp' in field('runtime.native_library', None)['help']


def test_missing_library_fails_before_model_resolution(tmp_path, monkeypatch):
    def unexpected(config): raise AssertionError('No model download before library validation')
    monkeypatch.setattr('process.app_core.inference.llama_native.resolve_model', unexpected)
    provider = InProcessLlamaProvider(RuntimeConfig(model_path=tmp_path / 'model.gguf',
        native_library=tmp_path / 'missing.dll'))
    try:
        with pytest.raises(RuntimeError, match='native_library does not exist'): provider.warmup()
    finally:
        provider.close()


def test_native_probe_capture_checks_utf8_prefix_and_turn_alignment(monkeypatch):
    from process.app_core.emotion.probe_hook import FEATURE_VERSION, ProbeHook
    runtime = fake_runtime([])
    samples = []
    probe = SimpleNamespace(active_group=None, close=lambda: None)
    probe.activate = lambda group: setattr(probe, 'active_group', group)
    probe.capture = lambda features, text, group, **kwargs: samples.append((features, text, group, kwargs))
    def request(handle, path, body, output, cancel, user):
        path = path.decode()
        if path == '/apply-template': data = json.dumps({'prompt': 'rendered'}).encode()
        elif path == '/tokenize': data = json.dumps({'tokens': [1]}).encode()
        else:
            assert path == '/v1/responses'
            sample = {'type': 'riko.emotion_probe.sample', 'feature_version': FEATURE_VERSION,
                'prefix_bytes': 5, 'features': [0.] * 256}
            events = [{'type': 'response.output_text.delta', 'delta': 'café'},
                {**sample, 'prefix_bytes': 4}, sample,
                {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': 'café'}]}]}}]
            data = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events).encode()
        buffer = ctypes.create_string_buffer(data)
        output(200, ctypes.cast(buffer, ctypes.c_void_p), len(data), user)
        return 0
    runtime.dll.riko_request = request
    provider = InProcessLlamaProvider(RuntimeConfig(model_path='unused.gguf'))
    hook = ProbeHook(lambda identity, idle: probe)
    provider.attach_probe(hook, 32)
    hook.start({}, provider.probe_idle)  # what _start does once the model is up
    provider.native = runtime
    provider.client = NativeClient(runtime)
    monkeypatch.setattr(provider, '_start', lambda: None)
    runtime.dll.riko_stop = lambda _: None
    runtime.dll.riko_destroy = lambda _: None
    try:
        provider.generate([ChatMessage('user', 'Bonjour')], emotion_turn_id='message-one')
        assert len(samples) == 1
        features, text, group, options = samples[0]
        assert features.shape == (256,) and features.device.type == 'cpu'
        assert text == 'user: Bonjour\nassistant: café' and group == 'message-one'
        assert options['input_text'] == 'Bonjour' and options['offset'] == 4
        assert not options['cancelled']()
    finally:
        provider.close()


def gguf(path, context_length):
    import struct
    def text(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data
    entries = [text('general.architecture') + struct.pack('<I', 8) + text('llama'),
        text('llama.context_length') + struct.pack('<II', 4, context_length)]
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, len(entries)) + b''.join(entries))


def test_budgets_above_the_training_context_fail_before_loading(tmp_path, monkeypatch):
    gguf(tmp_path / 'model.gguf', 4096)
    (tmp_path / 'riko-native.dll').write_bytes(b'test-library')
    def unexpected(*args): raise AssertionError('The model must not load')
    monkeypatch.setattr('process.app_core.inference.llama_native.NativeRuntime', unexpected)
    config = RuntimeConfig(provider='llama_cpp', model_path=tmp_path / 'model.gguf', native_library=tmp_path / 'riko-native.dll', n_ctx=8192)
    config.initiative_n_ctx = config.reflection_n_ctx = 4096
    provider = InProcessLlamaProvider(config)
    try:
        with pytest.raises(RuntimeError, match=r'trained for 4096 tokens.*Lower runtime\.n_ctx \(8192\) to 4096'): provider.warmup()
    finally:
        provider.close()


def test_short_slots_and_cpu_fallback_are_reported(tmp_path, monkeypatch, caplog):
    gguf(tmp_path / 'model.gguf', 32768)
    (tmp_path / 'riko-native.dll').write_bytes(b'test-library')
    runtime = fake_runtime([])
    runtime.notes = 'riko-native: no GPU device from libggml-vulkan; llama.cpp runs on the CPU (is the GPU driver installed? ggml-hip also needs ROCm)\nload_tensors: offloaded 0/37 layers to GPU'
    def request(handle, path, body, output, cancel, user):
        data = json.dumps([{'n_ctx': 4096}, {'n_ctx': 4096}]).encode()
        buffer = ctypes.create_string_buffer(data)
        output(200, ctypes.cast(buffer, ctypes.c_void_p), len(data), user)
        return 0
    runtime.dll = SimpleNamespace(riko_request=request, riko_stop=lambda _: None, riko_destroy=lambda _: None)
    monkeypatch.setattr('process.app_core.inference.llama_native.NativeRuntime', lambda *args: runtime)
    provider = InProcessLlamaProvider(RuntimeConfig(provider='llama_cpp', model_path=tmp_path / 'model.gguf',
        native_library=tmp_path / 'riko-native.dll', n_ctx=8192))
    try:
        with caplog.at_level('WARNING'), pytest.raises(RuntimeError, match=r'allocated 2 slots of \[4096\] tokens, but Riko needs 2 slots of 8192'):
            provider.warmup()
        assert 'only 0 of 37 layers on the GPU' in caplog.text
        assert 'WARNING' in caplog.text and 'no GPU device from libggml-vulkan; llama.cpp runs on the CPU' in caplog.text  # the bridge's notes
    finally:
        provider.close()


class FakeNativeLibrary:
    """Stands in for ctypes.CDLL(riko-native): plain functions accept the argtypes/restype that _bind sets."""
    def __init__(self, create):
        self.destroyed = []
        def riko_create(*args): return create(*args)
        def riko_request(*args): return 0
        def riko_set_interval(*args): return 0
        def riko_stop(handle): pass
        def riko_destroy(handle): self.destroyed.append(handle)
        self.riko_create, self.riko_request, self.riko_set_interval = riko_create, riko_request, riko_set_interval
        self.riko_stop, self.riko_destroy = riko_stop, riko_destroy


def test_native_runtime_is_live_from_create_until_destroy(monkeypatch, tmp_path):
    import process.app_core.inference.llama_native as native
    library = FakeNativeLibrary(lambda *args: 77)
    monkeypatch.setattr(native.ctypes, 'CDLL', lambda path: library)
    runtime = native.NativeRuntime(tmp_path / 'riko-native.dll', ['riko-native'])
    try:
        assert runtime in native._LIVE and runtime.handle == 77
        runtime.close()
        assert library.destroyed == [77] and runtime not in native._LIVE and runtime.handle is None
    finally: native._LIVE.discard(runtime)
    def failed(arguments, interval, error, capacity):
        error.value = b'native model initialization failed'
    monkeypatch.setattr(native.ctypes, 'CDLL', lambda path: FakeNativeLibrary(failed))
    before = set(native._LIVE)
    with pytest.raises(RuntimeError, match='initialization failed'): native.NativeRuntime(tmp_path / 'riko-native.dll', ['riko-native'])
    assert native._LIVE == before


def device(index, name, backend, kind, device_id=None, total=8 << 30, free=6 << 30, description='Device'):
    return {'index': index, 'name': name, 'description': description, 'backend': backend, 'type': kind, 'device_id': device_id,
        'memory_free': free, 'memory_total': total}


def devices_library(results, calls):
    """FakeNativeLibrary with a riko_devices answering each call from `results`: (length, text written)."""
    library = FakeNativeLibrary(lambda *args: 77)
    def riko_devices(buffer, capacity):
        length, text = results.pop(0)
        calls.append(capacity)
        if text is not None and (length < 0 or length < capacity): buffer.value = text.encode()
        return length
    library.riko_devices = riko_devices
    return library


def test_riko_devices_is_optional_parsed_retried_and_never_runs_at_exit(monkeypatch, tmp_path, caplog):
    import process.app_core.inference.llama_native as native
    monkeypatch.setattr(native.ctypes, 'CDLL', lambda path: FakeNativeLibrary(lambda *args: 77))
    old = native.NativeRuntime(tmp_path / 'riko-native.dll', ['riko-native'])  # a bridge from before riko_devices
    try: assert old.devices_function is None and old.devices() is None
    finally: old.close()
    listed = [device(0, 'CUDA0', 'CUDA', 'gpu', '0000:01:00.0'), device(1, 'CPU', 'CPU', 'cpu', total=64 << 30)]
    text = json.dumps({'version': 1, 'devices': listed})
    calls, results = [], []
    monkeypatch.setattr(native.ctypes, 'CDLL', lambda path: devices_library(results, calls))
    runtime = native.NativeRuntime(tmp_path / 'riko-native.dll', ['riko-native'])
    try:
        assert runtime.devices_function.argtypes == [ctypes.c_char_p, ctypes.c_size_t] and runtime.devices_function.restype is ctypes.c_int
        results[:] = [(len(text), text)]
        assert runtime.devices() == listed and calls == [16384]
        results[:] = [(20000, None), (len(text), text)]  # too long for the first buffer: once more with room for it
        calls.clear()
        assert runtime.devices() == listed and calls == [16384, 20256]
        results[:] = [(-2, 'riko_devices before riko_create')]
        assert runtime.devices() is None and 'riko_devices' not in caplog.text  # not initialised yet: not a failure
        for answer in [(-1, 'no backend'), (len('{"version": 2}'), '{"version": 2}'),
                (len(json.dumps({'version': 1, 'devices': [{**listed[0], 'type': 'tpu'}]})), json.dumps({'version': 1, 'devices': [{**listed[0], 'type': 'tpu'}]}))]:
            results[:] = [answer]
            assert runtime.devices() is None
        assert caplog.text.count('riko_devices failed') == 1 and 'no backend' in caplog.text  # logged once, then quiet
        monkeypatch.setattr(native, '_EXITING', True)  # _release_at_exit: no call may run into exit()'s static destructors
        results[:] = [(len(text), text)]
        assert runtime.devices() is None and results
    finally: runtime.close()
    assert runtime.devices() is None  # closed


def test_release_at_exit_stops_device_queries_first(monkeypatch):
    import process.app_core.inference.llama_native as native
    monkeypatch.setattr(native, '_EXITING', False)
    monkeypatch.setattr(native, '_LIVE', set())
    native._release_at_exit()
    assert native._EXITING is True and not native._DEVICES_LOCK.locked()


def test_model_devices_follows_llama_cpp_default_device_selection():
    # llama_prepare_model_devices (src/llama.cpp at the pin): RPC first, discrete GPUs once per device_id, iGPUs only without
    # a discrete GPU (and only the first iGPU's backend), CPU and BLAS never; split_mode none keeps devices[main_gpu].
    from process.app_core.inference.devices import model_devices
    cuda0, cuda1 = device(0, 'CUDA0', 'CUDA', 'gpu', '0000:02:00.0'), device(1, 'CUDA1', 'CUDA', 'gpu', '0000:01:00.0')
    vulkan = device(2, 'Vulkan0', 'Vulkan', 'gpu', '0000:01:00.0')  # the same card as CUDA1 through another backend
    igpu, igpu_vk = device(3, 'Vulkan1', 'Vulkan', 'igpu', '0000:00:02.0'), device(4, 'SYCL0', 'SYCL', 'igpu', '0000:00:02.0')
    rpc, cpu, blas = device(5, 'RPC0', 'RPC', 'gpu'), device(6, 'CPU', 'CPU', 'cpu'), device(7, 'BLAS', 'BLAS', 'accel')
    names = lambda chosen: chosen if chosen is None else [d['name'] for d in chosen]
    every = [cuda0, cuda1, vulkan, igpu, igpu_vk, rpc, cpu, blas]
    assert names(model_devices(every, split_mode='layer', main_gpu=0)) == ['RPC0', 'CUDA0', 'CUDA1']
    assert names(model_devices(every, split_mode='none', main_gpu=2)) == ['CUDA1']  # RPC counts in main_gpu's indexing
    assert names(model_devices([igpu, igpu_vk, cpu], split_mode='layer', main_gpu=0)) == ['Vulkan1']
    assert names(model_devices([igpu, device(8, 'Vulkan2', 'Vulkan', 'igpu'), cpu], split_mode='layer', main_gpu=0)) == ['Vulkan1', 'Vulkan2']
    assert names(model_devices([cuda0, igpu, cpu], split_mode='layer', main_gpu=0)) == ['CUDA0']
    assert names(model_devices([cuda0, device(1, 'CUDA1', 'CUDA', 'gpu'), device(2, 'CUDA2', 'CUDA', 'gpu')], split_mode='layer', main_gpu=0)) == ['CUDA0', 'CUDA1', 'CUDA2']  # no id: never a duplicate
    assert names(model_devices([cuda0, cpu], split_mode='none', main_gpu=1)) is None  # llama.cpp refuses this main_gpu
    assert names(model_devices([cpu, blas], split_mode='none', main_gpu=3)) == []  # the CPU only, whatever main_gpu says
    metal = device(0, 'MTL0', 'MTL', 'gpu', description='Apple M5 Pro')  # Metal reports a gpu, without a PCI id
    assert names(model_devices([metal, blas, cpu], split_mode='none', main_gpu=0)) == ['MTL0']


def test_provider_devices_mark_the_model_and_main_device(tmp_path):
    from types import SimpleNamespace as NS
    library = tmp_path / 'riko-native.dll'; library.write_bytes(b'library')
    provider = InProcessLlamaProvider(RuntimeConfig(provider='llama_cpp', model_path=tmp_path / 'model.gguf', native_library=library, split_mode='layer'))
    listed = [device(0, 'CUDA0', 'CUDA', 'gpu', '0000:02:00.0'), device(1, 'CUDA1', 'CUDA', 'gpu', '0000:01:00.0'), device(2, 'CPU', 'CPU', 'cpu')]
    try:
        assert provider.devices() is None  # before the model loads
        provider.native = NS(devices=lambda: listed)
        assert [(d['name'], d['model'], d['main']) for d in provider.devices()] == [('CUDA0', True, False), ('CUDA1', True, False), ('CPU', False, False)]
        provider.config.split_mode, provider.config.main_gpu = 'none', 1
        assert [(d['name'], d['model'], d['main']) for d in provider.devices()] == [('CUDA0', False, False), ('CUDA1', True, True), ('CPU', False, False)]
        provider.config.n_gpu_layers = 0  # no layer offloaded: no device runs the model
        assert not any(d['model'] or d['main'] for d in provider.devices())
        provider.native = NS(devices=lambda: None)  # an older bridge
        assert provider.devices() is None
    finally:
        provider.native = None
        provider.close()


def test_release_native_refuses_while_loading_and_waits_for_a_destroy_in_progress(monkeypatch, tmp_path):
    import process.app_core.inference.llama_native as native
    loading, loaded, entered, release = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    def create(*args):
        loading.set()
        assert loaded.wait(5)
        return 5
    library = FakeNativeLibrary(create)
    monkeypatch.setattr(native.ctypes, 'CDLL', lambda path: library)
    built = []
    thread = threading.Thread(target=lambda: built.append(native.NativeRuntime(tmp_path / 'riko-native.dll', ['riko-native'])))
    thread.start()
    try:
        assert loading.wait(5)
        assert native.release_native(5) is False  # riko_create cannot be interrupted: exit must skip destructors
        loaded.set(); thread.join(5)
        def stalled(*args):
            entered.set()
            assert release.wait(5)  # a native call still inside llama.cpp when shutdown starts
            return 1
        library.riko_request = stalled
        built[0].request('/v1/responses', {})
        assert entered.wait(5)
        closing = threading.Thread(target=built[0].close); closing.start()  # e.g. an abandoned close_bounded
        assert native.release_native(.2) is False and library.destroyed == []
        release.set()
        assert native.release_native(5) is True and library.destroyed == [5]
        closing.join(5)
    finally:
        loaded.set(); release.set(); native._LIVE.difference_update(built)


def test_exit_guard_skips_native_destructors_only_when_a_context_cannot_be_destroyed():
    import os, subprocess, sys
    from pathlib import Path
    script = """
import sys
import process.app_core.inference.llama_native as native
class Runtime:
    handle = None if sys.argv[1] == 'loading' else 1
    def close(self): native._LIVE.discard(self)
native._LIVE.add(Runtime())
sys.exit(3)
"""
    env = {**os.environ, 'PYTHONPATH': str(Path('Code').resolve())}
    run = lambda state: subprocess.run([sys.executable, '-c', script, state], env=env, capture_output=True, text=True, timeout=120)
    ready, loading = run('ready'), run('loading')
    assert ready.returncode == 3, ready.stderr  # destroyed in time: the real exit status survives
    assert loading.returncode == 1 and 'leaving it to the OS' in loading.stderr, loading.stderr


@pytest.mark.parametrize('requested, notes, identity', [
    ('auto', 'load_tensors: offloaded 37/37 layers to GPU\nresolve_fused_ops: Flash Attention enabled', True),
    ('auto', 'resolve_fused_ops: Flash Attention not supported, set to disabled', False),
    ('auto', '', 'auto'),  # an older bridge reports nothing
    ('on', 'resolve_fused_ops: Flash Attention not supported, set to disabled', True), (False, '', False)])
def test_probe_identity_records_the_flash_attention_llama_cpp_chose(requested, notes, identity):
    provider = InProcessLlamaProvider(RuntimeConfig(provider='llama_cpp', model_path='unused.gguf', flash_attn=requested))
    try:
        provider.native = SimpleNamespace(notes=notes)
        assert provider._flash_attention_identity() == identity
    finally: provider.native = None; provider.close()


def chat_over_native(monkeypatch, replies, tool_output='found'):
    """ChatService on the native provider; the fake tokenizer counts four characters per token."""
    from process.app_core.conversation.chat import ChatDeps, ChatService
    from process.app_core.tools.registry import RegisteredTool, ToolRegistry
    runtime, requests, replies = fake_runtime([]), [], iter(replies)
    def request(handle, path, body, output, cancel, user):
        path, payload = path.decode(), json.loads(body)
        requests.append((path, payload))
        if path == '/apply-template': data = {'prompt': '\n'.join(m['content'] for m in payload['messages'])}
        elif path == '/tokenize': data = {'tokens': [0] * (len(payload['content']) // 4)}
        else: data = {'type': 'response.completed', 'response': {'status': 'completed', 'output': [next(replies)]}}
        data = ('data: ' + json.dumps(data) + '\n\n' if path == '/v1/responses' else json.dumps(data)).encode()
        buffer = ctypes.create_string_buffer(data)
        output(200, ctypes.cast(buffer, ctypes.c_void_p), len(data), user)
        return 0
    runtime.dll = SimpleNamespace(riko_request=request, riko_stop=lambda _: None, riko_destroy=lambda _: None)
    provider = InProcessLlamaProvider(RuntimeConfig(model_path='unused.gguf'))
    provider.native, provider.client = runtime, NativeClient(runtime)
    monkeypatch.setattr(provider, '_start', lambda: None)
    registry = ToolRegistry()
    registry.tools['lookup'] = RegisteredTool('lookup', 'Look it up', {'type': 'object', 'properties': {}}, lambda args: tool_output)
    chat = ChatService(provider, system_prompt='Riko', tool_registry=registry, deps=ChatDeps(context_limit=8192))
    chat.conversation.append([ChatMessage(role, f'{role} {i} ' + 'q' * 400) for i in range(120) for role in ('user', 'assistant')])
    return chat, requests


CALL = {'type': 'function_call', 'call_id': 'call', 'name': 'lookup', 'arguments': '{}'}
SAID = {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Summary'}]}


def test_native_chat_keeps_sent_observations_and_a_fixed_prefix_with_one_count_per_reply(monkeypatch):
    chat, requests = chat_over_native(monkeypatch, [CALL, CALL, SAID] + [SAID] * 5)
    observed = iter(range(100))
    observe = TurnContext(runtime=lambda: {'observation': next(observed)})
    try:
        for turn in range(6): assert chat.respond(f'turn {turn} ' + 'z' * 400, context=observe).message.content == 'Summary'
    finally: chat.tool_registry.close(); chat.provider.close()
    inputs = [body['input'] for path, body in requests if path == '/v1/responses']
    counts, pending = [], 0
    for path, _ in requests:
        if path == '/apply-template': pending += 1
        elif path == '/v1/responses': counts.append(pending); pending = 0
    assert len(inputs) == 8
    for number, items in enumerate(inputs):  # every request carries the latest observation
        assert f'{{\\"observation\\": {number}}}' in json.dumps(items)
    # A tool loop keeps the observations it already sent, so each of its requests extends the previous one.
    assert [json.dumps(items).count('Current runtime observation') for items in inputs] == [1, 2, 3, 1, 1, 1, 1, 1]
    assert all(later[:len(earlier)] == earlier for earlier, later in zip(inputs[:2], inputs[1:3]))
    assert counts[1:] == [1] * 7  # after the first pack, one template+tokenize per request
    assert len({json.dumps(items[1]) for items in inputs}) == 1  # one retained start for every request
    for earlier, later in zip(inputs[3:], inputs[4:]):  # each reply's prompt, minus its current input, prefixes the next
        assert later[:len(earlier) - 1] == earlier[:-1]


def test_native_chat_answers_with_cut_tool_output_instead_of_failing(monkeypatch):
    chat, requests = chat_over_native(monkeypatch, [CALL, SAID], 'HEAD' + 'x' * 100000 + 'TAIL')
    try: assert chat.respond('summarise report.pdf').message.content == 'Summary'
    finally: chat.tool_registry.close(); chat.provider.close()
    sent = [body['input'] for path, body in requests if path == '/v1/responses'][-1]
    result = next(item['output'] for item in sent if item.get('type') == 'function_call_output')
    assert result.startswith('HEAD') and result.endswith('TAIL') and '[tool output truncated to fit the context: ' in result
    prompt = sum(len(item['output'] if 'output' in item else item.get('content') if isinstance(item.get('content'), str) else '') for item in sent)
    assert prompt // 4 + 1024 <= 8192 and len(result) > 20000
    assert [m.content for m in chat.history[-2:]] == ['User: summarise report.pdf', 'Summary']
