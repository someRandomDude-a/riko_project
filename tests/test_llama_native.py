import ctypes
import json
import threading
from types import SimpleNamespace

import pytest

from process.app_core.inference.llama_native import NativeClient, NativeRuntime
from process.app_core.configuration.config import load_config
from process.app_core.configuration.config import RuntimeConfig
from process.app_core.conversation.messages import ChatMessage
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
    assert field('emotion.device', 'cpu')['options'] == ['cpu', 'cuda', 'cuda:0']
    from process.app_core.emotion.probe import ProbeConfig
    with pytest.raises(ValueError): ProbeConfig.from_raw({'device': 'cuda'})


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
    from process.app_core.emotion.probe import FEATURE_VERSION
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
    provider.native = runtime
    provider.client = NativeClient(runtime)
    provider.probe = probe
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
