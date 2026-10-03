import ctypes
import threading
from types import SimpleNamespace

import pytest

from process.app_core.inference.llama_native import NativeClient, NativeRuntime
from process.app_core.configuration.config import load_config
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
    from process.app_core.inference.llama_server import LlamaServerProvider
    with pytest.raises(RuntimeError, match='failed'): LlamaServerProvider._check_response(response)
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
