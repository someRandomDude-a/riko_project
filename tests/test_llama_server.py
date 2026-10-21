"""The external llama-server provider, against a fake server on a real loopback socket."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

import pytest

from process.app_core.configuration.config import RuntimeConfig, load_config
from process.app_core.configuration.settings_store import SettingsStore, field
from process.app_core.kernel.messages import ChatMessage
from process.app_core.kernel.cancellation import BackgroundPreempted
from process.app_core.inference import llama_server
from process.app_core.inference.llama_runtime import server_address
from process.app_core.inference.llama_server import LlamaServerProvider
from process.app_core.inference.providers import create_provider

EVENTS = [
    {'type': 'response.output_text.delta', 'delta': 'Hello ', 'timings': {'predicted_n': 2, 'predicted_per_second': 20}},
    {'type': 'response.output_text.delta', 'delta': 'world'},
    {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Hello world'}]},
        {'type': 'function_call', 'call_id': 'call-1', 'name': 'lookup', 'arguments': '{"query":"color"}'}]}}]


class FakeServer:
    """Speaks the llama-server routes Riko uses: /health, /slots, /apply-template, /tokenize, /v1/responses."""
    def __init__(self, slots=(8192, 8192), health=(200,), stall=None):  # stall: None, 'prefill' or 'stream'
        self.slots, self.health, self.stall = list(slots), list(health), stall
        self.requests, self.disconnected, self.streaming = [], threading.Event(), threading.Event()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args): pass
            def reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def wait_for_disconnect(self):
                # Like llama-server: send nothing more, and stop as soon as the client goes away.
                owner.streaming.set()
                self.connection.settimeout(.05)
                while owner.stall:
                    try:
                        if self.connection.recv(1, socket.MSG_PEEK) == b'': break
                    except socket.timeout: continue
                    except OSError: break
                else: return False
                owner.disconnected.set()
                return True
            def do_GET(self):
                owner.requests.append((self.path, None, self.headers.get('Authorization')))
                if self.path == '/health':
                    status = owner.health.pop(0) if len(owner.health) > 1 else owner.health[0]
                    self.reply(status, {'status': 'ok' if status == 200 else 'loading model'})
                elif self.path == '/slots': self.reply(200, [{'id': i, 'n_ctx': n} for i, n in enumerate(owner.slots)])
                else: self.reply(404, {'error': {'message': 'not found'}})
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
                owner.requests.append((self.path, body, self.headers.get('Authorization')))
                if self.path == '/apply-template': return self.reply(200, {'prompt': 'rendered'})
                if self.path == '/tokenize': return self.reply(200, {'tokens': [1, 2, 3]})
                if self.path != '/v1/responses': return self.reply(404, {'error': {'message': 'not found'}})
                if not body.get('stream'): return self.reply(200, {'status': 'completed', 'output': []})
                self.close_connection = True
                if owner.stall == 'prefill' and self.wait_for_disconnect(): return  # silent before any header
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    for index, event in enumerate(EVENTS):
                        self.wfile.write(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'.encode()); self.wfile.flush()
                        if index == 0 and owner.stall == 'stream' and self.wait_for_disconnect(): return
                except (BrokenPipeError, ConnectionResetError): owner.disconnected.set()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self): self.server.shutdown(); self.server.server_close()


def provider_for(url, **settings):
    return LlamaServerProvider(RuntimeConfig(provider='llama_server', base_url=url, api_key='secret', n_ctx=8192, **settings))


def test_llama_server_keeps_slots_exact_counts_tools_timings_and_auth():
    server = FakeServer()
    provider = provider_for(server.url + '/v1/')  # an OpenAI-style /v1 address works too
    text, timing = [], []
    try:
        result = provider.generate([ChatMessage('user', 'Hi')], on_delta=text.append, on_metrics=timing.append,
            tools=[{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}])
        assert result.message.content == 'Hello world' and ''.join(text) == 'Hello world'
        assert result.message.tool_calls[0].id == 'call-1' and result.message.tool_calls[0].arguments == {'query': 'color'}
        assert timing == [{'predicted_n': 2, 'predicted_per_second': 20}]
        assert provider.count_tokens([ChatMessage('user', 'Hi')]) == 3
        provider.reflection.generate([ChatMessage('user', 'Reflect')])
        provider.initiative.generate([ChatMessage('user', 'Consider')])
        provider.warmup()
    finally: provider.close(); server.close()
    payloads = [body for path, body, _ in server.requests if path == '/v1/responses']
    assert [body['id_slot'] for body in payloads] == [0, 1, 1, 0, 1]  # live keeps slot 0; warmup touches each slot
    assert all(body['cache_prompt'] for body in payloads) and payloads[0]['tools'][0]['name'] == 'lookup'
    assert {'/health', '/slots', '/apply-template', '/tokenize'} <= {path for path, _, _ in server.requests}
    assert all(auth == 'Bearer secret' for _, _, auth in server.requests)


def test_waits_while_the_model_loads_then_explains_a_slot_mismatch():
    server = FakeServer(slots=(2048, 2048), health=(503, 503, 200))
    provider = provider_for(server.url)
    try:
        with pytest.raises(RuntimeError) as error: provider.generate([ChatMessage('user', 'Hi')])
        assert '[2048, 2048]' in str(error.value) and '--parallel 2 --ctx-size 16384' in str(error.value)
        assert [path for path, _, _ in server.requests].count('/health') == 3
        assert provider.client is None  # nothing half-started; the next turn checks again
    finally: provider.close(); server.close()


def test_a_missing_server_fails_fast_with_the_launch_command():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    provider = provider_for(f'http://127.0.0.1:{port}', startup_timeout_seconds=600)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match=f'No llama-server is answering.*--port {port} --parallel 2'): provider.warmup()
        assert time.monotonic() - started < 5
    finally: provider.close()


@pytest.mark.parametrize('stall', ['prefill', 'stream'])
def test_cancelling_while_the_server_is_silent_stops_at_once_and_frees_the_slot(stall):
    server = FakeServer(stall=stall)
    provider = provider_for(server.url)  # request_timeout_seconds stays 120: cancelling must not wait for it
    stop = threading.Event()
    threading.Thread(target=lambda: server.streaming.wait(10) and stop.set(), daemon=True).start()
    try:
        started = time.monotonic()
        with pytest.raises(BackgroundPreempted): provider.generate([ChatMessage('user', 'Hi')], on_delta=lambda _: None, cancelled=stop.is_set)
        assert time.monotonic() - started < 5 and server.disconnected.wait(5)  # llama-server stops that request too
        server.stall = None
        assert provider.generate([ChatMessage('user', 'Again')]).message.content == 'Hello world'  # slot 0 was released
    finally: provider.close(); server.close()


def test_a_server_that_goes_away_is_checked_again_and_named_in_the_error():
    server = FakeServer()
    provider = provider_for(server.url)
    try:
        assert provider.generate([ChatMessage('user', 'Hi')]).message.content == 'Hello world'
        server.close()
        with pytest.raises(RuntimeError, match='No llama-server is answering at http://127.0.0.1.*--parallel 2'): provider.generate([ChatMessage('user', 'Again')])
        assert provider.client is None  # the next turn re-runs /health and /slots
    finally: provider.close()


def test_an_unreachable_host_fails_within_the_connect_timeout(monkeypatch):
    monkeypatch.setattr(llama_server, 'CONNECT_TIMEOUT', .3)
    provider = provider_for('http://10.255.255.1:8080', startup_timeout_seconds=600)  # not routable: no answer at all
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match='No llama-server is answering'): provider.warmup()
        assert time.monotonic() - started < 5
    finally: provider.close()


def test_configuration_settings_and_probe_rules(tmp_path):
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: llama_server\n  n_ctx: 8192\n', encoding='utf-8')
    config = load_config(path)  # no GGUF, native library or GPU settings: the server owns them
    assert config.runtime.base_url == 'http://127.0.0.1:8080'
    provider = create_provider(config.runtime)
    try: assert isinstance(provider, LlamaServerProvider) and not provider.capabilities.latent_probe and provider.probe_host is None
    finally: provider.close()
    path.write_text('runtime:\n  provider: llama_server\nemotion:\n  enabled: true\n  probe:\n    enabled: true\n', encoding='utf-8')
    with pytest.raises(ValueError, match='llama_cpp'): load_config(path)
    path.write_text('runtime:\n  provider: llama_server\n  base_url: http://127.0.0.1:8080\n', encoding='utf-8')
    store = SettingsStore(path)
    fields = {item['path']: item for item in store.snapshot()['fields']}
    assert {'runtime.base_url', 'runtime.api_key', 'runtime.parallel_slots'} <= fields.keys()
    # Sent for every provider, so a draft that switches to an OpenAI-compatible one shows it; the renderer hides it here.
    assert 'llama_server' not in fields['runtime.api_mode']['visible_when']['runtime.provider']
    assert 'llama_server' in fields['runtime.parallel_slots']['visible_when']['runtime.provider']
    assert store.validate({'runtime.temperature': .5}) == {'valid': True, 'errors': {}}  # the minimal config stays editable
    assert not store.validate({'runtime.base_url': 'http://127.0.0.1:8080/llama'})['valid']  # rejected before it is saved
    assert 'llama_server' in field('runtime.provider', 'llama_server')['options']
    path.write_text('runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n', encoding='utf-8')
    paths = {item['path'] for item in SettingsStore(path).snapshot()['fields']}
    assert {'runtime.base_url', 'runtime.api_key'} <= paths  # switching to llama_server can set the address in the same draft
    for text, problem in [('base_url: http://127.0.0.1:8080/llama', 'llama-server address'), ('base_url: http://user:pw@host:8080', 'llama-server address'),
                          ('api_key: 123456', 'quotes')]:
        path.write_text(f'runtime:\n  provider: llama_server\n  {text}\n', encoding='utf-8')
        with pytest.raises(ValueError, match=problem): load_config(path)
    path.write_text('runtime:\n  provider: llama_server\n  base_url:\n', encoding='utf-8')
    assert load_config(path).runtime.base_url == 'http://127.0.0.1:8080'
    assert server_address(' http://127.0.0.1:8080/v1/ ') == 'http://127.0.0.1:8080'
