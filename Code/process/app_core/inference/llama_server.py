"""External llama-server transport: Riko talks to a llama-server that the user runs.

Any llama.cpp build works (CUDA, ROCm/HIP, Metal, Vulkan or CPU), on this machine or
another one. Riko keeps its scheduling: slot 0 is reserved for the live lane, every
request is packed with the server's own template and tokenizer, and cancelling a request
shuts its connection down. The server owns the model, its GPU settings and the context it
allocated, so the runtime.* model and compute settings do not apply. Stock servers expose
no hidden states, so the emotion probe needs the native library (runtime.provider: llama_cpp).
"""
import http.client
from json import dumps, loads
import logging
import socket
import threading
import time
from urllib.parse import urlsplit

from .llama_context import LlamaContextProvider
from .llama_runtime import server_address

logger = logging.getLogger(__name__)
CONNECT_TIMEOUT = 5.0


class Unreachable(ConnectionError): pass


class ServerResponse:
    """The parts of an httpx-style response that LlamaContextProvider uses."""
    def __init__(self, client, connection, response):
        self.client, self.connection, self.response = client, connection, response
        self.status_code, self.content, self.consumed = response.status, b'', False

    @property
    def is_success(self): return 200 <= self.status_code < 300

    @property
    def text(self): return self.content.decode('utf-8', errors='replace')

    def read(self):
        if not self.consumed: self.content, self.consumed = self.response.read(), True
        return self.content

    def json(self): return loads(self.read())

    def raise_for_status(self):
        if not self.is_success: raise RuntimeError(f'llama-server operation {self.status_code}: {self.read().decode("utf-8", errors="replace")[:2000]}')

    def iter_lines(self):
        while True:
            line = self.response.readline(1 << 20)
            if not line: return
            yield line.rstrip(b'\r\n').decode('utf-8')

    def close(self): self.client._release(self.connection)
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class ServerClient:
    """One connection per request, so threads never share one and any of them can be cancelled.

    close() shuts the open sockets down: that wakes a read blocked while llama-server prefills
    (closing alone does not, and on Linux does not even send FIN) and stops the server's work at once.
    """
    def __init__(self, root, api_key, timeout):
        parts = urlsplit(root)
        self.kind = http.client.HTTPSConnection if parts.scheme == 'https' else http.client.HTTPConnection
        self.host, self.port, self.timeout = parts.hostname, parts.port, timeout
        self.headers = {'Authorization': 'Bearer ' + api_key} if api_key else {}
        self.connections, self.lock, self.closed = {}, threading.Lock(), False  # connection -> its socket

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def _open(self, timeout):
        if self.closed: raise RuntimeError('llama-server request cancelled')
        connection = self.kind(self.host, self.port, timeout=min(CONNECT_TIMEOUT, timeout))
        try: connection.connect()
        except OSError as exc:
            connection.close()
            raise Unreachable(str(exc) or type(exc).__name__) from None
        connection.sock.settimeout(timeout)  # then it bounds inactivity, not the whole reply
        with self.lock:
            if not self.closed:
                # Kept here because http.client hands the socket to the response (and forgets it) on Connection: close.
                self.connections[connection] = connection.sock
                return connection
        connection.close()
        raise RuntimeError('llama-server request cancelled')

    def _release(self, connection):
        with self.lock: self.connections.pop(connection, None)
        connection.close()

    def stream(self, method, path, json=None, timeout=None):
        connection = self._open(self.timeout if timeout is None else timeout)
        try:
            body = None if json is None else dumps(json).encode()
            headers = {**self.headers, **({'Content-Type': 'application/json'} if body is not None else {})}
            connection.request(method, path, body=body, headers=headers)
            return ServerResponse(self, connection, connection.getresponse())
        except BaseException:
            self._release(connection)
            raise

    def post(self, path, json=None, timeout=None):
        with self.stream('POST', path, json, timeout) as response:
            response.read()
            return response

    def get(self, path, timeout=None):
        with self.stream('GET', path, None, timeout) as response:
            response.read()
            return response

    def close(self):
        with self.lock:
            self.closed = True
            sockets = list(self.connections.values())
        for sock in sockets:  # each request's own thread closes its connection
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass


class LlamaServerProvider(LlamaContextProvider):
    transport = 'llama-server'
    missing_route_hint = ' Update llama-server to a recent llama.cpp build; Riko needs /v1/responses, /apply-template, /tokenize and /slots.'

    def __init__(self, config):
        super().__init__(config)
        self.root = server_address(config.base_url)

    def _client(self): return ServerClient(self.root, self.config.api_key, self.config.request_timeout_seconds)

    def _required_context(self):
        return max(self.config.n_ctx, self.config.initiative_n_ctx, self.config.reflection_n_ctx)

    def launch_hint(self):
        slots, port = self.config.parallel_slots, urlsplit(self.root).port or 8080
        return (f'Start it with: llama-server -m <model.gguf> --port {port} --parallel {slots} '
                f'--ctx-size {slots * self._required_context()} --jinja (or change runtime.parallel_slots and runtime.n_ctx to match the server).')

    def _unreachable(self, error):
        # Check /health and /slots again on the next request: the server may come back with other settings.
        self.client = None
        return RuntimeError(f'No llama-server is answering at {self.root} ({error}). {self.launch_hint()}')

    def _wait_until_ready(self, client):
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while True:
            if self.closed: raise RuntimeError('llama-server provider closed')
            try: status = client.get('/health', timeout=CONNECT_TIMEOUT).status_code
            except Unreachable as exc: raise self._unreachable(exc) from None
            except (OSError, http.client.HTTPException): status = None
            if status == 200: return
            if status in {401, 403}: raise RuntimeError(f'llama-server at {self.root} refused runtime.api_key ({status}).')
            if status not in {None, 503}: raise RuntimeError(f'{self.root} answered /health with {status}; is runtime.base_url a llama-server?')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'llama-server at {self.root} was still loading after runtime.startup_timeout_seconds')
            time.sleep(.25)  # 503 while the server loads its model

    def _start(self):
        with self.start_lock:
            if self.closed: raise RuntimeError('llama-server provider closed')
            if self.client: return
            client = self._client()
            self._wait_until_ready(client)
            try: response = client.get('/slots')
            except Unreachable as exc: raise self._unreachable(exc) from None
            if response.status_code in {404, 501}:
                raise RuntimeError(f'llama-server at {self.root} does not expose /slots; start it without --no-slots. {self.launch_hint()}')
            self._check_response(response)
            slots, required = response.json(), self._required_context()
            if not isinstance(slots, list) or len(slots) != self.config.parallel_slots or any(
                    not isinstance(slot, dict) or type(slot.get('n_ctx')) is not int or slot['n_ctx'] < required for slot in slots):
                found = ', '.join(str(slot.get('n_ctx')) if isinstance(slot, dict) else '?' for slot in slots) if isinstance(slots, list) else 'none'
                raise RuntimeError(f'llama-server at {self.root} has slots with [{found}] tokens of context; Riko needs '
                    f'{self.config.parallel_slots} slots with at least {required} each. {self.launch_hint()}')
            self.client = client
            logger.info('Inference transport=llama_server url=%s slots=%s per_slot_context=%s', self.root, len(slots), min(slot['n_ctx'] for slot in slots))

    def _inference_client(self): return self._client()

    def _generate(self, role, messages, **options):
        try: return super()._generate(role, messages, **options)
        except Unreachable as exc: raise self._unreachable(exc) from None

    def count_tokens(self, messages):
        try: return super().count_tokens(messages)
        except Unreachable as exc: raise self._unreachable(exc) from None

    def count_text_tokens(self, text):
        try: return super().count_text_tokens(text)
        except Unreachable as exc: raise self._unreachable(exc) from None
