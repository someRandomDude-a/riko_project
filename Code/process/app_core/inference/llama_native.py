"""In-process llama.cpp transport. No server process, listener or loopback I/O.

The private DLL reuses native server-context queues, Responses formatting and
chat/tool parsers. Transport-compatible response objects keep the application
streaming/cancellation contract unchanged. All callbacks enqueue bytes only.
"""
import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time

from .llama_server import LlamaServerProvider, server_arguments
from .llama_runtime import resolve_model

logger = logging.getLogger(__name__)
OUTPUT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
CANCEL = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class NativeRuntime:
    def __init__(self, library, arguments, interval=0):
        path = Path(library).resolve()
        self.directory = os.add_dll_directory(str(path.parent)) if os.name == 'nt' else None
        try:
            self.dll = ctypes.CDLL(str(path))
            self._bind()
            error = ctypes.create_string_buffer(4096)
            self.handle = self.dll.riko_create(json.dumps(arguments).encode(), interval, error, len(error))
            if not self.handle: raise RuntimeError(error.value.decode('utf-8', errors='replace'))
        except BaseException:
            if self.directory: self.directory.close()
            raise
        self.lock = threading.Lock()
        self.requests = set()
        self.closed = False

    def _bind(self):
        self.dll.riko_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        self.dll.riko_create.restype = ctypes.c_void_p
        self.dll.riko_request.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, OUTPUT, CANCEL, ctypes.c_void_p]
        self.dll.riko_request.restype = ctypes.c_int
        self.dll.riko_set_interval.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.riko_set_interval.restype = ctypes.c_int
        self.dll.riko_stop.argtypes = self.dll.riko_destroy.argtypes = [ctypes.c_void_p]
        self.dll.riko_stop.restype = self.dll.riko_destroy.restype = None

    def set_interval(self, interval):
        with self.lock:
            if self.closed: return
            if self.dll.riko_set_interval(self.handle, interval) != 0: raise ValueError('Invalid probe interval')

    def request(self, path, body, timeout=None):
        with self.lock:
            if self.closed: raise RuntimeError('Native runtime closed')
            response = NativeResponse(self, path, body, timeout)
            self.requests.add(response)
            try: response.thread.start()
            except BaseException:
                self.requests.discard(response)
                raise
            return response

    def close(self):
        with self.lock:
            if self.closed: return
            self.closed = True
            requests = list(self.requests)
            self.dll.riko_stop(self.handle)
        for request in requests: request.close()
        for request in requests: request.thread.join(timeout=3)
        if any(request.thread.is_alive() for request in requests):
            logger.warning('Native calls still stopping; deferring context destruction')
            threading.Thread(target=self._destroy_after, args=(requests,),
                name='llama-native-cleanup', daemon=True).start()
            return
        self._destroy_after(requests)

    def _destroy_after(self, requests):
        for request in requests: request.thread.join()
        self.dll.riko_destroy(self.handle)
        self.handle = None
        if self.directory: self.directory.close()


class NativeResponse:
    def __init__(self, runtime, path, body, timeout=None):
        self.runtime, self.path, self.body = runtime, path, body
        self.pending = queue.Queue(maxsize=64)
        self.stop = threading.Event()
        self.status_code = 200
        self.content = b''
        self.exhausted = False
        self.done = threading.Event()
        self.error = None
        self.timeout = timeout
        self.thread = threading.Thread(target=self._run, name='llama-native-request', daemon=True)

    @property
    def is_success(self): return 200 <= self.status_code < 300

    @property
    def text(self): return self.content.decode('utf-8', errors='replace')

    def raise_for_status(self):
        if not self.is_success: raise RuntimeError(f'Native operation {self.status_code}: {self.text[:2000]}')

    def _put(self, value):
        while not self.stop.is_set():
            try: self.pending.put(value, timeout=.05); return 1
            except queue.Full: pass
        return 0

    def _run(self):
        @OUTPUT
        def output(status, pointer, size, _):
            try: return self._put((status, ctypes.string_at(pointer, size)))
            except Exception: self.stop.set(); return 0
        @CANCEL
        def cancelled(_): return int(self.stop.is_set())
        try:
            result = self.runtime.dll.riko_request(self.runtime.handle, self.path.encode(),
                json.dumps(self.body or {}).encode(), output, cancelled, None)
            if result != 0 and not self.stop.is_set():
                self.error = RuntimeError(f'Native operation failed ({result})')
        except Exception as exc:
            self.error = exc
        finally:
            self._put(None)
            self.done.set()
            with self.runtime.lock: self.runtime.requests.discard(self)

    def __enter__(self):
        try:
            first = self._next()
            if first is None:
                if self.error: raise self.error
                raise RuntimeError('Native operation ended before response headers')
            self.status_code, self.content = first
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *args): self.close()
    def close(self): self.stop.set()

    def _next(self):
        deadline = time.monotonic() + self.timeout if self.timeout is not None else None
        while True:
            if self.stop.is_set(): raise RuntimeError('Native request cancelled')
            try: return self.pending.get(timeout=.05)
            except queue.Empty:
                if self.done.is_set():
                    if self.error: raise self.error
                    return None
                if deadline is not None and time.monotonic() >= deadline:
                    self.close()
                    raise TimeoutError('Native request timed out waiting for data')

    def read(self):
        if self.exhausted: return self.content
        chunks = [self.content]
        while True:
            part = self._next()
            if part is None:
                if self.error and self.is_success: raise self.error
                self.exhausted = True
                break
            self.status_code, data = part
            chunks.append(data)
        self.content = b''.join(chunks)
        return self.content

    def json(self): return json.loads(self.content)

    def iter_lines(self):
        buffer = self.content
        while True:
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                yield line.rstrip(b'\r').decode('utf-8')
            part = self._next()
            if part is None:
                if self.error: raise self.error
                self.exhausted = True
                break
            status, data = part
            if status >= 400: raise RuntimeError(data.decode('utf-8', errors='replace'))
            buffer += data
        if buffer: yield buffer.decode('utf-8')


class NativeClient:
    def __init__(self, runtime, timeout=None):
        self.runtime, self.responses, self.closed = runtime, set(), False
        self.timeout, self.lock = timeout, threading.Lock()
    def __enter__(self): return self
    def __exit__(self, *args):
        responses = self.close()
        for response in responses: response.thread.join(timeout=3)
    def close(self):
        with self.lock:
            self.closed = True
            responses = list(self.responses)
            for response in responses: response.close()
            return responses
    def stream(self, method, path, json=None):
        with self.lock:
            if self.closed: raise RuntimeError('Native request cancelled')
            self.responses = {response for response in self.responses if not response.done.is_set()}
            response = self.runtime.request(path, json, self.timeout)
            self.responses.add(response)
            return response
    def post(self, path, json=None, **kwargs):
        response = self.stream('POST', path, json)
        try:
            with response:
                response.read()
                return response
        finally:
            with self.lock: self.responses.discard(response)
    def get(self, path, **kwargs): return self.post(path)


class InProcessLlamaProvider(LlamaServerProvider):
    def __init__(self, config):
        super().__init__(config)
        self.native = None
        self.probe_interval = 32

    def _start(self):
        with self.start_lock:
            if self.closed: raise RuntimeError('Native provider closed')
            if self.native: return
            model = resolve_model(self.config)
            args = server_arguments(self.config, model, 0)
            args[0] = 'riko-native'
            native = NativeRuntime(self.config.native_library, args, self.probe_interval if self.probe_factory else 0)
            self.native, self.client = native, NativeClient(native, self.config.request_timeout_seconds)
            try:
                response = self.client.get('/slots')
                self._check_response(response)
                slots = response.json()
                required = max(self.config.n_ctx, self.config.initiative_n_ctx, self.config.reflection_n_ctx)
                if len(slots) != self.config.parallel_slots or any(s['n_ctx'] < required for s in slots):
                    raise RuntimeError('Native context did not allocate the requested per-slot context')
                self._initialize_probe(model)
            except BaseException:
                self.client.close()
                native.close()
                self.native = self.client = None
                raise

    def _inference_client(self): return NativeClient(self.native, self.config.request_timeout_seconds)

    def _probe_runtime_fingerprint(self):
        path = Path(self.config.native_library)
        files = sorted({path, *path.parent.glob('*.dll'), *path.parent.glob('*.so*'), *path.parent.glob('*.dylib')})
        digest = hashlib.sha256()
        for library in files:
            digest.update(library.name.encode())
            with library.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
        return digest.hexdigest()

    def set_probe_interval(self, interval):
        if type(interval) is not int or not 1 <= interval <= 512: raise ValueError('Invalid probe interval')
        with self.start_lock:
            self.probe_interval = interval
            if self.native and self.probe_factory: self.native.set_interval(interval)
            if self.probe: self.probe.config.interval_tokens = interval

    def close(self):
        self.closed = True
        self.scheduler.close()
        with self.start_lock:
            if self.client: self.client.close()
            if self.native: self.native.close()
            if self.probe: self.probe.close()
