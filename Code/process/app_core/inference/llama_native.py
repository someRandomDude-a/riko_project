"""In-process llama.cpp transport. No server process, listener or loopback I/O.

The private DLL reuses native server-context queues, Responses formatting and
chat/tool parsers. Transport-compatible response objects keep the application
streaming/cancellation contract unchanged. All callbacks enqueue bytes only.
"""
import atexit
import ctypes
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import queue
import re
import threading
import time

from .llama_context import InferenceLane, LlamaContextProvider, context_capacity, fingerprint, native_arguments
from .llama_runtime import check_native_build, flash_attention, resolve_model, validate_runtime
from ..kernel.lifecycle import close_bounded

logger = logging.getLogger(__name__)
OUTPUT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
CANCEL = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
# Every NativeRuntime from riko_create until riko_destroy returns; a None handle is still loading. ggml-metal keeps
# its devices in a C++ static whose destructor, run by exit(), asserts that every Metal buffer was already freed.
_LIVE = set()
# riko_devices runs on telemetry threads: at exit, no call may still be in flight when exit() runs those destructors.
_DEVICES_LOCK, _EXITING = threading.Lock(), False
DEVICE_TYPES = {'cpu', 'gpu', 'igpu', 'accel', 'other'}


def release_native(timeout):
    """Destroy every live native context. False when one is still loading or does not stop in time."""
    runtimes = list(_LIVE)
    if any(runtime.handle is None for runtime in runtimes): return False  # riko_create cannot be interrupted
    deadline = time.monotonic() + timeout
    for runtime in runtimes: close_bounded(runtime, max(0, deadline - time.monotonic()))
    while _LIVE and time.monotonic() < deadline: time.sleep(.02)
    return not _LIVE


@atexit.register
def _release_at_exit():
    # Python's atexit runs before C++ static destructors. A context that cannot be destroyed in time is left to the
    # OS instead: os._exit skips those destructors, so Metal cannot turn the exit into a GGML_ASSERT abort.
    global _EXITING
    locked = _DEVICES_LOCK.acquire(timeout=2)  # a device query finishes in microseconds; never hang the exit on one
    _EXITING = True
    if locked: _DEVICES_LOCK.release()
    if not _LIVE: return
    try:
        import signal
        if threading.current_thread() is threading.main_thread(): signal.signal(signal.SIGINT, signal.SIG_IGN)  # a late Ctrl+C must not skip the fallback
        if release_native(2): return
    except BaseException: pass
    logger.error('Native inference context still loading or in use at exit; leaving it to the OS')
    logging.shutdown()
    os._exit(1)


class NativeRuntime:
    devices_function, devices_warned = None, False  # riko_devices, when the bridge has it (_bind)

    def __init__(self, library, arguments, interval=0):
        path = Path(library).resolve()
        self.lock, self.requests, self.closed, self.handle = threading.Lock(), set(), False, None
        self.directory = os.add_dll_directory(str(path.parent)) if os.name == 'nt' else None
        _LIVE.add(self)
        try:
            self.dll = ctypes.CDLL(str(path))
            self._bind()
            error = ctypes.create_string_buffer(8192)
            self.handle = self.dll.riko_create(json.dumps(arguments).encode(), interval, error, len(error))
            if not self.handle: raise RuntimeError(error.value.decode('utf-8', errors='replace'))
            # On success the bridge reports layer offload and flash attention here (older builds leave it empty).
            self.notes = error.value.decode('utf-8', errors='replace')
        except BaseException:
            _LIVE.discard(self)
            if self.directory: self.directory.close()
            raise

    def _bind(self):
        self.dll.riko_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        self.dll.riko_create.restype = ctypes.c_void_p
        self.dll.riko_request.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, OUTPUT, CANCEL, ctypes.c_void_p]
        self.dll.riko_request.restype = ctypes.c_int
        self.dll.riko_set_interval.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.riko_set_interval.restype = ctypes.c_int
        self.dll.riko_stop.argtypes = self.dll.riko_destroy.argtypes = [ctypes.c_void_p]
        self.dll.riko_stop.restype = self.dll.riko_destroy.restype = None
        # Optional: a bridge built before riko_devices lacks it (ctypes raises AttributeError), and telemetry falls back.
        self.devices_function = getattr(self.dll, 'riko_devices', None)
        if self.devices_function is not None:
            self.devices_function.argtypes, self.devices_function.restype = [ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int

    def devices(self):
        """ggml's devices with their memory in bytes (riko_devices), or None: an older bridge, a closed runtime, the process
        exiting, or a failure (logged once). Metal's figures are this process's working set, not the whole GPU's."""
        function = self.devices_function
        if function is None: return None
        with _DEVICES_LOCK:
            if _EXITING or self.closed: return None
            buffer = ctypes.create_string_buffer(16384)
            length = function(buffer, len(buffer))
            if length >= len(buffer):  # the list outgrew the buffer (nothing was written): once more, with room for it
                buffer = ctypes.create_string_buffer(length + 256)
                length = function(buffer, len(buffer))
        text = buffer.value.decode('utf-8', errors='replace')
        if length == -2: return None  # before the first riko_create (the backends are not initialised)
        try:
            if length < 0: raise RuntimeError(text or f'riko_devices returned {length}')
            if length >= len(buffer): raise RuntimeError('riko_devices outgrew its buffer twice')
            data = json.loads(text)
            if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('devices'), list): raise ValueError('unknown riko_devices format')
            return [parse_device(device) for device in data['devices']]
        except (RuntimeError, ValueError) as exc:
            if not self.devices_warned: logger.warning('riko_devices failed; GPU telemetry falls back to other sources: %s', exc)
            self.devices_warned = True
            return None

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
        _LIVE.discard(self)  # before the handle clears: release_native reads a None handle as still loading
        self.handle = None
        if self.directory: self.directory.close()


def parse_device(device):
    """One riko_devices entry, checked: {index, name, description, backend, type, device_id (PCI bus id or None),
    memory_free, memory_total (bytes)}."""
    if not isinstance(device, dict): raise ValueError('riko_devices entry is not an object')
    for key in ('name', 'description', 'backend'):
        if not isinstance(device.get(key), str): raise ValueError(f'riko_devices entry without a text {key}')
    for key in ('index', 'memory_free', 'memory_total'):
        if type(device.get(key)) is not int or device[key] < 0: raise ValueError(f'riko_devices entry without a whole {key}')
    if device.get('type') not in DEVICE_TYPES: raise ValueError(f'riko_devices entry of unknown type {device.get("type")!r}')
    if device.get('device_id') is not None and not isinstance(device['device_id'], str): raise ValueError('riko_devices device_id is not text')
    return {key: device.get(key) for key in ('index', 'name', 'description', 'backend', 'type', 'device_id', 'memory_free', 'memory_total')}


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


def training_context(model):
    """The GGUF's {arch}.context_length: llama.cpp caps every slot's context there. None when unreadable."""
    from .gguf import read_gguf
    try: meta = read_gguf(model)
    except (OSError, ValueError, UnicodeDecodeError): return None
    value = meta.get(str(meta.get('general.architecture', '')) + '.context_length')
    return value if type(value) is int and value > 0 else None


class InProcessLlamaProvider(LlamaContextProvider):
    """llama.cpp in this process, and the ProbeHost of the emotion probe: the patched bridge captures hidden states on slot 0."""
    def __init__(self, config):
        validate_runtime(config)
        check_native_build(config)
        context_capacity(config)
        super().__init__(config)
        self.capabilities = replace(self.capabilities, latent_probe=True)
        self.native = None
        self.probe_interval = 32
        self.probe_hook = None  # the CaptureHook attach_probe installed
        self.probe_error = ''
        self.replay_lane = InferenceLane(self, 'probe_replay')

    @property
    def probe_host(self): return self

    @property
    def probe(self): return self.probe_hook.probe if self.probe_hook else None

    def attach_probe(self, hook, interval_tokens):
        """Capture for hook from the first load: the library arms the capture at riko_create, and hook.start runs once the
        model is up."""
        with self.start_lock:
            if self.native: raise RuntimeError('Attach the emotion probe before the model loads')
            self.probe_hook = hook
            self.observers.append(hook)
        self.set_probe_interval(interval_tokens)

    def probe_idle(self): return self.scheduler.idle() and self.expression_idle()

    def devices(self):
        """riko_devices' list, each device marked `model` (llama.cpp offloads this model to it; none with n_gpu_layers 0) and
        `main` (the one device with split_mode none, or the only model device); None before the model loads or with a bridge
        without it."""
        native = self.native
        devices = native.devices() if native else None
        if devices is None: return None
        from .devices import model_devices
        chosen = model_devices(devices, split_mode=self.config.split_mode, main_gpu=self.config.main_gpu) or [] if self.config.n_gpu_layers else []
        indices = {device['index'] for device in chosen}
        main = chosen[0]['index'] if len(chosen) == 1 else None
        return [{**device, 'model': device['index'] in indices, 'main': device['index'] == main} for device in devices]

    def _start(self):
        with self.start_lock:
            if self.closed: raise RuntimeError('Native provider closed')
            if self.native: return
            if not self.config.native_library:
                raise RuntimeError('Set runtime.native_library to a compatible riko-native library, or set runtime.provider: llama_server to use a llama-server you run.')
            if not Path(self.config.native_library).is_file():
                raise RuntimeError(f'runtime.native_library does not exist: {self.config.native_library}. Build or select a compatible riko-native library before loading the model.')
            model = resolve_model(self.config)
            budgets = {'runtime.n_ctx': self.config.n_ctx, 'initiative.context_window_tokens': self.config.initiative_n_ctx,
                'memory.reflection_context_window_tokens': self.config.reflection_n_ctx}
            limit = training_context(model)
            if limit and max(budgets.values()) > limit:
                over = ', '.join(f'{key} ({value})' for key, value in budgets.items() if value > limit)
                raise RuntimeError(f'This model was trained for {limit} tokens and llama.cpp caps each slot there. Lower {over} to {limit} or less.')
            args = native_arguments(self.config, model)
            interval = self.probe_interval if self.probe_hook else 0
            native = NativeRuntime(self.config.native_library, args, interval)
            notes = getattr(native, 'notes', '')
            logger.info('Inference transport=in_process requested_gpu_layers=%s threads=%s flash_attention=%s kv_k=%s kv_v=%s slots=%s probe_interval=%s native=%r',
                self.config.n_gpu_layers, self.config.n_threads, self.config.flash_attn, self.config.type_k, self.config.type_v,
                self.config.parallel_slots, interval, notes)
            for line in notes.splitlines():  # a GGML_BACKEND_DL bundle whose GPU module did not load or found no device
                if line.startswith('riko-native: '): logger.warning('%s', line.removeprefix('riko-native: '))
            offload = re.search(r'offloaded (\d+)/(\d+) layers', notes)
            if offload and self.config.n_gpu_layers == -1 and int(offload[1]) < int(offload[2]):
                logger.warning('llama.cpp put only %s of %s layers on the GPU to keep 1 GiB of GPU memory free; replies will be slower. '
                    'Free GPU memory, or set runtime.n_gpu_layers to -2 to require every layer on the GPU.', offload[1], offload[2])
            self.native, self.client = native, NativeClient(native, self.config.request_timeout_seconds)
            try:
                response = self.client.get('/slots')
                self._check_response(response)
                slots = response.json()
                required = max(budgets.values())
                if len(slots) != self.config.parallel_slots or any(s['n_ctx'] < required for s in slots):
                    raise RuntimeError(f'llama.cpp allocated {len(slots)} slots of {sorted({s["n_ctx"] for s in slots})} tokens, but Riko needs '
                        f'{self.config.parallel_slots} slots of {required} (the largest of {", ".join(budgets)}).')
            except BaseException:
                self.client.close()
                native.close()
                self.native = self.client = None
                raise
            # The probe is optional: chat works without expressions, so its failure (a stale library, torch or Julia
            # missing, an unreadable file) leaves the model up. /api/neural/status reports why; restart Python to retry.
            try: self._initialize_probe(model)
            except Exception as exc:
                self.probe_error = f'The emotion probe did not start: {exc}'
                logger.exception('Emotion probe did not start; chat continues without it')
                # Capture stays loaded until restart (the bridge has no off switch): sample as rarely as it allows.
                try: native.set_interval(512)
                except Exception: logger.exception('Could not slow the unused emotion capture')

    def _initialize_probe(self, model):
        hook = self.probe_hook
        if hook is None or hook.probe is not None: return
        response = self.client.get('/props')
        self._check_response(response)
        props = response.json()
        if props.get('riko_emotion_probe') != hook.feature_version:  # before hashing gigabytes for an identity
            raise RuntimeError(f'it needs a riko-native library that captures {hook.feature_version} hidden states, and this one reports '
                f'{props.get("riko_emotion_probe")!r}; rebuild it from this checkout (tools/llama_cpp/README.md)')
        # Every split shard, not just the first file. The keys and values stay as they are: they name existing probe data.
        split = re.fullmatch(r'(.*)-00001-of-(\d{5})\.gguf', model.name)
        files = [model.with_name(f'{split[1]}-{i:05d}-of-{split[2]}.gguf') for i in range(1, int(split[2]) + 1)] if split else [model]
        identity = {'gguf_sha256': fingerprint(files), 'server_build': props.get('build_info'),
            'chat_template': props.get('chat_template'), 'feature_version': hook.feature_version,
            'type_k': self.config.type_k, 'type_v': self.config.type_v,
            'flash_attn': self._flash_attention_identity(), 'n_ctx': self.config.n_ctx,
            'runtime_fingerprint': self._probe_runtime_fingerprint()}
        hook.start(identity, self.probe_idle)

    def _inference_client(self): return NativeClient(self.native, self.config.request_timeout_seconds)

    def _flash_attention_identity(self):
        # on/off keep the true/false of probe data recorded before flash_attn had an auto setting. With auto, record what
        # llama.cpp chose (the bridge reports it), so a config that never set flash_attn and resolves to off keeps its
        # earlier probe data, and captures with and without it never mix.
        requested = {'on': True, 'off': False}.get(flash_attention(self.config.flash_attn), 'auto')
        notes = getattr(self.native, 'notes', '')
        if requested != 'auto' or 'Flash Attention' not in notes: return requested
        return 'Flash Attention enabled' in notes

    def _probe_runtime_fingerprint(self):
        path = Path(self.config.native_library)
        return fingerprint(sorted({path, *path.parent.glob('*.dll'), *path.parent.glob('*.so*'), *path.parent.glob('*.dylib')}), names=True)

    def set_probe_interval(self, interval):
        if type(interval) is not int or not 1 <= interval <= 512: raise ValueError('Invalid probe interval')
        with self.start_lock:
            self.probe_interval = interval
            if self.native and self.probe_hook and not self.probe_error: self.native.set_interval(interval)  # a failed probe stays sparse
            probe = self.probe
            if probe: probe.config.interval_tokens = interval

    def close(self):
        self.closed = True
        self.scheduler.close()
        with self.start_lock:
            if self.client: self.client.close()
            if self.native: self.native.close()
            if self.probe_hook: self.probe_hook.close()
