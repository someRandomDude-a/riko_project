"""One managed model, reserved live slot and priority-ordered background leases."""
from contextlib import contextmanager
import logging
import queue
import shutil
import socket
import subprocess
import threading
import time

from .llama_runtime import resolve_model, validate_runtime


class BackgroundPreempted(RuntimeError): pass


class SlotScheduler:
    def __init__(self, count, *, pause_background=False):
        self.condition = threading.Condition()
        self.count = count
        self.active = {}
        self.waiting = []
        self.sequence = 0
        self.closed = False
        self.pause_background = pause_background
        self.foreground = False

    def set_foreground(self, active):
        with self.condition:
            self.foreground = active
            if active and self.pause_background:
                for role, stop in self.active.values():
                    if role != 'live': stop.set()
            self.condition.notify_all()

    def set_pause_background(self, enabled):
        with self.condition:
            self.pause_background = enabled
            self.set_foreground(self.foreground)

    @contextmanager
    def lease(self, role, cancelled=lambda: False):
        priority = {'live': 0, 'initiative': 1, 'reflection': 2}[role]
        with self.condition:
            ticket = (priority, self.sequence)
            self.sequence += 1
            self.waiting.append(ticket)
            try:
                while True:
                    if self.closed or cancelled(): raise BackgroundPreempted('Inference cancelled while queued')
                    candidates = [0] if role == 'live' else list(range(1, self.count))
                    eligible = [t for t in self.waiting if (t[0] == 0) == (priority == 0)]
                    free = next((s for s in candidates if s not in self.active), None)
                    live_demand = self.foreground or 0 in self.active or any(t[0] == 0 for t in self.waiting)
                    if self.pause_background and live_demand:
                        for active_role, active_stop in self.active.values():
                            if active_role != 'live': active_stop.set()
                        if role != 'live' or any(s != 0 for s in self.active):
                            self.condition.wait(.05)
                            continue
                    if ticket == min(eligible) and free is not None:
                        stop = threading.Event()
                        self.active[free] = (role, stop)
                        break
                    if role == 'initiative' and free is None:
                        victim = next((v for s, v in self.active.items() if s != 0 and v[0] == 'reflection'), None)
                        if victim: victim[1].set()
                    self.condition.wait(.05)
            finally: self.waiting.remove(ticket)
        try: yield free, stop
        finally:
            with self.condition:
                self.active.pop(free, None)
                self.condition.notify_all()

    def cancel_live(self):
        with self.condition:
            if 0 in self.active: self.active[0][1].set()

    def close(self):
        with self.condition:
            self.closed = True
            for _, stop in self.active.values(): stop.set()
            self.condition.notify_all()


def context_capacity(config):
    from .kv_budget import pool_capacity
    return pool_capacity(config)


def server_arguments(config, model, port):
    if not config.n_ctx: raise ValueError('Managed llama-server requires explicit per-slot runtime.n_ctx > 0')
    args = [config.server_path, '--model', str(model), '--host', '127.0.0.1', '--port', str(port),
        '--parallel', str(config.parallel_slots), '--ctx-size', str(context_capacity(config)),
        '--kv-unified' if config.kv_unified else '--no-kv-unified', '--cont-batching', '--jinja', '--slots', '--no-context-shift',
        '--n-gpu-layers', str(config.n_gpu_layers), '--batch-size', str(config.n_batch),
        '--ubatch-size', str(config.n_ubatch), '--flash-attn', 'on' if config.flash_attn else 'off',
        '--cache-type-k', config.type_k, '--cache-type-v', config.type_v,
        '--main-gpu', str(config.main_gpu), '--split-mode', config.split_mode,
        '--cache-ram', str(config.cache_size_mb), '--seed', str(config.seed)]
    for key, flag in [('n_threads', '--threads'), ('n_threads_batch', '--threads-batch')]:
        if getattr(config, key) is not None: args.extend([flag, str(getattr(config, key))])
    if config.tensor_split: args.extend(['--tensor-split', ','.join(map(str, config.tensor_split))])
    if not config.offload_kqv: args.append('--no-kv-offload')
    mode = 'mmap+mlock' if config.use_mmap and config.use_mlock else 'mmap' if config.use_mmap else 'mlock' if config.use_mlock else 'none'
    args.extend(['--load-mode', mode])
    if config.chat_format: args.extend(['--chat-template', config.chat_format])
    return args


class ServerLane:
    def __init__(self, owner, role): self.owner, self.role = owner, role
    def generate(self, messages, *, tools=None, **options):
        return self.owner._generate(self.role, messages, tools=tools, **options)
    def count_tokens(self, messages): return self.owner.count_tokens(messages)


class LlamaServerProvider(ServerLane):
    def __init__(self, config):
        validate_runtime(config)
        context_capacity(config)
        if not config.n_ctx: raise ValueError('Managed llama-server requires explicit per-slot runtime.n_ctx > 0')
        super().__init__(self, 'live')
        self.config = config
        self.scheduler = SlotScheduler(config.parallel_slots, pause_background=config.pause_background_on_live)
        self.start_lock = threading.Lock()
        self.process = None
        self.client = None
        self.closed = False
        self.log_tail = []
        self.initiative = ServerLane(self, 'initiative')
        self.reflection = ServerLane(self, 'reflection')
        self.reflection_parallelism = config.parallel_slots - 1

    def set_foreground(self, active): self.scheduler.set_foreground(active)

    def set_pause_background(self, enabled):
        self.config.pause_background_on_live = enabled
        self.scheduler.set_pause_background(enabled)

    def _start(self):
        with self.start_lock:
            if self.closed: raise RuntimeError('llama-server provider closed')
            if self.process and self.process.poll() is None: return
            if self.client: self.client.close()
            executable = shutil.which(self.config.server_path)
            if not executable: raise RuntimeError('Install llama-server and set runtime.server_path to its executable')
            model = resolve_model(self.config)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            args = server_arguments(self.config, model, port)
            args[0] = executable
            self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            process = self.process
            def logs():
                for line in process.stdout:
                    self.log_tail = (self.log_tail + [line.rstrip()])[-20:]
                    log = logging.getLogger(__name__)
                    (log.info if self.config.verbose else log.debug)('llama-server: %s', line.rstrip())
                process.stdout.close()
            threading.Thread(target=logs, daemon=True, name='llama-server-logs').start()
            import httpx
            self.client = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=self.config.request_timeout_seconds)
            deadline = time.monotonic() + self.config.startup_timeout_seconds
            try:
                while time.monotonic() < deadline:
                    if self.closed or process.poll() is not None: raise RuntimeError('llama-server startup failed: ' + '\n'.join(self.log_tail))
                    try:
                        if self.client.get('/health', timeout=1).status_code == 200:
                            slots = self.client.get('/slots').json()
                            required = max(self.config.n_ctx, getattr(self.config, 'initiative_n_ctx', 4096), getattr(self.config, 'reflection_n_ctx', 4096))
                            if len(slots) != self.config.parallel_slots or any(s['n_ctx'] < required for s in slots):
                                raise RuntimeError('Server did not allocate the requested per-slot context')
                            return
                    except httpx.TransportError: pass
                    time.sleep(.1)
                raise TimeoutError('llama-server startup deadline exceeded')
            except BaseException:
                if process.poll() is None: process.kill()
                process.wait(timeout=5)
                raise

    def _generate(self, role, messages, *, tools=None, **options):
        self._start()
        cancelled = options.get('cancelled', lambda: False)
        with self.scheduler.lease(role, cancelled) as (slot, stop):
            def check():
                if stop.is_set() or cancelled() or self.closed: raise BackgroundPreempted('Inference preempted/cancelled')
            check()
            from .responses import response_input, response_tools, template_messages, assemble_responses, sse_events
            ceiling = self.config.n_ctx if role == 'live' else getattr(self.config, role + '_n_ctx', 4096)
            limit = options.get('context_limit', ceiling)
            if limit > ceiling: raise ValueError(f'{role} context exceeds the server allocation ({ceiling}); restart Python after changing budgets')
            formatted = template_messages(messages)
            payload = dict(input=response_input(formatted), stream=True,
                temperature=options.get('temperature', self.config.temperature),
                max_output_tokens=options.get('max_output_tokens', self.config.max_output_tokens),
                id_slot=slot, cache_prompt=True)
            if tools: payload['tools'] = response_tools(tools)
            # A per-request transport lets cancellation close even a stalled
            # header read without affecting other slots' HTTP connections.
            with self._inference_client() as client:
                finished = threading.Event()
                def watch():
                    while not finished.wait(.05):
                        if stop.is_set() or cancelled() or self.closed:
                            try: client.close()
                            except Exception: pass
                            return
                threading.Thread(target=watch, daemon=True, name='slot-cancellation').start()
                try:
                    # Tokenizer/template preflight is not inference; it shares this
                    # request's cancellable transport, including before first token.
                    from .context_budget import pack_context
                    def count(value):
                        check()
                        rendered = client.post('/apply-template', json={'messages': [m.as_dict() for m in template_messages(value)], 'tools': tools or [], 'add_generation_prompt': True})
                        self._check_response(rendered)
                        tokenized = client.post('/tokenize', json={'content': rendered.json()['prompt'], 'add_special': True, 'parse_special': True})
                        self._check_response(tokenized)
                        return len(tokenized.json()['tokens'])
                    packed = pack_context(messages, count, limit, payload['max_output_tokens'], cancelled=lambda: stop.is_set() or cancelled() or self.closed)
                    payload['input'] = response_input(template_messages(packed))
                    with client.stream('POST', '/v1/responses', json=payload) as response:
                        self._check_response(response)
                        def chunks():
                            for event in sse_events(response.iter_lines()):
                                check()
                                yield event
                            check()
                        result = assemble_responses(chunks(), options.get('on_delta', lambda _: None), on_reasoning=options.get('on_reasoning'))
                        result.context_messages = packed
                        check()
                        return result
                except Exception:
                    check()
                    raise
                finally: finished.set()

    def _inference_client(self):
        import httpx
        return httpx.Client(base_url=self.client.base_url, timeout=self.config.request_timeout_seconds)

    @staticmethod
    def _check_response(response):
        if response.is_success: return
        response.read()
        try:
            body = response.json()
            error = body.get('error', body)
            detail = error.get('message', str(error)) if isinstance(error, dict) else str(error)
        except (ValueError, AttributeError): detail = response.text
        hint = ' Install a recent llama-server build with /v1/responses support.' if response.status_code in {404, 405, 501} else ''
        raise RuntimeError(f'llama-server Responses HTTP {response.status_code}: {str(detail).strip()[:2000]}{hint}')

    def stream(self, messages, *, tools=None, **options):
        output = queue.Queue(maxsize=128)
        stopped = threading.Event()
        cancelled = options.pop('cancelled', lambda: False)
        def emit(item):
            while not stopped.is_set():
                try: output.put(item, timeout=.1); return
                except queue.Full: pass
        def run():
            try: self.generate(messages, tools=tools, on_delta=emit,
                cancelled=lambda: stopped.is_set() or cancelled(), **options)
            except BaseException as exc: emit(exc)
            finally: emit(None)
        threading.Thread(target=run, daemon=True, name='live-stream').start()
        try:
            while True:
                item = output.get()
                if item is None: return
                if isinstance(item, BaseException): raise item
                yield item
        finally: stopped.set()

    def count_tokens(self, messages):
        self._start()
        from .responses import template_messages
        rendered = self.client.post('/apply-template', json={'messages': [m.as_dict() for m in template_messages(messages)], 'add_generation_prompt': True})
        self._check_response(rendered)
        response = self.client.post('/tokenize', json={'content': rendered.json()['prompt'], 'add_special': True, 'parse_special': True})
        self._check_response(response)
        return len(response.json()['tokens'])

    def count_text_tokens(self, text):
        self._start()
        response = self.client.post('/tokenize', json={'content': text, 'add_special': False, 'parse_special': True})
        response.raise_for_status()
        return len(response.json()['tokens'])

    def warmup(self):
        self._start() # Native llama-server performs model warmup at load.
        for slot in range(self.config.parallel_slots):
            response = self.client.post('/v1/responses', json={'input': [{'role':'user','content':'Hello'}],
                'max_output_tokens': 1, 'id_slot': slot, 'cache_prompt': True})
            self._check_response(response)

    def cancel(self): self.scheduler.cancel_live()
    def close(self):
        self.closed = True
        self.scheduler.close()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=3)
        if self.client: self.client.close()
