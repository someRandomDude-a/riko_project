"""Native model request protocol, context budgeting and priority-ordered slots."""
from contextlib import contextmanager
import logging
import queue
import threading
import time
import hashlib
import uuid
import math

from .llama_runtime import validate_runtime
logger = logging.getLogger(__name__)


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


def native_arguments(config, model):
    if not config.n_ctx: raise ValueError('Native llama.cpp requires explicit per-slot runtime.n_ctx > 0')
    args = ['riko-native', '--model', str(model),
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


class InferenceLane:
    def __init__(self, owner, role): self.owner, self.role = owner, role
    def generate(self, messages, *, tools=None, **options):
        return self.owner._generate(self.role, messages, tools=tools, **options)
    def count_tokens(self, messages): return self.owner.count_tokens(messages)


class LlamaContextProvider(InferenceLane):
    supports_latent_probe = True
    def __init__(self, config):
        validate_runtime(config)
        context_capacity(config)
        if not config.n_ctx: raise ValueError('Native llama.cpp requires explicit per-slot runtime.n_ctx > 0')
        super().__init__(self, 'live')
        self.config = config
        self.scheduler = SlotScheduler(config.parallel_slots, pause_background=config.pause_background_on_live)
        self.start_lock = threading.Lock()
        self.client = None
        self.closed = False
        self.initiative = InferenceLane(self, 'initiative')
        self.reflection = InferenceLane(self, 'reflection')
        self.reflection_parallelism = config.parallel_slots - 1
        self.probe_factory = None
        self.probe = None

    def probe_idle(self):
        with self.scheduler.condition:
            available=not self.scheduler.foreground and not self.scheduler.active and not self.scheduler.waiting
        return available and getattr(self,'expression_idle',lambda:True)()

    def _initialize_probe(self, model):
        if not self.probe_factory or self.probe: return
        from ..emotion.probe import FEATURE_VERSION
        response = self.client.get('/props')
        self._check_response(response)
        props = response.json()
        if props.get('riko_emotion_probe') != FEATURE_VERSION:
            raise RuntimeError('Emotion probe requires a compatible riko-native library with hidden-state capture')
        digest = hashlib.sha256()
        # Include all split shards, not just the first file.
        files = [model]
        import re
        split = re.fullmatch(r'(.*)-00001-of-(\d{5})\.gguf', model.name)
        if split:
            files = [model.with_name(f'{split[1]}-{i:05d}-of-{split[2]}.gguf') for i in range(1, int(split[2]) + 1)]
        for path in files:
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
        identity = {'gguf_sha256': digest.hexdigest(), 'server_build': props.get('build_info'),
            'chat_template': props.get('chat_template'), 'feature_version': FEATURE_VERSION,
            'type_k': self.config.type_k, 'type_v': self.config.type_v,
            'flash_attn': self.config.flash_attn, 'n_ctx': self.config.n_ctx,
            'runtime_fingerprint': self._probe_runtime_fingerprint()}
        self.probe = self.probe_factory(identity, self.probe_idle)

    def _probe_runtime_fingerprint(self): return None

    def set_foreground(self, active): self.scheduler.set_foreground(active)

    def set_pause_background(self, enabled):
        self.config.pause_background_on_live = enabled
        self.scheduler.set_pause_background(enabled)

    def _start(self):
        raise NotImplementedError('Native transport must initialize the model context')

    def _generate(self, role, messages, *, tools=None, **options):
        queued_at = time.perf_counter()
        self._start()
        cancelled = options.get('cancelled', lambda: False)
        with self.scheduler.lease(role, cancelled) as (slot, stop):
            leased_at = time.perf_counter()
            def check():
                if stop.is_set() or cancelled() or self.closed: raise BackgroundPreempted('Inference preempted/cancelled')
            check()
            group = options.get('emotion_turn_id') or str(uuid.uuid4())
            if self.probe and role == 'live': self.probe.activate(group)
            from .responses import response_input, response_tools, template_messages, assemble_responses, sse_events
            ceiling = self.config.n_ctx if role == 'live' else getattr(self.config, role + '_n_ctx', 4096)
            limit = options.get('context_limit', ceiling)
            if limit > ceiling: raise ValueError(f'{role} context exceeds the server allocation ({ceiling}); restart Python after changing budgets')
            formatted = template_messages(messages)
            payload = dict(input=response_input(formatted), stream=True,
                temperature=options.get('temperature', self.config.temperature),
                max_output_tokens=options.get('max_output_tokens', self.config.max_output_tokens),
                id_slot=slot, cache_prompt=True, timings_per_token=True)
            if tools: payload['tools'] = response_tools(tools)
            # A per-request transport lets cancellation close even a stalled
            # first callback without affecting other slots' native requests.
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
                    logger.debug('Inference preflight provider=%s role=%s slot=%s wait_s=%.3f context_pack_s=%.3f messages=%s context_limit=%s',
                        type(self).__name__, role, slot, leased_at-queued_at, time.perf_counter()-leased_at, len(packed), limit)
                    payload['input'] = response_input(template_messages(packed))
                    with client.stream('POST', '/v1/responses', json=payload) as response:
                        self._check_response(response)
                        def chunks():
                            visible = ''
                            last_user = next((m.content for m in reversed(packed) if m.role == 'user'), '')
                            for event in sse_events(response.iter_lines()):
                                check()
                                if event.get('timings') and options.get('on_metrics'): options['on_metrics'](event['timings'])
                                if event.get('type') == 'response.output_text.delta': visible += event.get('delta') or ''
                                elif event.get('type') == 'riko.emotion_probe.sample' and self.probe and role == 'live':
                                    from ..emotion.probe import FEATURE_VERSION
                                    features = event.get('features')
                                    if (event.get('feature_version') == FEATURE_VERSION and visible
                                        and event.get('prefix_bytes') == len(visible.encode('utf-8'))
                                        and isinstance(features, list) and len(features) == 256
                                        and all(type(n) in (int, float) and math.isfinite(n) for n in features)):
                                        import torch
                                        from ..conversation.output_filter import clean_output
                                        self.probe.capture(torch.tensor(features, device='cpu'), f'user: {last_user}\nassistant: {clean_output(visible)}', group,
                                             cancelled=lambda: stop.is_set() or cancelled() or self.closed or self.probe.active_group != group,
                                             replay=bool(options.get('probe_replay')),input_text=last_user,offset=len(clean_output(visible)))
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
        raise NotImplementedError('Native transport must provide a request client')

    @staticmethod
    def _check_response(response):
        if response.is_success: return
        response.read()
        try:
            body = response.json()
            error = body.get('error', body)
            detail = error.get('message', str(error)) if isinstance(error, dict) else str(error)
        except (ValueError, AttributeError): detail = response.text
        hint = ' Rebuild the compatible riko-native library with Responses support.' if response.status_code in {404, 405, 501} else ''
        raise RuntimeError(f'Native llama.cpp operation {response.status_code}: {str(detail).strip()[:2000]}{hint}')

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
        self._start() # llama.cpp performs model warmup at load.
        for slot in range(self.config.parallel_slots):
            response = self.client.post('/v1/responses', json={'input': [{'role':'user','content':'Hello'}],
                'max_output_tokens': 1, 'id_slot': slot, 'cache_prompt': True})
            self._check_response(response)

    def cancel(self): self.scheduler.cancel_live()
    def close(self):
        self.closed = True
        self.scheduler.close()
        if self.probe: self.probe.close()
        if self.client: self.client.close()
