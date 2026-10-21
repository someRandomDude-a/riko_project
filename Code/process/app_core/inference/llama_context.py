"""llama.cpp request protocol, context budgeting and priority-ordered slots, shared by the
in-process native library and an external llama-server."""
from contextlib import contextmanager
import logging
import threading
import time
import hashlib
import uuid

from ..kernel.cancellation import BackgroundPreempted  # defined here before kernel/: old imports from this module still work
from .llama_runtime import flash_attention, validate_slots
from .provider import BaseProvider, Generation, ProviderCapabilities
logger = logging.getLogger(__name__)


PRIORITY = {'live': 0, 'initiative': 1, 'reflection': 2, 'probe_replay': 3}


class SlotScheduler:
    """Slot 0 serves the live lane; initiative and reflection share the others, initiative first and at most one at a time
    (kv_budget.suggested_pool sizes a unified KV pool for one: an overflow makes llama.cpp abort every slot, live included).
    probe_replay (EmotionProbe.replay) needs slot 0, where the patch captures, but only while no live request wants it:
    live demand preempts it whatever pause_background says, and cancel_live never reaches it."""
    def __init__(self, count, *, pause_background=False):
        self.condition = threading.Condition()
        self.count = count
        self.active = {}  # slot -> (role, stop)
        self.waiting = []
        self.sequence = 0
        self.closed = False
        self.pause_background = pause_background
        self.foreground = False

    def _live_demand(self):
        return self.foreground or self.active.get(0, ('',))[0] == 'live' or any(t[0] == 0 for t in self.waiting)

    def _preempt(self, live_demand):
        """Stop what live demand displaces: a replay on slot 0 always, every background request when pausing."""
        if not live_demand: return
        for role, stop in self.active.values():
            if role == 'probe_replay' or (self.pause_background and role != 'live'): stop.set()

    def set_foreground(self, active):
        with self.condition:
            self.foreground = active
            self._preempt(active)
            self.condition.notify_all()

    def set_pause_background(self, enabled):
        with self.condition:
            self.pause_background = enabled
            self.set_foreground(self.foreground)

    def idle(self):
        """Nothing running or queued and no foreground turn."""
        with self.condition: return not self.foreground and not self.active and not self.waiting

    def _grant(self, role, ticket, live_demand):
        """The slot this ticket takes now, or None."""
        initiative_running = any(active == 'initiative' for active, _ in self.active.values())
        if role == 'probe_replay':
            if live_demand or 0 in self.active or ticket != min(t for t in self.waiting if t[0] == PRIORITY[role]): return None
            return 0
        if role == 'live': peers, candidates = [t for t in self.waiting if t[0] == 0], [0]
        elif role == 'initiative' and initiative_running: return None
        else:
            # A queued initiative that must wait for the running one does not hold reflections back.
            peers = [t for t in self.waiting if t[0] in (1, 2) and not (t[0] == 1 and initiative_running)]
            candidates = range(1, self.count)
        if ticket != min(peers): return None
        return next((s for s in candidates if s not in self.active), None)

    @contextmanager
    def lease(self, role, cancelled=lambda: False):
        with self.condition:
            ticket = (PRIORITY[role], self.sequence)
            self.sequence += 1
            self.waiting.append(ticket)
            try:
                while True:
                    if self.closed or cancelled(): raise BackgroundPreempted('Inference cancelled while queued')
                    live_demand = self._live_demand()
                    self._preempt(live_demand)
                    if self.pause_background and live_demand and (role != 'live' or any(s != 0 for s in self.active)):
                        self.condition.wait(.05)
                        continue
                    free = self._grant(role, ticket, live_demand)
                    if free is not None:
                        stop = threading.Event()
                        self.active[free] = (role, stop)
                        break
                    if role == 'initiative' and not any(r == 'initiative' for r, _ in self.active.values()) and all(s in self.active for s in range(1, self.count)):
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
        """Stop the live request (Stop): never a background request, nor a replay holding slot 0."""
        with self.condition:
            role, stop = self.active.get(0, ('', None))
            if role == 'live': stop.set()

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
        # -1 lets llama.cpp's --fit lower the layer count to keep 1 GiB of GPU memory free; any other value is exact.
        '--n-gpu-layers', str(config.n_gpu_layers), '--fit', 'on' if config.n_gpu_layers == -1 else 'off',
        '--batch-size', str(config.n_batch), '--ubatch-size', str(config.n_ubatch), '--flash-attn', flash_attention(config.flash_attn),
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


_FINGERPRINTS, _FINGERPRINT_LOCK = {}, threading.Lock()


def fingerprint(paths, *, names=False):
    """SHA-256 of the files' bytes in order (each preceded by its name when names), cached for this process by every
    file's (path, size, mtime): an identity hashes a multi-GB GGUF once, and a changed file is hashed again."""
    paths = list(paths)
    def stamp(): return tuple((str(path), stat.st_size, stat.st_mtime_ns) for path in paths for stat in [path.stat()])
    key = (names, stamp())
    with _FINGERPRINT_LOCK: known = _FINGERPRINTS.get(key)
    if known: return known
    digest = hashlib.sha256()
    for path in paths:
        if names: digest.update(path.name.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    value = digest.hexdigest()
    if stamp() == key[1]:  # not rewritten while it was read
        with _FINGERPRINT_LOCK: _FINGERPRINTS[key] = value
    return value


class LlamaContextProvider(InferenceLane, BaseProvider):
    transport = 'Native llama.cpp'  # names the transport in errors
    missing_route_hint = ' Rebuild the compatible riko-native library with Responses support.'
    def __init__(self, config):
        validate_slots(config)
        if not config.n_ctx: raise ValueError('llama.cpp requires explicit per-slot runtime.n_ctx > 0')
        super().__init__(self, 'live')
        self.config = config
        self.capabilities = ProviderCapabilities(slots=config.parallel_slots, exact_tokens=True)
        self.scheduler = SlotScheduler(config.parallel_slots, pause_background=config.pause_background_on_live)
        self.start_lock = threading.Lock()
        self.client = None
        self.closed = False
        self.initiative = InferenceLane(self, 'initiative')
        self.reflection = InferenceLane(self, 'reflection')
        self.observers = []  # GenerationObservers, added before the first request

    @property
    def lanes(self): return {'live': self, 'initiative': self.initiative, 'reflection': self.reflection}

    def set_foreground(self, active): self.scheduler.set_foreground(active)

    def set_pause_background(self, enabled):
        self.config.pause_background_on_live = enabled
        self.scheduler.set_pause_background(enabled)

    def _start(self):
        raise NotImplementedError('Native transport must initialize the model context')

    def _notify(self, method, *args):
        for observer in self.observers:
            try: getattr(observer, method)(*args)
            except Exception: logger.exception('Generation observer %s failed in %s', type(observer).__name__, method)

    def _generate(self, role, messages, *, tools=None, **options):
        queued_at = time.perf_counter()
        self._start()
        cancelled = options.get('cancelled', lambda: False)
        with self.scheduler.lease(role, cancelled) as (slot, stop):
            leased_at = time.perf_counter()
            def stopped(): return stop.is_set() or cancelled() or self.closed
            def check():
                if stopped(): raise BackgroundPreempted('Inference preempted/cancelled')
            check()
            generation = Generation(role, slot, options.get('emotion_turn_id') or str(uuid.uuid4()), stopped)
            from .responses import response_input, response_tools, template_messages, assemble_responses, sse_events
            ceiling = {'live': self.config.n_ctx, 'probe_replay': self.config.n_ctx, 'initiative': self.config.initiative_n_ctx,
                'reflection': self.config.reflection_n_ctx}[role]
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
                        if stopped():
                            try: client.close()
                            except Exception: pass
                            return
                threading.Thread(target=watch, daemon=True, name='slot-cancellation').start()
                self._notify('on_start', generation)
                try:
                    # Tokenizer/template preflight is not inference; it shares this
                    # request's cancellable transport, including before first token.
                    from .context_budget import pack_context
                    probes = []
                    def count(value):
                        check()
                        probes.append(len(value))
                        rendered = client.post('/apply-template', json={'messages': [m.as_dict() for m in template_messages(value)], 'tools': tools or [], 'add_generation_prompt': True})
                        self._check_response(rendered)
                        tokenized = client.post('/tokenize', json={'content': rendered.json()['prompt'], 'add_special': True, 'parse_special': True})
                        self._check_response(tokenized)
                        return len(tokenized.json()['tokens'])
                    packed = pack_context(messages, count, limit, payload['max_output_tokens'], cancelled=stopped,
                        state=options.get('context_state'))
                    logger.debug('Inference preflight provider=%s role=%s slot=%s wait_s=%.3f context_pack_s=%.3f token_counts=%s counted_messages=%s messages=%s context_limit=%s',
                        type(self).__name__, role, slot, leased_at-queued_at, time.perf_counter()-leased_at, len(probes), sum(probes), len(packed), limit)
                    payload['input'] = response_input(template_messages(packed))
                    generation.messages = packed
                    with client.stream('POST', '/v1/responses', json=payload) as response:
                        self._check_response(response)
                        def chunks():
                            visible = ''
                            for event in sse_events(response.iter_lines()):
                                check()
                                if event.get('timings') and options.get('on_metrics'): options['on_metrics'](event['timings'])
                                if event.get('type') == 'response.output_text.delta': visible += event.get('delta') or ''
                                if self.observers: self._notify('on_event', generation, event, visible)  # e.g. the probe's samples
                                yield event
                            check()
                        result = assemble_responses(chunks(), options.get('on_delta', lambda _: None), on_reasoning=options.get('on_reasoning'))
                        result.context_messages = packed
                        check()
                        return result
                except Exception:
                    check()
                    raise
                finally:
                    finished.set()
                    self._notify('on_finish', generation)

    def _inference_client(self):
        raise NotImplementedError('Native transport must provide a request client')

    @classmethod
    def _check_response(cls, response):
        if response.is_success: return
        response.read()
        try:
            body = response.json()
            error = body.get('error', body)
            detail = error.get('message', str(error)) if isinstance(error, dict) else str(error)
        except (ValueError, AttributeError): detail = response.text
        hint = cls.missing_route_hint if response.status_code in {404, 405, 501} else ''
        raise RuntimeError(f'{cls.transport} operation {response.status_code}: {str(detail).strip()[:2000]}{hint}')

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
        if self.client: self.client.close()
