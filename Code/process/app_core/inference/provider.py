"""The inference provider contract. Callers use these members directly, never hasattr/getattr: BaseProvider gives every
one a default, so a provider (or a test fake) implements only what it supports and declares the rest in capabilities.
The emotion probe's members are behind provider.probe_host (a ProbeHost, or None), and what a provider's generations
show its observers (emotion/probe_hook.py's ProbeHook is one) is a GenerationObserver."""
from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from ..kernel.messages import ChatMessage, ModelResponse

ROLES = ('live', 'initiative', 'reflection')  # the lanes: the foreground turn, initiative checks, memory reflection


@dataclass(frozen=True)
class ProviderCapabilities:
    slots: int = 0  # llama.cpp slots: slot 0 serves the live lane, the others initiative and reflection; 0 without slots
    exact_tokens: bool = False  # count_tokens and count_text_tokens use the model's own template and tokenizer
    latent_probe: bool = False  # hidden-state capture for the emotion probe: probe_host is not None

    @property
    def background_parallelism(self):
        """How many background requests (reflections) may run at once: one per background slot, one without slots."""
        return max(1, self.slots - 1)


class Lane(Protocol):
    """Generation on one role (provider.lanes[role]): the live lane is the provider itself."""
    def generate(self, messages: Sequence[ChatMessage], *, tools: list[dict] | None = None, **options) -> ModelResponse: ...
    def count_tokens(self, messages: Sequence[ChatMessage]) -> int: ...


class InferenceProvider(Lane, Protocol):
    capabilities: ProviderCapabilities
    lanes: Mapping[str, Lane]  # every role in ROLES
    probe_host: ProbeHost | None
    def stream(self, messages: Sequence[ChatMessage], *, tools: list[dict] | None = None, **options) -> Iterable[str]: ...
    def count_text_tokens(self, text: str) -> int: ...
    def warmup(self) -> None: ...  # load the model now rather than on the first request
    def cancel(self) -> None: ...  # stop the live request; never an initiative, reflection or replay request
    def set_foreground(self, active: bool) -> None: ...  # a foreground turn starts or ends
    def set_pause_background(self, enabled: bool) -> None: ...  # runtime.pause_background_on_live, applied live
    def set_expression_idle(self, idle: Callable[[], bool]) -> None: ...  # the session's quiet check (RuntimeStatus.is_quiet)
    def expression_idle(self) -> bool: ...
    def devices(self) -> list[dict] | None: ...  # ggml's devices with their memory (riko_devices), None when unknown
    def close(self) -> None: ...


ModelProvider = InferenceProvider  # the old name; app_core exports both


class BaseProvider:
    """Defaults for every InferenceProvider member except generate: one lane for every role, estimated token counts, no
    slots and no probe. cancel is a no-op on purpose: a request stops through its own `cancelled` option (polled while it
    streams, see providers.cancellable_stream), and initiative and reflection share the provider, so a provider-wide
    cancel would stop them too."""
    capabilities = ProviderCapabilities()
    probe_host = None
    _expression_idle = staticmethod(lambda: True)

    @property
    def lanes(self): return {role: self for role in ROLES}

    def generate(self, messages, *, tools=None, **options):
        raise NotImplementedError(f'{type(self).__name__} does not generate')

    def stream(self, messages, *, tools=None, **options):
        """Text deltas of one reply: generate on a helper thread with on_delta, ended early when the caller stops reading."""
        output, stopped = queue.Queue(maxsize=128), threading.Event()
        cancelled = options.pop('cancelled', lambda: False)
        def emit(item):
            while not stopped.is_set():
                try: output.put(item, timeout=.1); return
                except queue.Full: pass
        def run():
            try: self.generate(messages, tools=tools, on_delta=emit, cancelled=lambda: stopped.is_set() or cancelled(), **options)
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
        from .context_budget import estimate_tokens
        from .responses import template_messages
        return estimate_tokens(template_messages(messages))

    def count_text_tokens(self, text):
        from .context_budget import estimate_text_tokens
        return estimate_text_tokens(text)  # a calibrated estimate, not the server's tokenizer

    def warmup(self): pass
    def cancel(self): pass
    def set_foreground(self, active): pass
    def set_pause_background(self, enabled): pass
    def set_expression_idle(self, idle): self._expression_idle = idle
    def expression_idle(self): return self._expression_idle()
    def devices(self): return None  # only the in-process llama.cpp provider can ask ggml
    def close(self): pass


@dataclass
class Generation:
    """One request, as its GenerationObservers see it from its lease to its end."""
    role: str  # a ROLES entry, or 'probe_replay'
    slot: int | None  # the llama.cpp slot it runs on
    group: str  # options['emotion_turn_id'] (the turn it answers), or a fresh id
    cancelled: Callable[[], bool]  # preempted, cancelled or provider closed
    messages: list = field(default_factory=list)  # as packed and sent, set before the reply streams


class GenerationObserver(Protocol):
    """Watches a provider's generations: on_start once leased, on_event for each streamed event with the visible text
    so far, on_finish at the end, however it ended. Called on the request's thread and outside provider locks; a
    failure is logged and never fails the request."""
    def on_start(self, generation: Generation) -> None: ...
    def on_event(self, generation: Generation, event: dict, visible: str) -> None: ...
    def on_finish(self, generation: Generation) -> None: ...


class CaptureHook(GenerationObserver, Protocol):
    """What ProbeHost.attach_probe takes (emotion/probe_hook.py's ProbeHook): it parses the capture events and builds the
    probe once the model is loaded."""
    feature_version: str  # the capture format it parses, which the native library must report in /props
    probe: Any  # what start returned, or None
    def start(self, identity: dict, idle: Callable[[], bool]) -> Any: ...
    def close(self) -> None: ...


class ProbeHost(Protocol):
    """A provider's hidden-state capture for the emotion probe (provider.probe_host)."""
    probe: Any  # the running probe, or None
    probe_error: str  # why an attached probe is not running (its start failed; the model stays up), '' otherwise
    replay_lane: Lane  # where EmotionProbe.replay generates: the capture slot, preempted by any live request
    def attach_probe(self, hook: CaptureHook, interval_tokens: int) -> None: ...  # before the model loads
    def set_probe_interval(self, interval_tokens: int) -> None: ...
    def probe_idle(self) -> bool: ...  # no request running or queued and the session quiet
