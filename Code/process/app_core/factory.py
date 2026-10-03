from __future__ import annotations

from .configuration.config import AppConfig
from .inference.providers import create_provider
from .tools.registry import ToolRegistry
from .conversation.chat import ChatService
from .emotion import JuliaEmotionEngine, ExpressionActionBridge
from .desktop.state import get_desktop_state
from .persistence.memory import MemoryStore
from .runtime.actions import ActionController
from .persistence.tasks import TaskStore, TaskMCP, TASK_RULES
from .desktop.media import resolve_media
from .desktop.effects import EffectLibrary
from contextlib import ExitStack
from .runtime.lifecycle import close_bounded


def create_chat_service(config: AppConfig) -> ChatService:
    """Build the complete application core without importing audio or UI code."""
    with ExitStack() as cleanup:
        service = _build_chat_service(config, cleanup)
        service.close = cleanup.pop_all().close
        return service


def _build_chat_service(config, cleanup):
    def own(resource):
        cleanup.callback(close_bounded, resource)
        return resource
    emotion_engine = None
    actions = own(ActionController())
    get_desktop_state().action_controller = actions
    get_desktop_state().media_resolver = lambda path: resolve_media(config.root, path, config.raw.get('desktop', {}).get('effects_directory', 'effects/greenscreens'))
    get_desktop_state().effects_directory = config.raw.get('desktop', {}).get('effects_directory', 'effects/greenscreens')
    get_desktop_state().effect_library = EffectLibrary(config.root / get_desktop_state().effects_directory)
    if config.emotion.enabled:
        state = get_desktop_state()
        def on_emotion(emotion):
            state.set_emotion(emotion)
            actions.set_emotion(emotion)
        bridge = ExpressionActionBridge(on_emotion=on_emotion)
        emotion_engine = own(JuliaEmotionEngine(
            config.emotion.model_path,
            model_id=config.emotion.model_id,
            cache_dir=config.emotion.cache_dir,
            device=config.emotion.device,
            strict_encoding=config.emotion.strict_encoding,
            max_length=config.emotion.max_length,
            context_tokens=config.emotion.context_tokens,
            update_interval_tokens=config.emotion.update_interval_tokens,
            temperature=config.emotion.temperature,
            fallback=config.emotion.fallback,
            on_event=lambda event: publish_teacher(event),
        ))
    provider = own(create_provider(config.runtime))
    def publish_teacher(event):
        probe = getattr(provider, 'probe', None)
        if probe: probe.publish_teacher(event, bridge.update)
        else: bridge.update(event.state)
    if config.emotion.probe.get('enabled'):
        from .emotion.probe import EmotionProbe, ProbeConfig
        if not hasattr(provider, 'probe_factory'):
            raise ValueError('Selected provider does not expose hidden-state probe capture')
        probe_config = ProbeConfig.from_raw(config.emotion.probe)
        provider.set_probe_interval(probe_config.interval_tokens)
        provider.probe_factory = lambda identity, idle: EmotionProbe(
            config.root / 'persistent_memories' / 'emotion_probes', identity, emotion_engine,
            probe_config, idle=idle, on_prediction=bridge.update, on_fallback=bridge.update)
        # Julia remains the teacher and low-confidence fallback. Student
        # events use the same expression/action bridge, never tool execution.
    if config.runtime.warmup and hasattr(provider, 'warmup'): provider.warmup()
    task_path = config.raw.get('tasks', {}).get('store_file', 'persistent_memories/tasks.sqlite3')
    task_store = TaskStore(config.root / task_path)
    registry = own(ToolRegistry.from_config(config))
    from .tools.choices import ChoiceResolver
    registry.choice_resolver = ChoiceResolver(emotion_engine.choose_tool_input if emotion_engine else None,
        enabled=config.tools.best_fit_inputs, timeout=config.tools.best_fit_timeout_seconds,
        confidence=config.tools.best_fit_min_confidence)
    task_mcp = TaskMCP(task_store)
    registry.register_mcp(task_mcp)
    memory = own(MemoryStore(config.memory, reflection_provider=getattr(provider, 'reflection', provider), start_worker=not config.runtime.warmup))
    if hasattr(provider, 'count_text_tokens'): memory.token_counter = provider.count_text_tokens
    if config.runtime.warmup:
        from .runtime.warmup import warm_core
        warm_core(memory, emotion_engine, config.runtime.startup_timeout_seconds)
        memory.start()
    service = ChatService(
        provider,
        system_prompt=config.system_prompt + '\n' + TASK_RULES + '\n' + str(config.raw.get('tasks', {}).get('rules', '')),
        character_name=config.character_name,
        tool_registry=registry,
        history_file=config.memory.history_file,
        emotion_engine=emotion_engine,
        memory_store=memory,
    )
    service.action_controller = actions
    service.task_store = task_store
    service.task_mcp = task_mcp
    service.initiative_provider = getattr(provider, 'initiative', provider)
    service.context_limit = min(config.runtime.n_ctx, config.memory.context_window_tokens + config.runtime.max_output_tokens)
    if service.emotion_worker: own(service.emotion_worker)
    return service
