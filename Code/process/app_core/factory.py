from __future__ import annotations

from .configuration.config import AppConfig
from .inference.providers import create_provider
from .tools.registry import ToolRegistry
from .conversation.chat import ChatDeps, ChatService
from .emotion import JuliaEmotionEngine, ExpressionActionBridge
from .emotion.worker import EmotionWorker
from .desktop.activity import ActivityLog
from .desktop.effects import EffectsModel
from .desktop.geometry import SurfaceGeometry
from .desktop.listeners import DesktopEvents
from .desktop.presence import PlaybackFlags, Presence
from .desktop.state import DesktopState
from .desktop.tools import DesktopServices, iter_tools as desktop_tools
from .desktop.whiteboard import WhiteboardModel
from .persistence.memory import MemoryStore
from .runtime.actions import ActionController
from .persistence.tasks import TaskStore, TaskMCP, TASK_RULES
from .desktop.media import effects_directory, resolve_media
from .desktop.effects import EffectLibrary
from .persistence.atomic import atomic_write
from contextlib import ExitStack
from .kernel.lifecycle import close_bounded


def create_desktop_state(paths) -> DesktopState:
    """The desktop components the HTTP routes serve and the desktop tools act through, over the data root's whiteboard.json
    and desktop_settings.json (configuration/paths.DataPaths). Built before the model loads, reading nothing: the lifespan
    loads both files (WhiteboardModel.load, SurfaceGeometry.load_settings), and nothing is saved before that."""
    events = DesktopEvents()
    geometry = SurfaceGeometry(events, settings_file=paths.desktop_settings, write=atomic_write)
    return DesktopState(events=events, geometry=geometry, board=WhiteboardModel(events, geometry, board_file=paths.whiteboard),
        effects=EffectsModel(events), playback=PlaybackFlags(events), activity=ActivityLog(events), presence=Presence(events))


def create_chat_service(config: AppConfig, desktop_state: DesktopState | None = None) -> ChatService:
    """Build the complete application core without importing audio or UI code. The service's close() owns everything built
    here; a failure part way closes what was built so far. desktop_state: the one the app serves (default: a new
    create_desktop_state over config.paths, loading nothing)."""
    cleanup = ExitStack()
    try: return _build_chat_service(config, cleanup, desktop_state if desktop_state is not None else create_desktop_state(config.paths))
    except BaseException:
        cleanup.close()
        raise


def _build_chat_service(config, cleanup, desktop_state):
    def own(resource):
        cleanup.callback(close_bounded, resource)
        return resource
    emotion_engine = emotion_worker = None
    actions = own(ActionController())
    effects = effects_directory(config.raw)
    desktop = DesktopServices.of(desktop_state, actions=actions, effects_directory=effects,
        media_resolver=lambda path: resolve_media(config.root, path, effects), effect_library=EffectLibrary(config.root / effects))
    if config.emotion.enabled:
        presence = desktop_state.presence
        def on_emotion(emotion):
            presence.set_emotion(emotion)
            actions.set_emotion(emotion)
        bridge = ExpressionActionBridge(on_emotion=on_emotion)
        emotion_engine = own(JuliaEmotionEngine(
            config.emotion.model_path,
            model_id=config.emotion.model_id,
            revision=config.emotion.revision,
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
        emotion_worker = own(EmotionWorker(emotion_engine))
    provider = own(create_provider(config.runtime))
    hook = None  # the emotion probe's ProbeHook; its probe exists once the model has loaded
    def publish_teacher(event):
        probe = hook.probe if hook else None
        if probe: probe.publish_teacher(event, bridge.update)
        else: bridge.update(event.state)
    if config.emotion.probe.get('enabled'):
        from .emotion.probe import EmotionProbe, ProbeConfig
        from .emotion.probe_hook import ProbeHook
        from .emotion.probe_storage import probe_directory, training_directory
        host = provider.probe_host
        if host is None: raise ValueError('Selected provider does not expose hidden-state probe capture')
        probe_config = ProbeConfig.from_raw(config.emotion.probe)
        playback_active = emotion_worker.playback_active  # the probe needs emotion.enabled (load_config), so the worker exists
        def expression(state):  # while sentences play, they drive the expression: the probe's reading waits
            if not playback_active(): bridge.update(state)
        hook = ProbeHook(lambda identity, idle: EmotionProbe(
            probe_directory(config.paths, config.runtime), identity, emotion_engine,
            probe_config, idle=idle, legacy_directory=config.paths.emotion_probes,
            training_directory=training_directory(config.paths, config.runtime),
            on_prediction=expression, on_fallback=expression))
        host.attach_probe(hook, probe_config.interval_tokens)
        # Julia remains the teacher and low-confidence fallback. Student
        # events use the same expression/action bridge, never tool execution.
    if config.runtime.warmup: provider.warmup()
    task_store = TaskStore(config.paths.tasks)  # tasks.store_file (configuration/paths.py), as Code/task_mcp_server.py resolves it
    registry = own(ToolRegistry.from_config(config, activity=desktop_state.activity, local_tools={'desktop': desktop_tools(desktop)}))
    from .tools.choices import ChoiceResolver
    registry.choice_resolver = ChoiceResolver(emotion_engine.choose_tool_input if emotion_engine else None,
        enabled=config.tools.best_fit_inputs, timeout=config.tools.best_fit_timeout_seconds,
        confidence=config.tools.best_fit_min_confidence)
    task_mcp = TaskMCP(task_store)
    from .tools.tool import RIKO
    registry.register_mcp(task_mcp, source=RIKO, owner='tasks')  # Riko's own server: its read-only hints are trusted
    # One Julia-1 model for memory's System 1 and the emotion engine when they agree (emotion/teacher.py shares it per snapshot,
    # device and options): with strict encoding, the emotion engine's longer question head encodes memory's questions alike.
    shared = emotion_engine is not None and config.emotion.strict_encoding and config.emotion.max_length == config.memory.system1_max_length
    memory = own(MemoryStore(config.memory, reflection_provider=provider.lanes['reflection'], parallelism=provider.capabilities.background_parallelism,
        token_counter=provider.count_text_tokens, start_worker=not config.runtime.warmup, julia_options=emotion_engine.load_options if shared else None))
    if config.runtime.warmup:
        from .runtime.warmup import warm_core
        warm_core(memory, emotion_engine, config.runtime.startup_timeout_seconds)
        memory.start()
    return ChatService(
        provider,
        system_prompt=config.system_prompt + '\n' + TASK_RULES + '\n' + str(config.raw.get('tasks', {}).get('rules', '')),
        character_name=config.character_name,
        tool_registry=registry,
        history_file=config.paths.chat_history,
        emotion_engine=emotion_engine,
        memory_store=memory,
        deps=ChatDeps(action_controller=actions, task_store=task_store, task_mcp=task_mcp,
            initiative_provider=provider.lanes['initiative'],
            context_limit=min(config.runtime.n_ctx, config.memory.context_window_tokens + config.runtime.max_output_tokens),
            emotion_worker=emotion_worker, desktop=desktop, cleanup=cleanup),
    )
