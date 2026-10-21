"""What one app's routes and lifespan share (app.state.backend): the config and desktop state it serves, the services the
lifespan builds, and the backend's own helpers. Routes take it through the `Services` dependency, never module globals."""
import threading
from typing import Annotated

from fastapi import Depends, HTTPException
from starlette.requests import HTTPConnection

from ..desktop.api_guard import Confirmations, issue_secrets
from ..desktop.media import effects_directory
from ..desktop.whiteboard_image import WhiteboardImages, board_revision
from ..events.bus import event_bus
from ..events.resources import ResourceUnavailable
from ..integrations.discord.launcher import DiscordLauncher
from ..kernel.lifecycle import run_bounded
from ..resources.gpu_memory import GPUMonitor


class Backend:
    def __init__(self, config, state, services_factory, *, session_factory, initiative_factory, bus=None):
        """config: the AppConfig run_server loaded (load_config(recover='setup')); state: the DesktopState every route, tool and
        the session share. The lifespan (app.py) builds chat = services_factory(config, desktop_state=state), then
        session_factory(config, chat, state, actions) and initiative_factory(session); until then, and in setup mode, chat
        and session are None and startup_error says why; bus: the event bus (default the process's). Building this reads no
        file and starts nothing."""
        self.config, self.state, self.bus = config, state, event_bus if bus is None else bus
        self.services_factory, self.session_factory, self.initiative_factory = services_factory, session_factory, initiative_factory
        self.chat = self.session = self.conversation_store = self.resource_events = self.task_file_events = None
        self.startup_error = ''
        self.discord_launcher = DiscordLauncher(config.paths, state.activity)
        self.discord_launcher.token = self.api_token  # the Discord worker it starts receives this start's token
        self.gpu_monitor, self.board_images, self.confirmations = GPUMonitor(), WhiteboardImages(), Confirmations()
        self.secrets, self.secrets_lock = None, threading.Lock()

    def api_secrets(self):
        """(token, confirmation key) for this start, minted once (desktop/api_guard.py). run_server mints them right after
        binding the port; tests mint them on first use."""
        with self.secrets_lock:
            if self.secrets is None: self.secrets = issue_secrets(self.config.paths)
            return self.secrets

    def api_token(self): return self.api_secrets()[0]

    def require_confirmation(self, action, changes, provided):
        """Security-sensitive changes need a single-use signature that Electron main makes with the
        confirmation key, which never leaves the machine's files, after the user approves a dialog."""
        if not changes or self.confirmations.consume(self.api_secrets()[1], action, changes, provided): return
        raise HTTPException(428, {'detail': 'Confirm this change in the Riko window', 'confirm': self.confirmations.challenge(action, changes)})

    def settings_store(self):
        from ..configuration.settings_store import SettingsStore
        # The config this backend started with, so Settings can say which saved settings still wait for a restart.
        return SettingsStore(self.config.paths.config_file, running=self.config)

    def effects_directory(self): return effects_directory(self.config.raw)

    def snapshot(self):
        session, current = self.session, self.state.snapshot()
        current['whiteboard_revision'] = board_revision(current)
        current['avatar'] = self.config.avatar
        current['runtime'] = session.runtime_snapshot()['runtime'] if session else {'generating': False, 'ready': False}
        animation = getattr(session, 'animation', None)
        current['animation'] = animation.status() if animation else {'enabled': False, 'error': getattr(session, 'animation_error', '')}
        current['startup_error'] = self.startup_error
        current['character_name'] = self.config.character_name
        current['session_id'] = self.conversation_store.session_id if self.conversation_store else None
        return current

    def publish_state(self, event_type, value):
        # Publish a complete state after every mutation so clients never need to
        # reconstruct nested whiteboard/tool objects from partial events.
        current = self.snapshot()
        self.bus.publish('state.snapshot', **current)
        if event_type.startswith('whiteboard') or event_type == 'surface_result': self.board_images.observe(current, self.bus)

    def record_action(self, event):
        if not event.type.startswith('action.'): return
        action = event.payload.get('action')
        if action: self.state.record_action(action)

    def stop_turn(self):
        """run_server calls this before uvicorn drains requests, so an in-flight chat returns now instead of generating
        through the drain. Bounded: a stuck cancel must not hold up the rest of shutdown."""
        session = self.session
        if session is not None and session.is_open: run_bounded(session.cancel, 1, 'SessionManager.cancel')

    def probe_host(self):
        """The provider's ProbeHost, or None (no chat service yet, or a provider without hidden-state capture)."""
        return self.chat.provider.probe_host if self.chat is not None else None

    def provider(self): return getattr(self.chat, 'provider', None)

    # Domain getters: the routes serve them and ResourceEvents publishes them (resource.<topic>). ResourceUnavailable means the
    # feature is off; the app answers it with 503.
    def tool_registry(self):
        registry = getattr(self.chat, 'tool_registry', None)
        if not registry or not getattr(registry, 'approvals', None): raise ResourceUnavailable('Tool approvals unavailable')
        return registry

    def approvals(self):
        registry = self.tool_registry()
        # Rules belong to (source, name) (tools/approval.py); policy and tools name them as the model calls them.
        return {**registry.approvals.snapshot(registry.tools), 'tools': [{'name': t.name, 'description': t.description, 'source': t.source} for t in list(registry.tools.values())]}

    def initiative(self): return self.session.initiative.snapshot()

    def animation_service(self):
        service = getattr(self.session, 'animation', None)
        if service is None: raise ResourceUnavailable('Animation service is disabled or unavailable')
        return service

    def animation(self):
        service = self.animation_service()
        return {**service.status(), 'entries': service.library.list()}

    def voice(self):
        session = self.session
        runtime = session.runtime_snapshot()['runtime']
        return {'listening': runtime['listening'], 'capture_running': session.status().capture_running,
                'microphone_status': runtime['microphone_status'], 'phase': runtime.get('voice_phase','stopped'),
                'latest_transcript': runtime['latest_transcript'], 'wake': runtime['wake']}

    def avatar_models(self):
        from ..desktop.avatar_models import AvatarModels
        return AvatarModels(self.config.root).listing()

    def tasks(self): return {'tasks': self.chat.deps.task_store.list(include_closed=True)}

    def resource_getters(self):
        return {'approvals': self.approvals, 'initiative': self.initiative, 'animation': self.animation, 'voice': self.voice,
            'avatar_models': self.avatar_models, 'discord': lambda: self.discord_launcher.status(), 'tasks': self.tasks}

    def resource_snapshot(self):
        """Every resource topic for a socket's resource.snapshot, or None before the runtime is ready (and in setup mode)."""
        events = self.resource_events
        return events.snapshot() if events else None


async def get_backend(connection: HTTPConnection) -> Backend:
    return connection.app.state.backend


Services = Annotated[Backend, Depends(get_backend)]  # a route's `backend: Services` parameter, for HTTP and WebSocket routes
