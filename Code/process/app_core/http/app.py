"""The FastAPI app over one Backend: the routers, the setup-mode guard, the token guard and the lifespan that builds and stops
the services. Code/desktop_server.create_app calls create_app with the composition root's factories."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ..desktop.api_guard import LocalAPIGuard
from ..events.resources import ResourceEvents, ResourceUnavailable
from ..integrations.discord.api import create_router as discord_router
from ..kernel.lifecycle import close_bounded
from ..persistence.conversation_store import ConversationStore
from ..runtime.initiative import Initiative
from ..runtime.session import SessionManager
from . import animation, approvals, avatar, chat, initiative, media, memories, neural, resources, settings, status, surfaces, tasks, voice, whiteboard
from .backend import Backend

logger = logging.getLogger(__name__)
ROUTERS = (neural, resources, approvals, chat, status, settings, media, whiteboard, avatar, animation, surfaces, tasks, initiative, memories, voice)
SETUP_ROUTES = ('/api/settings', '/api/resources', '/api/status', '/api/displays', '/api/media', '/api/avatar/model')  # up in setup mode


def create_app(config, services_factory, desktop_state, *, session_factory=SessionManager, initiative_factory=Initiative, bus=None):
    """The app for config (the AppConfig run_server loaded), serving desktop_state. Builds and starts nothing: the lifespan
    calls services_factory(config, desktop_state=...) once uvicorn starts. app.state.backend is the Backend every route reads."""
    dev = os.environ.get('RIKO_DEV') == '1'  # The API map is published only while developing.
    app = FastAPI(title="Riko Local Desktop Runtime", lifespan=lifespan,
        docs_url='/docs' if dev else None, redoc_url='/redoc' if dev else None, openapi_url='/openapi.json' if dev else None)
    backend = app.state.backend = Backend(config, desktop_state, services_factory, session_factory=session_factory,
        initiative_factory=initiative_factory, bus=bus)
    app.include_router(discord_router(lambda: backend.session, lambda: backend.discord_launcher, confirm=backend.require_confirmation,
        resources=backend.resource_snapshot, bus=lambda: backend.bus))
    for module in ROUTERS: app.include_router(module.router)
    app.add_exception_handler(ResourceUnavailable, lambda request, exc: JSONResponse({'detail': str(exc)}, status_code=503))
    app.middleware('http')(require_runtime)
    # Every route, HTTP and WebSocket, needs the install's API token and a loopback Host.
    app.add_middleware(LocalAPIGuard, token=backend.api_token)
    # Outermost: even setup-mode 503 responses need readable CORS headers.
    app.add_middleware(CORSMiddleware, allow_origins=["null", "http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
    return app


async def require_runtime(request, call_next):
    error = request.app.state.backend.startup_error
    if error and request.url.path.startswith('/api/') and not request.url.path.startswith(SETUP_ROUTES):
        return JSONResponse({'detail': 'Runtime unavailable. Fix Settings and restart Python.', 'error': error}, status_code=503)
    return await call_next(request)


@asynccontextmanager
async def lifespan(app):
    # Building the app must not start model/memory/audio workers: this does, once uvicorn starts.
    backend = app.state.backend
    config, state, bus = backend.config, backend.state, backend.bus
    backend.startup_error, backend.session = '', None
    try:
        # A section load_config could not read (recover='setup') would otherwise run with its defaults: repair it first.
        if config.load_errors: raise ValueError('; '.join(config.load_errors))
        chat = backend.chat = await run_in_threadpool(backend.services_factory, config, desktop_state=state)
        # load_config has checked the voice, speech and GPT-SoVITS values; whatever else fails building the session
        # (after the model has loaded) takes the same fallback, so Settings can repair it instead of every launch failing.
        try: backend.session = backend.session_factory(config, chat, state, chat.deps.action_controller)
        except BaseException:
            await run_in_threadpool(close_bounded, chat, 6)
            raise
    except Exception as exc:
        if not config.raw.get('desktop', {}).get('setup_on_startup_error', False): raise
        logger.exception('Backend unavailable; keeping settings available')
        backend.startup_error, backend.chat, backend.session, backend.conversation_store = str(exc), None, None, None
        unsubscribe = state.subscribe(backend.publish_state)
        try: yield
        finally: unsubscribe()
        return
    session, backend.conversation_store, unsubscribers = backend.session, None, []
    try:
        if config.runtime.warmup and hasattr(session, 'speech'):
            from ..runtime.warmup import warm_session
            await run_in_threadpool(warm_session, session)
        backend.conversation_store = ConversationStore(config.paths.conversations,
            provider=config.runtime.provider, legacy=chat.conversation.snapshot())
        unsubscribers.append(bus.subscribe(backend.conversation_store.observe))
        state.board.load()  # whiteboard.json; saves start once it has been read
        unsubscribers.append(state.subscribe(backend.publish_state))
        unsubscribers.append(bus.subscribe(backend.record_action))
        state.geometry.load_settings()  # the avatar's geometry from desktop_settings.json, published as a state.snapshot
        session.initiative = backend.initiative_factory(session)
        backend.resource_events = ResourceEvents(bus, backend.resource_getters())
        from ..persistence.task_file_events import TaskFileEvents
        if chat.deps.task_store: backend.task_file_events = TaskFileEvents(chat.deps.task_store, bus)
        bus.publish("runtime.ready", provider=config.runtime.provider)
        yield
    finally:
        # Electron allows the whole shutdown 15 s: run_server spends at most 1 s stopping the turn and 2 s draining
        # requests, this block at most about 8.5 s, and llama_native 2 s destroying the native context at exit.
        discord = asyncio.create_task(run_in_threadpool(backend.discord_launcher.stop))  # a separate process; stop it meanwhile
        if backend.task_file_events: backend.task_file_events.close(); backend.task_file_events = None
        if backend.resource_events: backend.resource_events.close(); backend.resource_events = None
        if backend.session: await run_in_threadpool(close_bounded, backend.session, 8)
        elif backend.chat: await run_in_threadpool(close_bounded, backend.chat, 6)
        await discord
        for unsubscribe in reversed(unsubscribers): unsubscribe()
        if backend.conversation_store: backend.conversation_store.close()
        bus.publish("runtime.stopped")
