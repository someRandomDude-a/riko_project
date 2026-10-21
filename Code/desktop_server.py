"""Local HTTP/WebSocket bridge for the Electron desktop client: create_app builds the FastAPI app for a loaded config.

Importing this module builds nothing (no config, services or helpers): run_server.main passes the config it loaded to
create_app, whose lifespan builds the services once uvicorn starts. The routes live in app_core/http/ (one router per URL
domain) and read a Backend from app.state. `desktop_server.app` (uvicorn desktop_server:app) and the module-level
stop_turn/api_secrets/api_token of earlier releases still work for one release: the first use builds one app from
load_config(recover='setup')."""
from process.app_core.factory import create_chat_service, create_desktop_state
from process.app_core.http.app import create_app as assemble


def create_app(config, services_factory=create_chat_service, **factories):
    """The backend's app for config (an AppConfig; run_server passes the one it loaded with recover='setup'). The lifespan
    builds chat = services_factory(config, desktop_state=...) over this app's desktop state (factory.create_desktop_state,
    which reads nothing yet); factories (session_factory, initiative_factory, bus) replace SessionManager, Initiative and
    the event bus, for tests. app.state.backend (app_core/http/backend.py) holds what the routes share."""
    return assemble(config, services_factory, create_desktop_state(config.paths), **factories)


def __getattr__(name):
    if name not in ('app', 'stop_turn', 'api_secrets', 'api_token'): raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    if 'app' not in globals():
        from process.app_core.configuration.config import load_config
        globals()['app'] = create_app(load_config(recover='setup'))  # a broken section leaves Settings up to repair it (lifespan)
    app = globals()['app']
    return app if name == 'app' else getattr(app.state.backend, name)
