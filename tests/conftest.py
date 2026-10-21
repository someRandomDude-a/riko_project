"""Shared test helpers: the desktop backend built by create_app over a test data root with stub services (the `backend`
fixture), a client that passes its API guard, and vrm_bytes. Import helpers with `from conftest import ...`."""
import json
import struct
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from process.app_core.configuration.paths import DataPaths
from process.app_core.conversation.chat import ChatDeps
from process.app_core.conversation.history import ConversationHistory
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections


def vrm_bytes(version='vrm1', uri=None):
    extension = {'specVersion':'1.0' if version == 'vrm1' else '0.0', 'humanoid':{'humanBones':{'hips':{'node':0}} if version == 'vrm1' else [{'bone':'hips','node':0}]}}
    document = {'asset':{'version':'2.0'}, 'extensions':{'VRMC_vrm' if version == 'vrm1' else 'VRM':extension}, 'nodes':[{}]}
    if uri: document['images'] = [{'uri':uri}]
    data = json.dumps(document).encode(); data += b' ' * (-len(data) % 4)
    return struct.pack('<4sII', b'glTF', 2, len(data)+20)+struct.pack('<II',len(data),0x4E4F534A)+data


def chat_stub(**fields):
    """create_chat_service's result as the lifespan and SessionManager use it: the history, the provider and the factory's collaborators."""
    return SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), **{'provider': BaseProvider(), **fields})


def client_for(backend, **options):
    """A client that passes the local API guard: loopback Host plus this test data root's token."""
    return TestClient(backend.app, base_url='http://127.0.0.1:8765', headers={'Authorization': 'Bearer ' + backend.api_token()}, **options)


def backend_config(root):
    """The config a backend test serves: a data root in tmp_path and the defaults the routes read."""
    return SimpleNamespace(root=root, paths=DataPaths.at(root), raw={}, avatar={}, character_name='Test character',
        runtime=SimpleNamespace(provider='fake', warmup=False), load_errors=(), **audio_sections({}))


def build_backend(config, **factories):
    """desktop_server.create_app(config) with a stub chat service unless given; returns app.state.backend, whose .app is the app.
    Nothing starts until a client enters the app (`with client_for(backend)`), which runs the lifespan."""
    from desktop_server import create_app
    app = create_app(config, **{'services_factory': lambda config, desktop_state: chat_stub(), **factories})
    backend = app.state.backend
    backend.app = app
    return backend


@pytest.fixture(autouse=True)
def no_julia_download(monkeypatch):
    """No test downloads Julia-1 (about 600 MB, with Python code that would run here): every load falls back as offline.
    tests/test_julia_teacher.py tests the real emotion/teacher.snapshot, which it keeps from before this patch."""
    from process.app_core.emotion import teacher
    def refuse(*args, **kwargs): raise RuntimeError('tests never download Julia-1')
    monkeypatch.setattr(teacher, 'snapshot', refuse)


@pytest.fixture
def backend(tmp_path):
    """A backend over tmp_path that has not started: the routes see a stub session until a test replaces it (backend.session)
    or enters the lifespan, which builds services_factory's chat, session_factory's session and initiative_factory's
    initiative (defaults: chat_stub, SessionManager, Initiative; set them on the backend first)."""
    backend = build_backend(backend_config(tmp_path))
    backend.session = SimpleNamespace(is_open=True, runtime_snapshot=lambda: {'runtime': {}})
    return backend
