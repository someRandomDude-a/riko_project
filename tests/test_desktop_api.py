"""The desktop backend's HTTP and WebSocket API, each test on its own app (desktop_server.create_app over tmp_path, the
conftest `backend` fixture) with stub services: nothing is patched into a module, and no test sees another's data root."""
from types import SimpleNamespace
import json
import os
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from conftest import backend_config, build_backend, chat_stub, client_for, vrm_bytes
from process.app_core.configuration.paths import DataPaths
from process.app_core.http.chat import BUSY
from process.app_core.http.settings import security_sensitive
from process.app_core.kernel.audio_config import audio_sections


def test_server_lifespan_owns_workers_and_removes_subscriptions(backend):
    calls, given = [], []
    backend.services_factory = lambda config, desktop_state: calls.append('create') or given.append(desktop_state) or chat_stub()
    class Session:
        def __init__(self, *args): pass
        def runtime_snapshot(self): return {'runtime': {}}
        def close(self): calls.append('close')
    backend.session_factory, backend.initiative_factory = Session, lambda session: None
    before, listeners = len(backend.bus._listeners), list(backend.state.events._listeners)  # the board's own saver
    assert calls == []
    with client_for(backend) as client:
        assert calls == ['create'] and given == [backend.state]  # the tools and the emotion bridge act on the state the routes serve
        assert client.get('/api/status').status_code == 200
        assert len(backend.state.events._listeners) == len(listeners) + 1
    assert calls == ['create', 'close']
    assert backend.state.events._listeners == listeners
    assert len(backend.bus._listeners) == before


def test_animation_assets_and_results_are_scoped(backend):
    from process.app_core.runtime.actions import ActionController
    directory = backend.config.root / 'character_files'
    directory.mkdir()
    path = directory / 'wake.vrma'
    path.write_bytes(b'test animation')
    outside = backend.config.root / 'outside.vrma'
    outside.touch()
    actions = backend.session.actions = ActionController()
    client = client_for(backend)
    try:
        assert client.get('/api/avatar/animation', params={'path': 'character_files/wake.vrma'}).content == b'test animation'
        assert client.get('/api/avatar/animation', params={'path': 'outside.vrma'}).status_code == 400
        assert client.get('/api/media', params={'path': 'character_files/wake.vrma'}).status_code == 400
        action = actions.start('wake_animation', {'path': str(path)}, 10)
        assert client.post('/api/avatar/animation/result', json={'action_id': action.id, 'status': 'started'}).status_code == 200
        assert action.status == 'running'
        assert client.post('/api/avatar/animation/result', json={'action_id': action.id, 'status': 'completed'}).status_code == 200
        assert action.status == 'complete'
        assert client.post('/api/avatar/animation/result', json={'action_id': action.id, 'status': 'started'}).status_code == 404
    finally: actions.close()


def test_audio_volume_is_live_bounded_and_independent_of_mute(backend):
    client = client_for(backend)
    backend.state.toggle_audio()
    result = client.patch('/api/audio/volume', json={'volume': .35})
    assert result.status_code == 200
    assert result.json() == {'volume': .35, 'enabled': False}
    assert backend.state.snapshot()['audio_volume'] == .35
    assert not backend.state.audio_enabled
    for value in (-.1, 1.1, True, '0.5', None):
        assert client.patch('/api/audio/volume', json={'volume': value}).status_code == 422
    assert backend.state.audio_volume == .35
    assert client.post('/api/audio/toggle').json()['enabled']
    assert backend.state.audio_volume == .35


def test_discord_start_is_explicit_and_runtime_scoped(backend):
    calls = []
    backend.discord_launcher = SimpleNamespace(status=lambda: {'running': False, 'error': ''}, start=lambda: calls.append('start') or {'running': True, 'error': ''})
    client = client_for(backend)
    assert client.get('/api/discord/process').json()['running'] is False
    assert calls == []
    assert client.post('/api/discord/start').json()['running'] is True
    assert calls == ['start']
    backend.session = None
    assert client.post('/api/discord/start').status_code == 503
    assert calls == ['start']


def test_avatar_library_import_and_settings_apply_live(backend):
    from process.app_core.configuration.settings_store import SettingsStore
    path = backend.config.root/'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\n')
    store = SettingsStore(path)
    backend.settings_store = lambda: store
    source = backend.config.root/'source.vrm'; source.write_bytes(vrm_bytes())
    client = client_for(backend)
    imported = client.post('/api/avatar/models/import',json={'value':str(source)})
    assert imported.status_code == 200
    model = imported.json()['path']
    assert client.get('/api/avatar/models').json()['entries'][0]['path'] == model
    result = client.put('/api/settings',json={'revision':store.snapshot()['revision'], 'changes':{'avatar.model':model,'avatar.format':'auto'}})
    assert result.status_code == 200 and result.json()['saved']
    assert client.get('/api/status').json()['avatar']['model'] == model
    assert client.get('/api/avatar/model').content == source.read_bytes()
    other = backend.config.root/'second.vrm'; other.write_bytes(vrm_bytes('vrm0'))
    imported0 = client.post('/api/avatar/models/import',json={'value':str(other)}).json()['path']
    revision = store.snapshot()['revision']
    invalid = client.put('/api/settings',json={'revision':revision, 'changes':{'avatar.model':imported0,'avatar.format':'vrm1'}})
    assert not invalid.json()['saved']
    assert client.get('/api/status').json()['avatar']['model'] == model
    switched = client.put('/api/settings',json={'revision':revision, 'changes':{'avatar.model':imported0,'avatar.format':'vrm0'}})
    assert switched.json()['saved'] and not switched.json()['restart_required']
    assert client.get('/api/status').json()['avatar'] == {'model':imported0,'format':'vrm0'}
    # A query value cannot request an arbitrary local path or the old model.
    response = client.get('/api/avatar/model',params={'selection':str(source)})
    assert response.content == other.read_bytes() and response.headers['cache-control'] == 'no-store'
    assert client.post('/api/avatar/models/import',json={'value':str(path)}).status_code == 400


def test_background_pause_setting_applies_live_without_restart(backend):
    from process.app_core.configuration.settings_store import SettingsStore
    from process.app_core.kernel.turns import RuntimeStatus
    path=backend.config.root/'character_config.yaml';path.write_text('runtime:\n  provider: lm_studio\n')
    store=SettingsStore(path)
    backend.settings_store=lambda:store
    calls=[]
    provider=SimpleNamespace(set_pause_background=lambda value:calls.append(('pause',value)))
    memory=SimpleNamespace(set_foreground=lambda value:calls.append(('memory',value)))
    backend.chat=SimpleNamespace(provider=provider,memory_store=memory)
    backend.session.status=lambda: RuntimeStatus(generating=True)  # a reply is generating: turning the pause off releases memory
    result=client_for(backend).put('/api/settings',json={'revision':store.snapshot()['revision'],'changes':{'runtime.pause_background_on_live':False}})
    assert result.status_code==200 and result.json()['saved'] and not result.json()['restart_required']
    assert calls==[('pause',False),('memory',False)]


def test_settings_apply_live_through_the_hook_their_schema_entry_names(backend, monkeypatch):
    """Regression: a setting applied live had to be listed in field(), in save() and in an if-chain in save_settings; one
    missing left it reported as restart-only, or applied without Settings saying so. Each live setting now names its hook."""
    from process.app_core.configuration import schema as settings
    from process.app_core.configuration.settings_store import SettingsStore
    from process.app_core.http.settings import LIVE
    from process.app_core.kernel import schema
    named = {item.live for item in settings.sections()} | {spec.live for item in settings.sections() for spec in item.settings}
    assert named - {None} == set(LIVE)
    for item in settings.sections():
        for spec in item.settings:
            if settings.restart_scope(item.path_of(spec.key)) == 'none': assert settings.resolve(item.path_of(spec.key), 'live'), spec.key
    registry = schema.Registry()
    registry.register(*schema.REGISTRY.sections(), schema.Section('demo', restart='none', live='demo', settings=(schema.Setting('speed', 2.5),)))
    monkeypatch.setattr(schema, 'REGISTRY', registry)
    calls = []
    monkeypatch.setitem(LIVE, 'demo', lambda server, values, paths: calls.append((server is backend, values['demo.speed'], paths)))
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\n')
    store = SettingsStore(path)
    backend.settings_store = lambda: store
    client = client_for(backend)
    result = client.put('/api/settings', json={'revision': store.snapshot()['revision'], 'changes': {'demo.speed': 3}}).json()
    assert result['saved'] and not result['restart_required'] and calls == [(True, 3, ['demo.speed'])]  # the hook gets this app's backend
    result = client.put('/api/settings', json={'revision': result['revision'], 'changes': {'runtime.temperature': .5}}).json()
    assert result['saved'] and result['restart_required'] and len(calls) == 1


def test_gpu_draft_feedback_can_estimate_budget_mismatch_without_fixing_values(backend,monkeypatch):
    from process.app_core.configuration.settings_store import SettingsStore
    path=backend.config.root/'character_config.yaml'
    path.write_text('runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n  n_ctx: 8192\n  max_output_tokens: 1024\nmemory:\n  context_window_tokens: 7168\n')
    store=SettingsStore(path)
    before=path.read_bytes()
    backend.settings_store=lambda:store
    backend.gpu_monitor.sample=lambda *args:{}
    monkeypatch.setattr('process.app_core.resources.vram_estimate.estimate',lambda config,telemetry:{'n_ctx':config.runtime.n_ctx,'warnings':[]})
    client=client_for(backend)
    result=client.post('/api/resources/estimate',json={'changes':{'runtime.n_ctx':2048}})
    assert result.status_code==200 and result.json()['estimate']['n_ctx']==2048
    assert result.json()['validation_errors'] and result.json()['estimate']['warnings']
    assert not store.validate({'runtime.n_ctx':2048})['valid']
    assert path.read_bytes()==before


def test_animation_library_api_import_assignment_preview_and_interaction(backend):
    import json
    import threading
    from process.app_core.runtime.actions import ActionController
    from process.app_core.animation.runtime import AnimationRuntime
    from process.app_core.kernel.turns import RuntimeStatus
    session = SimpleNamespace(config=backend.config, chat=SimpleNamespace(), state=backend.state,
        actions=ActionController(), status=lambda: RuntimeStatus(listening=True, voice_status='ready'))
    service = AnimationRuntime(session, start=False)
    session.animation = service
    backend.session = session
    source = backend.config.root / 'source.pose.json'
    source.write_text(json.dumps({'version': 1, 'bones': {'head': [0, .1, 0]}}))
    client = client_for(backend)
    try:
        assert client.post('/api/animation/capabilities', json={'bones': ['head'], 'expressions': []}).status_code == 200
        response = client.post('/api/animation/import', json={'path': str(source), 'metadata': {'states': ['idle']}})
        assert response.status_code == 200
        entry = response.json()
        assert client.get(f"/api/animation/assets/{entry['id']}/file").content == source.read_bytes()
        assert client.get('/api/animation/assets/not-an-asset/file').status_code == 404
        assert client.patch(f"/api/animation/assets/{entry['id']}", json={'mask': ['tail']}).status_code == 400
        assert client.patch(f"/api/animation/assets/{entry['id']}", json={'speed': .5}).status_code == 200
        preview = client.post(f"/api/animation/assets/{entry['id']}/preview")
        assert preview.status_code == 200
        assert client.post('/api/avatar/animation/result', json={'action_id': preview.json()['action_id'], 'status': 'cancelled'}).status_code == 200
        assert client.post('/api/animation/interaction', json={'kind': 'hold'}).status_code == 200
        service.step()
        assert client.get('/api/animation').json()['state']['mode'] == 'held'
        assert client.post('/api/animation/interaction', json={'kind': 'pointer', 'pointer': {'x': 3, 'y': 0, 'near': True}}).status_code == 400
        assert client.post('/api/animation/stop').status_code == 200
    finally: service.close(); session.actions.close()


def test_resource_estimate_previews_drafts_without_saving(backend):
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: openai\nvoice:\n  asr_device: cpu\n')
    before = path.read_bytes()
    backend.gpu_monitor.sample = lambda *args: {'available': False, 'gpus': []}
    client = client_for(backend)
    response = client.post('/api/resources/estimate', json={'changes': {'memory.reflection_context_window_tokens': 8192}})
    assert response.status_code == 200
    assert response.json()['estimate']['kv']['reflection_context_tokens'] == 8192
    assert response.json()['draft']
    assert path.read_bytes() == before
    assert not list(path.parent.glob('.resource-estimate-*'))
    assert client.post('/api/resources/estimate', json={'changes': {'runtime.n_ctx': -1}}).status_code == 400


def test_resource_estimate_never_touches_a_drafted_network_path(backend, monkeypatch):
    from pathlib import Path
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: llama_cpp\n')
    backend.gpu_monitor.sample = lambda *args: {'available': False, 'gpus': []}
    touched = []
    for name in ('stat', 'exists', 'is_file', 'is_dir', 'resolve', 'open', 'rglob'):
        original = getattr(Path, name)
        def spy(self, *args, _original=original, **kwargs):
            if 'attacker' in str(self): touched.append(str(self)); raise AssertionError('a network path reached the filesystem')
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(Path, name, spy)
    client = client_for(backend)
    for key in ('runtime.model_path', 'emotion.model_path', 'emotion.cache_dir'):
        for value in ('//attacker/share/model.gguf', r'\\attacker\share\model.gguf'):
            response = client.post('/api/resources/estimate', json={'changes': {key: value}})
            assert response.status_code == 400 and 'Network and device paths' in response.json()['detail'], (key, value)
    assert not touched


def test_tool_approval_api_policy_and_decision(backend, tmp_path):
    from process.app_core.tools.registry import ToolRegistry, RegisteredTool
    from process.app_core.tools.approval import ToolApprovals
    registry = ToolRegistry()
    registry.tools['example'] = RegisteredTool('example', 'Test tool', {}, lambda args: 'ok')
    registry.approvals = ToolApprovals(tmp_path / 'approvals.json')
    backend.chat = SimpleNamespace(tool_registry=registry)
    client = client_for(backend)
    try:
        assert client.get('/api/tools/approvals').json()['tools'][0]['name'] == 'example'
        assert client.put('/api/tools/approvals', json={'policy': {'example': True}}).status_code == 200
        assert client.put('/api/tools/approvals', json={'policy': {'missing': True}}).status_code == 400
        assert client.post('/api/tools/approvals/expired', json={'approved': True}).status_code == 409
    finally: registry.close()


def test_resource_websocket_bootstrap_and_source_triggered_updates(backend):
    from process.app_core.events.bus import EventBus
    from process.app_core.events.resources import ResourceEvents
    bus = EventBus()
    pending = []
    backend.bus = bus
    bridge = backend.resource_events = ResourceEvents(bus, {'approvals': lambda: {'pending': list(pending)}})
    client = client_for(backend)
    try:
        with client.websocket_connect('ws://127.0.0.1:8765/ws/events') as socket:
            assert socket.receive_json()['type'] == 'state.snapshot'
            assert socket.receive_json()['payload']['approvals']['pending'] == []
            pending.append({'id':'request'})
            bus.publish('tool.approval_requested')
            # The trigger and its resource update, which the bridge computes on its worker (never the publisher's thread), so
            # either may arrive first.
            events = {event['type']: event for event in (socket.receive_json() for _ in range(2))}
            assert set(events) == {'tool.approval_requested', 'resource.approvals'}
            assert events['resource.approvals']['payload']['pending'] == [{'id':'request'}]
    finally: bridge.close()


def test_corrupt_saved_desktop_settings_do_not_abort_startup(backend):
    path = backend.config.root / 'persistent_memories' / 'desktop_settings.json'
    path.parent.mkdir()
    path.write_text('{"avatar_geometry":{"width":0}}', encoding='utf-8')
    original = path.read_bytes()
    backend.session_factory = lambda *args: SimpleNamespace(runtime_snapshot=lambda: {'runtime': {}}, close=lambda: None)
    backend.initiative_factory = lambda session: None
    with client_for(backend):
        assert backend.state.geometry.avatar()['width'] == 480
        assert not backend.state.geometry.settings_loaded
    assert path.read_bytes() == original


def test_surface_api_rejects_empty_bounds_and_noninteger_geometry(backend):
    client = client_for(backend)
    command = backend.state.add_whiteboard('text', {'text': 'hello'})
    response = client.post('/api/surfaces/result', json={
        'surface': 'whiteboard', 'command_id': command, 'status': 'rendered', 'bounds': {}})
    assert response.status_code == 400
    assert client.patch('/api/surfaces/avatar', json={'width': 1}).status_code == 400
    assert client.patch('/api/surfaces/avatar', json={'x': True}).status_code == 422
    assert client.patch('/api/surfaces/whiteboard', json={'geometry': {'screen': -1}}).status_code == 400


def test_failed_session_start_closes_constructed_chat(backend):
    closed=[]
    backend.services_factory=lambda config, **_:chat_stub(close=lambda:closed.append('chat'))
    def fail(*args): raise RuntimeError('session startup failed')
    backend.session_factory=fail
    with pytest.raises(RuntimeError,match='session startup failed'):
        with client_for(backend): pass
    assert closed==['chat']


def test_a_multi_word_companion_name_starts_the_real_session(backend):
    """Regression: WakeWord refused a two-word name such as the fixture's 'Test character' in every mode, which the
    packaged setup accepts, and the lifespan aborted after the model had loaded."""
    backend.initiative_factory = lambda session: None  # the real SessionManager over the stub chat
    with client_for(backend) as client:
        wake = client.get('/api/voice/status').json()['wake']
        assert (wake['mode'], wake['wake_word'], wake['error']) == ('wake_word', 'Test', '')
        assert client.get('/api/status').json()['startup_error'] == ''
    assert not backend.session.is_open


def test_setup_mode_survives_a_session_that_cannot_start(backend):
    """A session that fails to build after the model has loaded keeps Settings up like a model error. (load_config has
    checked the voice values already; wake feedback is checked again as the session builds it.)"""
    backend.config.raw.update(desktop={'setup_on_startup_error': True}, wake_feedback={'volume': 5})
    (backend.config.root / 'character_config.yaml').write_text('runtime:\n  provider: lm_studio\n')
    closed = []
    backend.services_factory = lambda config, **_: chat_stub(close=lambda: closed.append('chat'))
    listeners = len(backend.bus._listeners)
    with client_for(backend) as client:
        assert 'wake_feedback.volume must be' in client.get('/api/status').json()['startup_error']
        assert client.get('/api/settings').status_code == 200
        assert client.get('/api/voice/status').status_code == 503
        assert closed == ['chat'] and backend.chat is None and backend.session is None  # the loaded model is released
    assert len(backend.bus._listeners) == listeners


def test_a_voice_value_load_config_could_not_read_starts_setup_mode_instead_of_killing_the_backend(backend, tmp_path):
    """Regression: voice and GPT-SoVITS values are checked by load_config, which run_server calls before the lifespan's
    setup fallback. With desktop.setup_on_startup_error the section loads with its defaults and the lifespan starts in setup
    mode, without loading the model, so Settings can repair it, as when the session used to reject the value."""
    from process.app_core.configuration.config import load_config as real_load_config
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\ndesktop:\n  setup_on_startup_error: true\nvoice:\n  mode: push_to_talk\n'
        'sovits_ping_config:\n  max_in_flight_requests: 0\n', encoding='utf-8')
    loaded = real_load_config(path, recover='setup')
    assert loaded.voice.mode == 'wake_word' and len(loaded.load_errors) == 2 and 'voice.mode must be' in '; '.join(loaded.load_errors)
    with pytest.raises(ValueError, match='(voice.mode|sovits_ping_config.max_in_flight_requests) must be'): real_load_config(path)  # Settings validation stays strict
    path.write_text(path.read_text(encoding='utf-8').replace('  setup_on_startup_error: true', '  setup_on_startup_error: false'), encoding='utf-8')
    with pytest.raises(ValueError, match='(voice.mode|sovits_ping_config.max_in_flight_requests) must be'): real_load_config(path, recover='setup')  # no setup mode: fail fast
    backend.config.raw.update(desktop={'setup_on_startup_error': True})
    backend.config.load_errors = loaded.load_errors
    (backend.config.root / 'character_config.yaml').write_text('runtime:\n  provider: lm_studio\n')
    backend.services_factory = lambda config, **_: pytest.fail('the model must not load with a broken section')
    with client_for(backend) as client:
        assert 'voice.mode must be' in client.get('/api/status').json()['startup_error']
        assert client.get('/api/settings').status_code == 200


def test_setup_mode_keeps_settings_accessible_with_cors(backend):
    backend.config.raw['desktop']={'setup_on_startup_error':True}
    (backend.config.root/'character_config.yaml').write_text('runtime:\n  provider: lm_studio\n')
    def fail(config, **_): raise RuntimeError('llama-server missing')
    backend.services_factory=fail
    with client_for(backend) as client:
        assert client.get('/api/settings').status_code==200
        assert client.get('/api/status').json()['startup_error']=='llama-server missing'
        response=client.get('/api/voice/status',headers={'Origin':'http://127.0.0.1:5173'})
        assert response.status_code==503
        assert response.headers['access-control-allow-origin']=='http://127.0.0.1:5173'


def test_history_api_is_paginated_without_replaying_events(backend):
    from process.app_core.persistence.conversation_store import ConversationStore
    from process.app_core.kernel.messages import ChatMessage
    store=backend.conversation_store=ConversationStore(backend.config.root/'history.sqlite3',legacy=[ChatMessage('user',str(i)) for i in range(10)])
    client=client_for(backend)
    try:
        page=client.get('/api/chat/history?limit=3').json()
        assert [m['text'] for m in page['messages']]==['7','8','9']
        older=client.get('/api/chat/history',params={'limit':3,'before':page['before']}).json()
        assert [m['text'] for m in older['messages']]==['4','5','6']
        assert client.get('/api/chat/history?limit=1000').status_code==400
    finally: store.close()


def test_display_api_validates_payload_and_defaults_to_primary(backend):
    client = client_for(backend)
    assert client.post('/api/displays', json=[{}]).status_code == 422
    assert client.post('/api/displays', json=[]).status_code == 400
    displays = [{'index': index, 'id': index + 10, 'label': f'Screen {index}', 'primary': index == 1,
                 'bounds': {'x': index * 1920, 'y': 0, 'width': 1920, 'height': 1080}, 'scaleFactor': 1}
                for index in range(2)]
    assert client.post('/api/displays', json=displays).status_code == 200
    assert backend.state.geometry.avatar()['screen'] == 1
    assert client.patch('/api/surfaces/avatar', json={'screen': 9}).status_code == 400
    assert client.patch('/api/surfaces/avatar', json={'screen': 0}).status_code == 200
    # Removing the selected display safely falls back to the surviving primary.
    assert client.post('/api/displays', json=[{**displays[1], 'index': 0}]).status_code == 200
    assert backend.state.geometry.avatar()['screen'] == 0


def test_avatar_geometry_is_saved_by_its_component_and_restored_at_the_next_start(backend):
    displays = [{'index': index, 'id': index + 10, 'label': f'Screen {index}', 'primary': index == 1,
                 'bounds': {'x': index * 1920, 'y': 0, 'width': 1920, 'height': 1080}, 'scaleFactor': 1} for index in range(2)]
    assert client_for(backend).patch('/api/surfaces/avatar', json={'width': 360}).status_code == 200
    path = backend.config.paths.desktop_settings  # <data>/persistent_memories/desktop_settings.json, where it always was
    assert path == backend.config.root / 'persistent_memories' / 'desktop_settings.json'
    assert json.loads(path.read_text(encoding='utf-8')) == {'avatar_geometry': {'x': 0, 'y': 0, 'width': 360, 'height': 720, 'screen': 0}}
    restarted = build_backend(backend.config, initiative_factory=lambda session: None,  # the next backend start: a new app and state
        session_factory=lambda *args: SimpleNamespace(runtime_snapshot=lambda: {'runtime': {}}, close=lambda: None))
    assert restarted.state is not backend.state and not restarted.state.geometry.settings_loaded  # building the app read nothing
    with client_for(restarted) as client:
        assert restarted.state.geometry.avatar()['width'] == 360 and restarted.state.geometry.settings_loaded
        assert client.post('/api/displays', json=displays).status_code == 200
        assert client.get('/api/status').json()['avatar_geometry']['screen'] == 0  # the saved display, not the primary


def test_every_route_needs_the_install_token_and_a_loopback_host(backend):
    token = backend.api_token()
    assert client_for(backend).get('/api/discord/inbox').status_code == 200
    anonymous = TestClient(backend.app, base_url='http://127.0.0.1:8765')
    assert anonymous.get('/api/discord/inbox').status_code == 401  # e.g. a web page in a browser tab
    assert anonymous.get('/api/discord/inbox', headers={'Authorization': 'Bearer wrong'}).status_code == 401
    rebound = TestClient(backend.app, base_url='http://attacker.example:8765', headers={'Authorization': 'Bearer ' + token})
    assert rebound.get('/api/discord/inbox').status_code == 403  # DNS rebinding keeps the attacker's Host
    folder = backend.config.root / 'persistent_memories'
    key = backend.api_secrets()[1]
    assert (folder / 'api_token').read_text() == token and (folder / 'confirm_key').read_text() == key and len(token) >= 32 and key != token
    if os.name != 'nt': assert all((folder / name).stat().st_mode & 0o777 == 0o600 for name in ('api_token', 'confirm_key'))


def test_websockets_need_the_token_and_an_app_origin(backend):
    from starlette.websockets import WebSocketDisconnect
    url = 'ws://127.0.0.1:8765/ws/events'
    with client_for(backend).websocket_connect(url, headers={'Origin': 'file://'}) as socket:
        assert socket.receive_json()['type'] == 'state.snapshot'
    with pytest.raises(WebSocketDisconnect):
        with TestClient(backend.app).websocket_connect(url): pass  # no token
    with pytest.raises(WebSocketDisconnect):
        with client_for(backend).websocket_connect(url, headers={'Origin': 'https://attacker.example'}): pass


def test_api_docs_are_off_outside_development(backend):
    client = client_for(backend)
    assert client.get('/docs').status_code == 404 and client.get('/openapi.json').status_code == 404


def test_security_sensitive_settings_need_a_signature_only_electron_main_can_make(backend):
    from process.app_core.configuration.settings_store import SettingsStore
    from process.app_core.desktop.api_guard import signature
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\n')
    store = SettingsStore(path)
    backend.settings_store = lambda: store
    client = client_for(backend)
    body = {'revision': store.snapshot()['revision'], 'changes': {'sovits_ping_config.executable': '/tmp/not-sovits'}}
    asked = client.put('/api/settings', json=body)
    assert asked.status_code == 428 and path.read_text() == 'runtime:\n  provider: lm_studio\n'
    challenge = asked.json()['detail']['confirm']
    assert client.put('/api/settings', json=body, headers={'X-Riko-Confirmation': 'forged'}).status_code == 428
    # The API token is on the wire with every request, so it must not be able to sign confirmations.
    assert client.put('/api/settings', json=body, headers={'X-Riko-Confirmation': signature(backend.api_token(), challenge)}).status_code == 428
    signed = client.put('/api/settings', json=body, headers={'X-Riko-Confirmation': signature(backend.api_secrets()[1], challenge)})
    assert signed.status_code == 200 and signed.json()['saved'] and 'not-sovits' in path.read_text()
    path.write_text('runtime:\n  provider: lm_studio\n')  # changed back; the old approval must not apply again
    replayed = client.put('/api/settings', json={**body, 'revision': store.snapshot()['revision']}, headers={'X-Riko-Confirmation': signature(backend.api_secrets()[1], challenge)})
    assert replayed.status_code == 428 and 'not-sovits' not in path.read_text()


def test_turning_tool_approval_off_needs_confirmation(backend, tmp_path):
    from process.app_core.desktop.api_guard import signature
    from process.app_core.tools.registry import ToolRegistry, RegisteredTool
    from process.app_core.tools.approval import ToolApprovals
    registry = ToolRegistry()
    registry.tools['example'] = RegisteredTool('example', 'Test tool', {}, lambda args: 'ok')
    registry.approvals = ToolApprovals(tmp_path / 'approvals.json', True)
    backend.chat = SimpleNamespace(tool_registry=registry)
    client = client_for(backend)
    try:
        assert client.put('/api/tools/approvals', json={'policy': {'example': True}}).status_code == 200  # stricter: no prompt
        asked = client.put('/api/tools/approvals', json={'policy': {'example': False}})
        assert asked.status_code == 428 and registry.approvals.snapshot()['policy']['example'] is True
        proof = signature(backend.api_secrets()[1], asked.json()['detail']['confirm'])
        assert client.put('/api/tools/approvals', json={'policy': {'example': False}}, headers={'X-Riko-Confirmation': proof}).status_code == 200
        assert registry.approvals.snapshot()['policy']['example'] is False
    finally: registry.close()


def test_a_server_tool_named_like_a_built_in_has_its_own_rule_and_turning_it_off_is_confirmed(backend, tmp_path):
    from process.app_core.desktop.api_guard import signature
    from process.app_core.tools.registry import ToolRegistry, RegisteredTool
    from process.app_core.tools.approval import ToolApprovals
    from process.app_core.tools.tool import RIKO
    class Server:
        def list_tools(self): return [{'name': 'todo_list', 'inputSchema': {'type': 'object'}}]
        def call(self, name, arguments): return {'content': []}
        def close(self): pass
    (tmp_path / 'approvals.json').write_text('{"todo_list": false}', encoding='utf-8')  # saved before rules had sources
    registry = ToolRegistry()
    registry.register(RegisteredTool('todo_list', 'Built-in', {}, lambda args: 'ok'), source=RIKO)
    registry.register_mcp(Server(), source='mcp:evil')
    registry.approvals = ToolApprovals(tmp_path / 'approvals.json', True)
    backend.chat = SimpleNamespace(tool_registry=registry)
    client = client_for(backend)
    try:
        listed = client.get('/api/tools/approvals').json()
        assert listed['tools'] == [{'name': 'todo_list', 'description': 'Built-in', 'source': 'riko'}, {'name': 'evil__todo_list', 'description': '', 'source': 'mcp:evil'}]
        assert listed['policy'] == {'todo_list': False}  # the server's tool follows the default: ask first
        asked = client.put('/api/tools/approvals', json={'policy': {'evil__todo_list': False}})
        assert asked.status_code == 428 and registry.approvals.required(('mcp:evil', 'todo_list'))
        proof = signature(backend.api_secrets()[1], asked.json()['detail']['confirm'])
        saved = client.put('/api/tools/approvals', json={'policy': {'evil__todo_list': False}}, headers={'X-Riko-Confirmation': proof})
        assert saved.status_code == 200 and saved.json()['policy'] == {'todo_list': False, 'evil__todo_list': False}
        assert client.put('/api/tools/approvals', json={'policy': {'todo_list': True}}).status_code == 200  # stricter: no prompt
        assert registry.approvals.required((RIKO, 'todo_list')) and not registry.approvals.required(('mcp:evil', 'todo_list'))
    finally: registry.close()


def test_a_server_tool_asks_first_by_default_and_turning_that_off_is_confirmed_even_with_approval_off(backend, tmp_path):
    from process.app_core.tools.registry import ToolRegistry
    from process.app_core.tools.approval import ToolApprovals
    class Server:
        def list_tools(self): return [{'name': 'todo_list', 'description': 'Lists things', 'inputSchema': {'type': 'object'}}]
        def call(self, name, arguments): return {'content': []}
        def close(self): pass
    registry = ToolRegistry()
    registry.register_mcp(Server(), source='mcp:evil')
    registry.approvals = ToolApprovals(tmp_path / 'approvals.json', False)  # tools.require_approval: false
    backend.chat = SimpleNamespace(tool_registry=registry)
    client = client_for(backend)
    try:
        listed = client.get('/api/tools/approvals').json()
        assert listed['default_required'] is False and listed['policy'] == {} and listed['required'] == {'todo_list': True}
        asked = client.put('/api/tools/approvals', json={'policy': {'todo_list': False}})
        assert asked.status_code == 428 and registry.approvals.required(('mcp:evil', 'todo_list'))
    finally: registry.close()


def test_discord_client_receives_only_its_own_turns_not_local_activity(backend):
    import uuid
    from process.app_core.events.bus import EventBus
    bus = backend.bus = EventBus()
    assert backend.resource_events is None  # before the runtime is ready: the launcher's own status
    instance = str(uuid.uuid4())
    with client_for(backend).websocket_connect(f'ws://127.0.0.1:8765/ws/discord/client?instance={instance}') as socket:
        assert socket.receive_json() == {'type': 'state.snapshot', 'payload': {}}  # no local desktop state
        assert set(socket.receive_json()['payload']) <= {'discord', 'approvals'}
        bus.publish('voice.transcript', text='private words spoken at the desk')
        bus.publish('chat.delta', turn_id='local', text='a local reply', source='message')
        bus.publish('model.started', turn_id='remote', source='discord')
        bus.publish('chat.delta', turn_id='remote', text='hello Discord', source='discord')
        bus.publish('initiative.presented', message='marker')
        received = [socket.receive_json() for _ in range(2)]
    assert [(event['type'], event['payload'].get('text') or event['payload'].get('message')) for event in received] == [
        ('chat.delta', 'hello Discord'), ('initiative.presented', 'marker')]


def test_confirmations_are_single_use_bound_to_the_change_and_expire():
    from process.app_core.desktop.api_guard import Confirmations, signature
    confirmations, key = Confirmations(), 'k' * 43
    challenge = confirmations.challenge('settings', {'runtime.native_library': 'a.dll'})
    proof = signature(key, challenge)
    assert not confirmations.consume(key, 'settings', {'runtime.native_library': 'b.dll'}, proof)  # a different change
    assert not confirmations.consume(key, 'discord_access', {'runtime.native_library': 'a.dll'}, proof)
    assert not confirmations.consume('other' * 9, 'settings', {'runtime.native_library': 'a.dll'}, proof)
    assert confirmations.consume(key, 'settings', {'runtime.native_library': 'a.dll'}, proof)
    assert not confirmations.consume(key, 'settings', {'runtime.native_library': 'a.dll'}, proof)  # replay
    expired = Confirmations(ttl=-1)
    challenge = expired.challenge('settings', {'tools.mcp_config': 'x'})
    assert not expired.consume(key, 'settings', {'tools.mcp_config': 'x'}, signature(key, challenge))


@pytest.mark.parametrize('key', ['runtime.native_library', 'runtime.provider', 'runtime.base_url', 'runtime.model_path',
    'runtime.hf_repo_id', 'tools.mcp_config', 'emotion.enabled', 'emotion.model_id', 'memory.system1_enabled',
    'memory.system1_model_id', 'memory.store_file', 'sovits_ping_config.url', 'sovits_ping_config.ref_audio_path',
    'sovits_ping_config.executable', 'sovits_ping_config.arguments', 'voice.asr_model', 'emotion.revision', 'memory.system1_revision'])
def test_settings_that_run_code_or_send_data_elsewhere_are_security_sensitive(key):
    assert security_sensitive(key)


@pytest.mark.parametrize('key', ['runtime.pause_background_on_live', 'runtime.temperature', 'avatar.scale', 'speech.max_words', 'voice.vad_threshold'])
def test_everyday_settings_save_without_confirmation(key):
    assert not security_sensitive(key)


def test_sections_directories_and_locations_cannot_slip_past_confirmation(backend):
    sensitive = security_sensitive
    assert sensitive('tools', {'mcp_config': '/tmp/evil/mcp.json', 'require_approval': False})  # an empty YAML mapping is one editable value
    assert sensitive('sovits_ping_config', {'nested': {'auto_start': True}}) and not sensitive('animation', {'walk_speed': 2})
    assert sensitive('desktop.effects_directory', 'effects') and sensitive('initiative.rules', [{'id': 'x', 'webhook': 'https://example.com'}])
    assert sensitive('presets.default.name', 'C:\\Windows') and sensitive('voice.wake_word', '~/words')
    assert not sensitive('avatar.model', '/Users/me/Avatar.vrm') and not sensitive('presets.default.name', 'Riko')
    from process.app_core.configuration.settings_store import SettingsStore
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\ntools: {}\n')
    store = SettingsStore(path)
    backend.settings_store = lambda: store
    asked = client_for(backend).put('/api/settings', json={'revision': store.snapshot()['revision'], 'changes': {'tools': {'mcp_config': '/tmp/evil/mcp.json'}}})
    assert asked.status_code == 428 and 'evil' not in path.read_text()


def test_secrets_are_fresh_each_start_and_never_inherited_by_children(tmp_path, monkeypatch):
    from process.app_core.desktop.api_guard import issue_secrets, client_token
    paths = DataPaths.at(tmp_path)
    first, second = issue_secrets(paths), issue_secrets(paths)
    assert first != second and (tmp_path / 'persistent_memories' / 'api_token').read_text() == second[0]
    assert (tmp_path / 'persistent_memories' / 'confirm_key').read_text() == second[1]  # where electron/main.cjs reads both
    monkeypatch.setenv('RIKO_API_TOKEN', 'a' * 43)
    monkeypatch.setenv('RIKO_CONFIRM_KEY', 'b' * 43)
    assert issue_secrets(DataPaths.at(tmp_path / 'packaged')) == ('a' * 43, 'b' * 43)  # Electron's values, no files
    assert 'RIKO_API_TOKEN' not in os.environ and 'RIKO_CONFIRM_KEY' not in os.environ
    assert not (tmp_path / 'packaged' / 'persistent_memories').exists()
    monkeypatch.setenv('RIKO_API_TOKEN', 'short')
    monkeypatch.setenv('RIKO_CONFIRM_KEY', 'b' * 43)
    token, _ = issue_secrets(paths)
    assert token != 'short' and 'RIKO_CONFIRM_KEY' not in os.environ
    # A client such as the Discord worker finds the token beside the backend's config.
    monkeypatch.setenv('RIKO_DATA_DIR', str(tmp_path))
    monkeypatch.delenv('RIKO_CONFIG', raising=False)
    assert client_token() == token
    monkeypatch.setenv('RIKO_API_TOKEN', 'c' * 43)
    assert client_token() == 'c' * 43


def test_stop_turn_cancels_the_live_session_before_the_drain(backend):
    calls = []
    backend.session = SimpleNamespace(is_open=True, cancel=lambda: calls.append('cancel'))
    backend.stop_turn()
    assert calls == ['cancel']
    backend.session = SimpleNamespace(is_open=False, cancel=lambda: calls.append('closed'))
    backend.stop_turn()
    backend.session = None  # setup mode: nothing to stop
    backend.stop_turn()
    stuck = threading.Event()
    backend.session = SimpleNamespace(is_open=True, cancel=lambda: stuck.wait(5))
    started = time.monotonic()
    try: backend.stop_turn()
    finally: stuck.set()
    assert calls == ['cancel'] and time.monotonic() - started < 3  # a stuck cancel is abandoned after 1 s


def test_chat_while_another_turn_runs_is_a_conflict_not_a_server_error(backend, monkeypatch):
    from pathlib import Path
    from process.app_core.runtime.session import SessionManager
    class Speech:
        def __init__(self, *args): pass
        def submit(self, *args): return False
        def cancel(self): pass
        def close(self): pass
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    config = SimpleNamespace(raw={}, root=Path('.'), paths=DataPaths.at(Path('.')), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    session = backend.session = SessionManager(config, chat_stub(close=lambda: None), backend.state)
    client = client_for(backend, raise_server_exceptions=False)
    session._turns.begin()  # a voice or Discord turn, or a reply still stopping
    try:
        response = client.post('/api/chat', json={'text': 'hello'})
        assert response.status_code == 409 and 'still handling another reply' in response.json()['detail']
    finally: session._turns.end(); session.close()
    def fail(*args): raise RuntimeError('model exploded')
    backend.session = SimpleNamespace(respond=fail)
    assert client.post('/api/chat', json={'text': 'hello'}).status_code == 500  # any other failure is still a server error


def test_lifespan_stops_discord_while_the_session_closes(backend):
    closing, calls = threading.Event(), []
    class Session:
        def __init__(self, *args): pass
        def runtime_snapshot(self): return {'runtime': {}}
        def close(self): closing.set(); calls.append('close')
    backend.session_factory, backend.initiative_factory = Session, lambda session: None
    backend.discord_launcher = SimpleNamespace(status=lambda: {'running': False, 'error': ''},
        stop=lambda: calls.append(('discord', closing.wait(5))))
    with client_for(backend): pass
    assert ('discord', True) in calls and 'close' in calls  # Discord's stop overlapped the session close


def test_an_interjection_that_cuts_the_reply_in_flight_redoes_it(backend):
    redone, calls = threading.Event(), []
    def respond(text, **kwargs):
        calls.append((text, kwargs['record_user'], kwargs['reply_to']))
        redone.set()
    backend.session = SimpleNamespace(voice_anchor=lambda: ('turn-1', 0, False),
        voice_transcript=lambda text, started_at, ended_at, anchor: 'reply', respond=respond)
    client = client_for(backend)
    body = {'text': 'and tomorrow', 'started_at': 1, 'ended_at': 2}
    assert client.post('/api/voice/interjection', json=body).json() == {'accepted': True}
    assert redone.wait(5) and calls == [('and tomorrow', False, 'turn-1')]  # the words are already in history
    def unexpected(*args): raise AssertionError('no reply is in flight')
    backend.session = SimpleNamespace(voice_anchor=lambda: None, voice_transcript=unexpected)
    assert client.post('/api/voice/interjection', json=body).json() == {'accepted': False}  # the client sends a chat turn


def refuse_attacker_paths(monkeypatch):
    """Fail on any filesystem call that sees the attacker's host: on Windows resolving or stat-ing \\\\attacker\\share
    contacts it and offers the user's credentials, so a network path must be refused as text (kernel/safe_paths.py)."""
    from pathlib import Path
    touched = []
    def guard(original):
        def call(path, *args, **kwargs):
            if 'attacker' in str(path): touched.append(str(path)); raise AssertionError('a network path reached the filesystem')
            return original(path, *args, **kwargs)
        return call
    for name in ('resolve', 'stat', 'exists', 'is_file', 'is_dir', 'open', 'read_bytes'): monkeypatch.setattr(Path, name, guard(getattr(Path, name)))
    for module, name in ((os, 'stat'), (os, 'lstat'), (os.path, 'realpath'), (os.path, 'exists'), (os.path, 'isfile'), (os.path, 'isdir')):
        monkeypatch.setattr(module, name, guard(getattr(module, name)))
    return touched


def test_network_paths_are_refused_as_text_on_every_route_that_takes_a_path(backend, monkeypatch):
    from process.app_core.animation.runtime import AnimationRuntime
    from process.app_core.configuration.settings_store import SettingsStore
    from process.app_core.kernel.turns import RuntimeStatus
    from process.app_core.runtime.actions import ActionController
    session = SimpleNamespace(config=backend.config, chat=SimpleNamespace(), state=backend.state, actions=ActionController(), status=lambda: RuntimeStatus())
    session.animation = service = AnimationRuntime(session, start=False)
    backend.session = session
    path = backend.config.root / 'character_config.yaml'; path.write_text('runtime:\n  provider: lm_studio\n')
    store = SettingsStore(path); backend.settings_store = lambda: store
    client = client_for(backend)
    touched = refuse_attacker_paths(monkeypatch)
    try:
        for share in ('//attacker/share/', r'\\attacker\share' + '\\', r'\\?\UNC\attacker\share' + '\\'):
            for response in (client.get('/api/media', params={'path': share + 'x.png'}),
                    client.get('/api/avatar/animation', params={'path': share + 'x.vrma'}),
                    client.post('/api/avatar/models/import', json={'value': share + 'x.vrm'}),
                    client.post('/api/animation/import', json={'path': share + 'x.vrma', 'metadata': {}}),
                    client.post('/api/settings/path', json={'key': 'memory.store_file', 'value': share + 'x.json'})):
                assert response.status_code == 400, (share, response.request.url, response.text)
        assert not touched
    finally: service.close(); session.actions.close()


def test_settings_never_send_a_secret_and_a_masked_one_saves_as_kept(backend, monkeypatch):
    from process.app_core.configuration.settings_store import SECRET_MASK, SettingsStore
    monkeypatch.delenv('RIKO_API_KEY', raising=False)
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\n  api_key: sk-saved-secret-1234\n')
    store = SettingsStore(path); backend.settings_store = lambda: store
    client = client_for(backend)
    snapshot = client.get('/api/settings')
    assert snapshot.status_code == 200 and 'sk-saved-secret' not in snapshot.text
    assert snapshot.json()['values']['runtime.api_key'] == SECRET_MASK and 'path' not in snapshot.json()
    # Saving other settings with the mask untouched asks nothing and keeps the key; so does retyping the same key.
    kept = client.put('/api/settings', json={'revision': snapshot.json()['revision'], 'changes': {'runtime.api_key': SECRET_MASK, 'runtime.temperature': .5}})
    assert kept.status_code == 200 and kept.json()['saved'] and 'api_key: sk-saved-secret-1234' in path.read_text() and 'sk-saved' not in kept.text
    same = client.put('/api/settings', json={'revision': kept.json()['revision'], 'changes': {'runtime.api_key': 'sk-saved-secret-1234'}})
    assert same.status_code == 200
    asked = client.put('/api/settings', json={'revision': same.json()['revision'], 'changes': {'runtime.api_key': 'sk-another-key'}})
    assert asked.status_code == 428 and 'sk-another-key' not in path.read_text()
    # The native dialog shows (and signs) only a fingerprint of the new key, which still binds the approval to that key.
    from process.app_core.desktop.api_guard import signature
    challenge = asked.json()['detail']['confirm']
    assert 'sk-another-key' not in challenge and json.loads(challenge)['changes']['runtime.api_key'].startswith('new secret, sha256 ')
    swapped = client.put('/api/settings', json={'revision': same.json()['revision'], 'changes': {'runtime.api_key': 'sk-swapped-key'}},
        headers={'X-Riko-Confirmation': signature(backend.api_secrets()[1], challenge)})
    assert swapped.status_code == 428 and 'sk-swapped-key' not in path.read_text()
    signed = client.put('/api/settings', json={'revision': same.json()['revision'], 'changes': {'runtime.api_key': 'sk-another-key'}},
        headers={'X-Riko-Confirmation': signature(backend.api_secrets()[1], challenge)})
    assert signed.status_code == 200 and 'api_key: sk-another-key' in path.read_text()
    cleared = client.put('/api/settings', json={'revision': signed.json()['revision'], 'changes': {'runtime.api_key': ''}})
    assert cleared.status_code == 428 and json.loads(cleared.json()['detail']['confirm'])['changes'] == {'runtime.api_key': ''}
    assert client.post('/api/settings/validate', json={'changes': {'runtime.api_key': SECRET_MASK}}).json()['valid']
    for body in ({'key': 'runtime.temperature', 'value': 'x'}, {'key': 'memory.store_file', 'value': '//attacker/share/x'}, {'value': 'memory_store.json'}):
        assert client.post('/api/settings/path', json=body).status_code in (400, 422), body
    checked = client.post('/api/settings/path', json={'key': 'memory.store_file', 'value': 'memory_store.json'}).json()
    assert checked == {'resolved': str(backend.config.root / 'memory_store.json'), 'exists': False, 'directory': False, 'executable': False}


# Every route the Electron app and the Discord worker call, as before the routes moved into routers: same paths and methods.
ROUTES = {
    '/api/animation': 'GET', '/api/animation/assets/{identifier}': 'PATCH', '/api/animation/assets/{identifier}/file': 'GET',
    '/api/animation/assets/{identifier}/preview': 'POST', '/api/animation/capabilities': 'POST', '/api/animation/import': 'POST',
    '/api/animation/interaction': 'POST', '/api/animation/stop': 'POST', '/api/animation/walk': 'POST', '/api/audio/toggle': 'POST',
    '/api/audio/volume': 'PATCH', '/api/avatar/animation': 'GET', '/api/avatar/animation/result': 'POST', '/api/avatar/model': 'GET',
    '/api/avatar/models': 'GET', '/api/avatar/models/import': 'POST', '/api/chat': 'POST', '/api/chat/history': 'GET',
    '/api/chat/stop': 'POST', '/api/discord/chat': 'POST', '/api/discord/inbox': 'GET', '/api/discord/process': 'GET',
    '/api/discord/settings': 'GET/PUT', '/api/discord/speech': 'POST', '/api/discord/start': 'POST', '/api/discord/stop': 'POST',
    '/api/discord/stop-client': 'POST', '/api/discord/transcribe': 'POST', '/api/displays': 'POST', '/api/initiative': 'GET/PUT',
    '/api/initiative/event': 'POST', '/api/media': 'GET', '/api/memories': 'GET', '/api/memories/{record_id}': 'DELETE/PATCH',
    '/api/mic/toggle': 'POST', '/api/neural/corpora': 'GET', '/api/neural/data': 'GET', '/api/neural/data/{sample_id}': 'PATCH',
    '/api/neural/groups': 'GET', '/api/neural/replay/{key}': 'POST', '/api/neural/status': 'GET', '/api/neural/train': 'POST',
    '/api/resources/electron': 'POST', '/api/resources/estimate': 'POST', '/api/resources/gpu': 'GET', '/api/settings': 'GET/PUT',
    '/api/settings/huggingface/files': 'GET', '/api/settings/huggingface/search': 'GET', '/api/settings/path': 'POST',
    '/api/settings/validate': 'POST', '/api/sleep/toggle': 'POST', '/api/status': 'GET', '/api/surfaces/avatar': 'PATCH',
    '/api/surfaces/result': 'POST', '/api/surfaces/whiteboard': 'PATCH', '/api/tasks': 'GET/POST', '/api/tasks/{task_id}': 'GET/PATCH',
    '/api/tools/approvals': 'GET/PUT', '/api/tools/approvals/{request_id}': 'POST', '/api/voice/activate': 'POST',
    '/api/voice/activity': 'POST', '/api/voice/calibration': 'POST', '/api/voice/devices': 'GET', '/api/voice/interjection': 'POST',
    '/api/voice/start': 'POST', '/api/voice/status': 'GET', '/api/voice/stop': 'POST', '/api/whiteboard/image': 'GET/POST',
    '/ws/chat': 'WS', '/ws/discord/client': 'WS', '/ws/events': 'WS', '/ws/resources/gpu': 'WS',
}


def route_table(routes, prefix=''):
    """{path: 'METHOD/METHOD'} for an app or router, through included routers."""
    from fastapi.routing import APIRoute, APIWebSocketRoute
    table = {}
    for route in routes:
        if hasattr(route, 'original_router'): found = route_table(route.original_router.routes, prefix + route.include_context.prefix)
        elif isinstance(route, APIRoute): found = {prefix + route.path: '/'.join(sorted(route.methods))}
        elif isinstance(route, APIWebSocketRoute): found = {prefix + route.path: 'WS'}
        else: continue
        for path, methods in found.items(): table[path] = '/'.join(sorted({*table.get(path, '').split('/'), *methods.split('/')} - {''}))
    return table


def test_the_routers_serve_every_route_with_its_old_path_and_methods_and_discord_owns_its_own(backend):
    from process.app_core.http import app as assembly
    from process.app_core.integrations.discord import api as discord_api
    assert route_table(backend.app.routes) == ROUTES
    session_routes = route_table(discord_api.create_router(lambda: None).routes)  # without a launcher: the session's routes
    assert session_routes == {path: ROUTES[path] for path in ('/api/discord/chat', '/api/discord/stop', '/api/discord/transcribe', '/api/discord/speech')}
    with pytest.raises(TypeError, match='confirm'): discord_api.create_router(lambda: None, lambda: None)  # who may use Discord needs confirmation
    owned = route_table(discord_api.create_router(lambda: None, lambda: None, confirm=lambda *args: None).routes)
    assert owned == {path: methods for path, methods in ROUTES.items() if 'discord' in path}  # every Discord route, none elsewhere
    assert not any('discord' in path for module in assembly.ROUTERS for path in route_table(module.router.routes))


def test_importing_desktop_server_builds_nothing_and_an_app_serves_the_config_it_is_given(tmp_path, monkeypatch):
    """Regression: importing desktop_server loaded the cwd's YAML and built the Discord launcher, GPU monitor and whiteboard
    images, so tests had to stub load_config before the first import, and the cached module kept the first test's data root."""
    import importlib
    def unexpected(*args, **kwargs): raise AssertionError('load_config at import')
    monkeypatch.setattr('process.app_core.configuration.config.load_config', unexpected)
    monkeypatch.chdir(tmp_path)
    module = importlib.reload(sys.modules.get('desktop_server') or importlib.import_module('desktop_server'))
    try:
        assert not {'config', 'state', 'session', 'discord_launcher', 'gpu_monitor', 'board_images'} & set(vars(module))
        built = []
        first, second = (tmp_path / 'first', tmp_path / 'second')
        apps = [module.create_app(backend_config(root), services_factory=lambda config, **_: built.append(config)) for root in (first, second)]
        assert built == [] and not list(tmp_path.iterdir())  # building an app reads, writes and starts nothing
        one, two = (app.state.backend for app in apps)
        assert (one.config.root, two.config.root) == (first, second) and one.state is not two.state
        assert one.discord_launcher.paths.root == first and two.discord_launcher.paths.root == second
        assert one.settings_store().path == first / 'character_config.yaml' and two.settings_store().path == second / 'character_config.yaml'
        one.app, two.app = apps
        assert client_for(one).get('/api/status').json()['character_name'] == 'Test character'
        assert TestClient(apps[1], base_url='http://127.0.0.1:8765', headers={'Authorization': 'Bearer ' + one.api_token()}).get('/api/status').status_code == 401
        assert (first / 'persistent_memories' / 'api_token').read_text() == one.api_token() != two.api_token()  # minted on first use, each in its root
        # `desktop_server.app` (uvicorn desktop_server:app) still works: its first use loads the config once.
        loaded = []
        monkeypatch.setattr('process.app_core.configuration.config.load_config', lambda **options: loaded.append(options) or backend_config(tmp_path / 'lazy'))
        assert module.app.state.backend.config.root == tmp_path / 'lazy' and module.app is module.app and loaded == [{'recover': 'setup'}]
        assert module.stop_turn.__self__ is module.app.state.backend
    finally: monkeypatch.undo(); vars(module).pop('app', None); importlib.reload(module)  # later imports get a fresh module


def test_a_feature_that_is_off_answers_503_and_its_resource_keeps_the_previous_value(backend, caplog):
    import logging
    from process.app_core.events.bus import EventBus
    from process.app_core.events.resources import ResourceEvents, ResourceUnavailable
    client = client_for(backend)
    assert client.get('/api/tools/approvals').json() == {'detail': 'Tool approvals unavailable'}  # no chat service yet
    assert client.get('/api/tools/approvals').status_code == client.put('/api/tools/approvals', json={}).status_code == 503
    assert client.get('/api/animation').json() == {'detail': 'Animation service is disabled or unavailable'}
    with pytest.raises(ResourceUnavailable): backend.animation()  # a domain getter, not an HTTP handler
    bus, published = EventBus(), []
    bus.subscribe(lambda event: published.append(event.type) if event.type.startswith('resource.') else None)
    bridge = ResourceEvents(bus, {'approvals': backend.approvals, 'animation': backend.animation})
    try:
        with caplog.at_level(logging.DEBUG, logger='process.app_core.events.resources'):
            bus.publish('tool.approval_requested'); bus.publish('animation.state')
            bridge.settle()
        assert published == [] and all(record.levelno == logging.DEBUG for record in caplog.records)  # off, not failing
    finally: bridge.close()
