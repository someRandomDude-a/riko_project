from types import SimpleNamespace
import importlib

import pytest
from fastapi.testclient import TestClient

from process.app_core.desktop.state import DesktopState


@pytest.fixture
def backend(monkeypatch, tmp_path):
    # Load settings with the real loader before the server-import startup stub.
    # Otherwise isolated API tests capture the no-argument stub in settings_store.
    importlib.import_module('process.app_core.configuration.settings_store')
    config = SimpleNamespace(root=tmp_path, raw={}, avatar={}, character_name='Test character', runtime=SimpleNamespace(provider='fake', warmup=False))
    monkeypatch.setattr('process.app_core.configuration.config.load_config', lambda: config)
    server = importlib.import_module('desktop_server')
    monkeypatch.setattr(server, 'config', config)
    monkeypatch.setattr(server, 'state', DesktopState())
    monkeypatch.setattr(server, 'chat', None)
    monkeypatch.setattr(server, 'session', SimpleNamespace(runtime_snapshot=lambda: {'runtime': {}}))
    monkeypatch.setattr(server, '_desktop_settings_loaded', False)
    monkeypatch.setattr(server, 'conversation_store', None)
    monkeypatch.setattr(server, 'startup_error', '')
    return server


def test_server_lifespan_owns_workers_and_removes_subscriptions(backend, monkeypatch):
    calls = []
    monkeypatch.setattr(backend, 'create_chat_service', lambda config: calls.append('create') or SimpleNamespace())
    class Session:
        def __init__(self, *args): pass
        def runtime_snapshot(self): return {'runtime': {}}
        def close(self): calls.append('close')
    monkeypatch.setattr(backend, 'SessionManager', Session)
    monkeypatch.setattr(backend, 'Initiative', lambda session: None)
    before = len(backend.event_bus._listeners)
    assert calls == []
    with TestClient(backend.app) as client:
        assert calls == ['create']
        assert client.get('/api/status').status_code == 200
        assert len(backend.state._listeners) == 1
    assert calls == ['create', 'close']
    assert backend.state._listeners == []
    assert len(backend.event_bus._listeners) == before


def test_animation_assets_and_results_are_scoped(backend, monkeypatch):
    from process.app_core.runtime.actions import ActionController
    directory = backend.config.root / 'character_files'
    directory.mkdir()
    path = directory / 'wake.vrma'
    path.write_bytes(b'test animation')
    outside = backend.config.root / 'outside.vrma'
    outside.touch()
    actions = ActionController()
    monkeypatch.setattr(backend.session, 'actions', actions, raising=False)
    client = TestClient(backend.app)
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
    client = TestClient(backend.app)
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


def test_discord_start_is_explicit_and_runtime_scoped(backend, monkeypatch):
    calls = []
    monkeypatch.setattr(backend, 'discord_launcher', SimpleNamespace(status=lambda: {'running': False, 'error': ''}, start=lambda: calls.append('start') or {'running': True, 'error': ''}))
    client = TestClient(backend.app)
    assert client.get('/api/discord/process').json()['running'] is False
    assert calls == []
    assert client.post('/api/discord/start').json()['running'] is True
    assert calls == ['start']
    monkeypatch.setattr(backend, 'session', None)
    assert client.post('/api/discord/start').status_code == 503
    assert calls == ['start']


def test_avatar_library_import_and_settings_apply_live(backend, monkeypatch):
    from test_avatar_models import vrm_bytes
    from process.app_core.configuration.settings_store import SettingsStore, load_config
    path = backend.config.root/'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\n')
    store = SettingsStore(path)
    monkeypatch.setattr(backend, 'settings_store', lambda: store)
    monkeypatch.setattr(backend, 'load_config', load_config)
    source = backend.config.root/'source.vrm'; source.write_bytes(vrm_bytes())
    client = TestClient(backend.app)
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


def test_background_pause_setting_applies_live_without_restart(backend,monkeypatch):
    from process.app_core.configuration.settings_store import SettingsStore
    path=backend.config.root/'character_config.yaml';path.write_text('runtime:\n  provider: lm_studio\n')
    store=SettingsStore(path)
    monkeypatch.setattr(backend,'settings_store',lambda:store)
    calls=[]
    provider=SimpleNamespace(set_pause_background=lambda value:calls.append(('pause',value)))
    memory=SimpleNamespace(set_foreground=lambda value:calls.append(('memory',value)))
    monkeypatch.setattr(backend,'chat',SimpleNamespace(provider=provider,memory_store=memory))
    monkeypatch.setattr(backend.session,'_generation_active',True,raising=False)
    result=TestClient(backend.app).put('/api/settings',json={'revision':store.snapshot()['revision'],'changes':{'runtime.pause_background_on_live':False}})
    assert result.status_code==200 and result.json()['saved'] and not result.json()['restart_required']
    assert calls==[('pause',False),('memory',False)]


def test_gpu_draft_feedback_can_estimate_budget_mismatch_without_fixing_values(backend,monkeypatch):
    from process.app_core.configuration.settings_store import SettingsStore,load_config
    path=backend.config.root/'character_config.yaml'
    path.write_text('runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n  n_ctx: 8192\n  max_output_tokens: 1024\nmemory:\n  context_window_tokens: 7168\n')
    store=SettingsStore(path)
    before=path.read_bytes()
    monkeypatch.setattr(backend,'settings_store',lambda:store)
    monkeypatch.setattr(backend,'load_config',load_config)
    monkeypatch.setattr(backend.gpu_monitor,'sample',lambda *args:{})
    monkeypatch.setattr('process.app_core.resources.vram_estimate.estimate',lambda config,telemetry:{'n_ctx':config.runtime.n_ctx,'warnings':[]})
    client=TestClient(backend.app)
    result=client.post('/api/resources/estimate',json={'changes':{'runtime.n_ctx':2048}})
    assert result.status_code==200 and result.json()['estimate']['n_ctx']==2048
    assert result.json()['validation_errors'] and result.json()['estimate']['warnings']
    assert not store.validate({'runtime.n_ctx':2048})['valid']
    assert path.read_bytes()==before


def test_animation_library_api_import_assignment_preview_and_interaction(backend, monkeypatch):
    import json
    import threading
    from process.app_core.runtime.actions import ActionController
    from process.app_core.animation.runtime import AnimationRuntime
    session = SimpleNamespace(config=backend.config, chat=SimpleNamespace(), state=backend.state,
        actions=ActionController(), _voice_lock=threading.RLock(), _voice_status='ready',
        _user_speaking=False, _playing=None, _generation_active=False, _speech_pending=0)
    service = AnimationRuntime(session, start=False)
    session.animation = service
    monkeypatch.setattr(backend, 'session', session)
    source = backend.config.root / 'source.pose.json'
    source.write_text(json.dumps({'version': 1, 'bones': {'head': [0, .1, 0]}}))
    client = TestClient(backend.app)
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


def test_resource_estimate_previews_drafts_without_saving(backend, monkeypatch):
    from process.app_core.configuration.settings_store import load_config
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: openai\nvoice:\n  asr_device: cpu\n')
    before = path.read_bytes()
    monkeypatch.setattr(backend, 'load_config', load_config)
    monkeypatch.setattr(backend.gpu_monitor, 'sample', lambda *args: {'available': False, 'gpus': []})
    client = TestClient(backend.app)
    response = client.post('/api/resources/estimate', json={'changes': {'memory.reflection_context_window_tokens': 8192}})
    assert response.status_code == 200
    assert response.json()['estimate']['kv']['reflection_context_tokens'] == 8192
    assert response.json()['draft']
    assert path.read_bytes() == before
    assert not list(path.parent.glob('.resource-estimate-*'))
    assert client.post('/api/resources/estimate', json={'changes': {'runtime.n_ctx': -1}}).status_code == 400


def test_tool_approval_api_policy_and_decision(backend, monkeypatch, tmp_path):
    from process.app_core.tools.registry import ToolRegistry, RegisteredTool
    from process.app_core.tools.approval import ToolApprovals
    registry = ToolRegistry()
    registry.tools['example'] = RegisteredTool('example', 'Test tool', {}, lambda args: 'ok')
    registry.approvals = ToolApprovals(tmp_path / 'approvals.json')
    monkeypatch.setattr(backend, 'chat', SimpleNamespace(tool_registry=registry))
    client = TestClient(backend.app)
    try:
        assert client.get('/api/tools/approvals').json()['tools'][0]['name'] == 'example'
        assert client.put('/api/tools/approvals', json={'policy': {'example': True}}).status_code == 200
        assert client.put('/api/tools/approvals', json={'policy': {'missing': True}}).status_code == 400
        assert client.post('/api/tools/approvals/expired', json={'approved': True}).status_code == 409
    finally: registry.close()


def test_resource_websocket_bootstrap_and_source_triggered_updates(backend, monkeypatch):
    from process.app_core.events.bus import EventBus
    from process.app_core.events.resources import ResourceEvents
    bus = EventBus()
    pending = []
    bridge = ResourceEvents(bus, {'approvals': lambda: {'pending': list(pending)}})
    monkeypatch.setattr(backend, 'event_bus', bus)
    monkeypatch.setattr(backend, 'resource_events', bridge)
    client = TestClient(backend.app)
    try:
        with client.websocket_connect('/ws/events') as socket:
            assert socket.receive_json()['type'] == 'state.snapshot'
            assert socket.receive_json()['payload']['approvals']['pending'] == []
            pending.append({'id':'request'})
            bus.publish('tool.approval_requested')
            event = socket.receive_json()
            assert event['type'] == 'resource.approvals'
            assert event['payload']['pending'] == [{'id':'request'}]
    finally: bridge.close()


def test_corrupt_saved_desktop_settings_do_not_abort_startup(backend, monkeypatch):
    path = backend.config.root / 'persistent_memories' / 'desktop_settings.json'
    path.parent.mkdir()
    path.write_text('{"avatar_geometry":{"width":0}}', encoding='utf-8')
    original = path.read_bytes()
    monkeypatch.setattr(backend, 'create_chat_service', lambda config: SimpleNamespace())
    monkeypatch.setattr(backend, 'SessionManager', lambda *args: SimpleNamespace(
        runtime_snapshot=lambda: {'runtime': {}}, close=lambda: None))
    monkeypatch.setattr(backend, 'Initiative', lambda session: None)
    with TestClient(backend.app):
        assert backend.state.avatar_geometry['width'] == 480
        assert not backend._desktop_settings_loaded
    assert path.read_bytes() == original


def test_surface_api_rejects_empty_bounds_and_noninteger_geometry(backend):
    client = TestClient(backend.app)
    command = backend.state.add_whiteboard('text', {'text': 'hello'})
    response = client.post('/api/surfaces/result', json={
        'surface': 'whiteboard', 'command_id': command, 'status': 'rendered', 'bounds': {}})
    assert response.status_code == 400
    assert client.patch('/api/surfaces/avatar', json={'width': 1}).status_code == 400
    assert client.patch('/api/surfaces/avatar', json={'x': True}).status_code == 422
    assert client.patch('/api/surfaces/whiteboard', json={'geometry': {'screen': -1}}).status_code == 400


def test_failed_session_start_closes_constructed_chat(backend,monkeypatch):
    closed=[]
    monkeypatch.setattr(backend,'create_chat_service',lambda config:SimpleNamespace(close=lambda:closed.append('chat')))
    def fail(*args): raise RuntimeError('session startup failed')
    monkeypatch.setattr(backend,'SessionManager',fail)
    with pytest.raises(RuntimeError,match='session startup failed'):
        with TestClient(backend.app): pass
    assert closed==['chat']


def test_setup_mode_keeps_settings_accessible_with_cors(backend,monkeypatch):
    backend.config.raw['desktop']={'setup_on_startup_error':True}
    (backend.config.root/'character_config.yaml').write_text('runtime:\n  provider: lm_studio\n')
    def fail(config): raise RuntimeError('llama-server missing')
    monkeypatch.setattr(backend,'create_chat_service',fail)
    with TestClient(backend.app) as client:
        assert client.get('/api/settings').status_code==200
        assert client.get('/api/status').json()['startup_error']=='llama-server missing'
        response=client.get('/api/voice/status',headers={'Origin':'http://127.0.0.1:5173'})
        assert response.status_code==503
        assert response.headers['access-control-allow-origin']=='http://127.0.0.1:5173'


def test_history_api_is_paginated_without_replaying_events(backend,monkeypatch):
    from process.app_core.persistence.conversation_store import ConversationStore
    from process.app_core.conversation.messages import ChatMessage
    store=ConversationStore(backend.config.root/'history.sqlite3',legacy=[ChatMessage('user',str(i)) for i in range(10)])
    monkeypatch.setattr(backend,'conversation_store',store)
    client=TestClient(backend.app)
    try:
        page=client.get('/api/chat/history?limit=3').json()
        assert [m['text'] for m in page['messages']]==['7','8','9']
        older=client.get('/api/chat/history',params={'limit':3,'before':page['before']}).json()
        assert [m['text'] for m in older['messages']]==['4','5','6']
        assert client.get('/api/chat/history?limit=1000').status_code==400
    finally: store.close()


def test_display_api_validates_payload_and_defaults_to_primary(backend):
    client = TestClient(backend.app)
    assert client.post('/api/displays', json=[{}]).status_code == 422
    assert client.post('/api/displays', json=[]).status_code == 400
    displays = [{'index': index, 'id': index + 10, 'label': f'Screen {index}', 'primary': index == 1,
                 'bounds': {'x': index * 1920, 'y': 0, 'width': 1920, 'height': 1080}, 'scaleFactor': 1}
                for index in range(2)]
    assert client.post('/api/displays', json=displays).status_code == 200
    assert backend.state.avatar_geometry['screen'] == 1
    assert client.patch('/api/surfaces/avatar', json={'screen': 9}).status_code == 400
    assert client.patch('/api/surfaces/avatar', json={'screen': 0}).status_code == 200
    # Removing the selected display safely falls back to the surviving primary.
    assert client.post('/api/displays', json=[{**displays[1], 'index': 0}]).status_code == 200
    assert backend.state.avatar_geometry['screen'] == 0
