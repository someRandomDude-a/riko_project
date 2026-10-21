import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from process.app_core.conversation.chat import ChatService
from process.app_core.desktop.effects import EffectLibrary
from process.app_core.desktop.state import DesktopState
from process.app_core.desktop.tools import iter_tools
from process.app_core.events.stream import stream_events
from process.app_core.events.bus import EventBus
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.tools.registry import RegisteredTool, ToolRegistry


def test_tool_timeout_returns_without_waiting_for_worker():
    release, started, finished = threading.Event(), threading.Event(), threading.Event()
    def blocked(_):
        started.set()
        try: release.wait(2)
        finally: finished.set()
        return 'late result'
    registry = ToolRegistry(timeout_seconds=.02)
    registry.tools['blocked'] = RegisteredTool('blocked', '', {}, blocked)
    before = time.monotonic()
    try:
        result = registry.execute('blocked', {})
        assert started.is_set()
        assert result.is_error and 'may still finish' in result.content
        assert time.monotonic() - before < .5
        assert not finished.is_set()
    finally:
        release.set()
        assert finished.wait(1)


@pytest.mark.parametrize('text', ['# Heading\n\nBody', '[label](https://example.com)', '\\[x^2\\]', '```python\nprint(1)\n```'])
def test_chat_preserves_model_markdown_verbatim(text):
    provider = SimpleNamespace(generate=lambda *a, **k: ModelResponse(ChatMessage('assistant', text)))
    service = ChatService(provider, system_prompt='test')
    assert service.respond('hi').message.content == text
    assert service.history[-1].content == text


@pytest.mark.parametrize('data', [{}, ['bad record'], [{'role': 'assistant', 'content': None}]])
def test_invalid_history_shape_does_not_crash_or_overwrite(tmp_path, data):
    path = tmp_path / 'history.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    before = path.read_bytes()
    service = ChatService(None, system_prompt='', history_file=path)
    assert service.history == []
    assert path.read_bytes() == before


def test_effect_discovery_aliases_empty_rules_and_supported_extensions(tmp_path):
    for name in ['sadness_rain.mp4', 'sad_cloud.webm', 'happy_stars.m4v', 'unsupported.mkv']:
        (tmp_path / name).touch()
    (tmp_path / 'folder.mp4').mkdir()
    (tmp_path / 'effects.rules.json').write_text('[]', encoding='utf-8')
    library = EffectLibrary(tmp_path)
    assert len(library.assets['sadness']) == 2
    assert library.assets['joy'][0].suffix == '.m4v'
    assert library.rules == []
    assert sum(map(len, library.assets.values())) == 3


def test_desktop_tools_do_not_register_builtins_twice():
    assert {tool.TOOL_NAME for tool in iter_tools()} == {
        'whiteboard', 'move_avatar', 'move_whiteboard', 'visual_effect', 'avatar_gesture'}


def test_idle_websocket_disconnect_unsubscribes():
    async def run():
        bus = EventBus()
        class Socket:
            async def accept(self): pass
            async def send_json(self, message):
                assert message['type'] == 'state.snapshot'
            async def receive(self):
                return {'type': 'websocket.disconnect'}
        await asyncio.wait_for(stream_events(Socket(), bus, lambda: {}), .5)
        assert bus._listeners == []
    asyncio.run(run())


def test_websocket_cancellation_unsubscribes_and_stops_workers():
    async def run():
        bus = EventBus()
        ready = asyncio.Event()
        class Socket:
            async def accept(self): pass
            async def send_json(self, message): ready.set()
            async def receive(self): await asyncio.Event().wait()
        worker = asyncio.create_task(stream_events(Socket(), bus, lambda: {}))
        await ready.wait()
        worker.cancel()
        with pytest.raises(asyncio.CancelledError): await worker
        assert bus._listeners == []
    asyncio.run(run())


def test_speech_expiry_notifies_without_another_state_mutation():
    state = DesktopState()
    expired = threading.Event()
    state.subscribe(lambda kind, value: expired.set() if kind == 'speech' and value == '' else None)
    state.set_speech('Temporary bubble', seconds=.02)
    assert expired.wait(1)
    assert state.snapshot()['speech'] == ''


def test_stale_speech_timer_cannot_clear_a_new_bubble():
    state = DesktopState()
    state.set_speech('old', seconds=10)
    deadline = state.presence._until
    state.set_speech('new', seconds=20)
    state.presence._expire(deadline)  # the old bubble's timer, already firing when the new bubble replaced it
    assert state.snapshot()['speech'] == 'new'
    state.set_speech('', seconds=0)


def test_snapshot_and_input_payloads_do_not_share_mutable_board_state():
    state = DesktopState()
    payload = {'text': 'original', 'x': None}
    state.add_whiteboard('text', payload)
    assert 'auto_place' not in payload
    snapshot = state.snapshot()
    snapshot['whiteboard'][0]['payload']['text'] = 'changed externally'
    assert state.snapshot()['whiteboard'][0]['payload']['text'] == 'original'


@pytest.mark.parametrize('bounds', [{}, {'x': 0}, {'x': 0, 'y': 0, 'width': float('nan'), 'height': 1}, {'x': 0, 'y': 0, 'width': 1, 'height': 0}])
def test_invalid_surface_bounds_raise_value_error(bounds):
    state = DesktopState()
    command = state.add_whiteboard('text', {'text': 'hi'})
    with pytest.raises(ValueError): state.surface_result('whiteboard', command, 'rendered', bounds=bounds)
    assert state.snapshot()['whiteboard'][0]['status'] == 'queued'


def test_stale_coordinates_do_not_trigger_repeated_ack_events():
    state = DesktopState()
    command = state.add_whiteboard('text', {'text': 'hi'})
    events = []
    state.subscribe(lambda *args: events.append(args))
    bounds = {'x': 999, 'y': 999, 'width': 430, 'height': 120}
    state.surface_result('whiteboard', command, 'rendered', bounds=bounds)
    state.surface_result('whiteboard', command, 'rendered', bounds=bounds)
    assert len(events) == 1


@pytest.mark.parametrize('geometry', [{'screen': -1}, {'width': 1}, {'width': True}, {'x': 1.5}, {'unknown': 1}, {'screen': 2}])
def test_surface_geometry_is_validated_for_every_caller(geometry):
    state = DesktopState()
    state.set_displays([{'index': 0, 'primary': True}])
    before = state.snapshot()['avatar_geometry']
    with pytest.raises(ValueError): state.update_geometry('avatar', **geometry)
    assert state.snapshot()['avatar_geometry'] == before
