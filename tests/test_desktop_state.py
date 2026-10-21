"""DesktopState is a thin facade over components (desktop/state.py): the state.snapshot payload Electron reads is unchanged,
each component emits only after releasing its own lock, the factory builds them over the data root's files without reading
or writing them before the lifespan loads them, and the desktop tools act on the state they are handed."""
import json
import threading

from process.app_core.configuration.paths import DataPaths
from process.app_core.desktop.state import SNAPSHOT_KEYS, DesktopState, get_desktop_state
from process.app_core.desktop.tools import DesktopServices, WhiteboardTool, WhiteboardWindowTool
from process.app_core.factory import create_desktop_state


class Emotion:
    def as_dict(self): return {'emotion': 'joy'}


def test_the_snapshot_payload_keeps_its_keys_order_and_defaults():
    snapshot = DesktopState().snapshot()
    assert tuple(snapshot) == SNAPSHOT_KEYS
    assert snapshot == {'emotion': None, 'speech': '', 'mic': True, 'audio': True, 'audio_volume': 1.0, 'sleep': False, 'effect': None,
        'last_effect': None, 'tools': [], 'actions': [], 'whiteboard': [], 'whiteboard_visible': False, 'whiteboard_clear': None,
        'whiteboard_pages': ['page-1'], 'whiteboard_page': 'page-1', 'avatar_geometry': {'x': 0, 'y': 0, 'width': 480, 'height': 720, 'screen': 0},
        'displays': [], 'whiteboard_geometry': {'x': 500, 'y': 100, 'width': 900, 'height': 700, 'screen': 0},
        'board_persistence_error': '', 'notifications': [], 'incoming': [], 'discord': {'running': False, 'ready': False, 'status': 'stopped'}}


def test_every_component_emits_after_releasing_its_lock(tmp_path):
    """A listener reads the whole state from another thread (as desktop_server's snapshot publisher and the bus listeners
    behind it do, under the session's locks): a component that emitted under its lock would block it."""
    state, events, blocked = DesktopState(), [], []
    state.configure_board_store(tmp_path / 'whiteboard.json')
    def listener(event, value):
        reader = threading.Thread(target=lambda: (state.snapshot(), state.board.result('none', timeout=0)), daemon=True)
        reader.start(); reader.join(2)
        (blocked if reader.is_alive() else events).append(event)
    state.subscribe(listener)
    state.set_speech('hi'); state.set_emotion(Emotion()); state.toggle_mic(); state.set_mic(True); state.toggle_audio()
    state.set_audio_volume(.5); state.set_sleep(True)
    command = state.add_whiteboard('text', {'text': 'note'}); state.surface_result('whiteboard', command, 'rendered')
    state.board_page('new_page'); state.set_whiteboard_surface(visible=True, geometry={'width': 800}); state.update_geometry('avatar', x=3)
    state.set_displays([{'index': 0, 'primary': True}]); effect = state.trigger_effect('rain'); state.surface_result('effect', effect, 'completed')
    state.stop_effect(); state.tool_finished('lookup', 'done', activity_id=state.tool_started('lookup', {}))
    state.record_action({'id': 'a', 'status': 'running'}); state.notify('tools', 'note')
    state.observe_input('message', 'hello', message_id='1'); state.set_discord({'running': True}); state.clear_whiteboard()
    assert not blocked
    assert events == ['speech', 'emotion', 'mic', 'mic', 'audio', 'audio_volume', 'sleep', 'whiteboard', 'surface_result', 'whiteboard_page',
        'whiteboard_surface', 'avatar_geometry', 'displays', 'effect', 'surface_result', 'effect', 'tool', 'tool', 'actions', 'notification',
        'input_received', 'discord', 'whiteboard_clear']
    state.set_speech('', seconds=0)


def test_the_board_saves_before_other_listeners_so_a_failed_save_is_in_that_snapshot(tmp_path):
    (tmp_path / 'blocked').write_text('a file where the board directory should be')
    state, seen = DesktopState(), []
    state.configure_board_store(tmp_path / 'blocked' / 'whiteboard.json')
    state.subscribe(lambda event, value: seen.append(state.snapshot()['board_persistence_error']))
    state.update_geometry('whiteboard', width=1000)  # SurfaceGeometry's change: the board keeps the window's geometry
    assert seen and seen[0]


def test_the_factory_builds_over_the_data_paths_and_saves_nothing_before_the_lifespan_loads(tmp_path):
    paths = DataPaths.at(tmp_path)
    assert (paths.whiteboard, paths.desktop_settings) == (tmp_path / 'persistent_memories' / 'whiteboard.json', tmp_path / 'persistent_memories' / 'desktop_settings.json')
    paths.memories.mkdir()
    saved = {'version': 1, 'pages': ['page-1'], 'page': 'page-1', 'visible': True, 'geometry': {'width': 1000},
             'commands': [{'id': 'kept', 'kind': 'text', 'payload': {'text': 'saved note', 'x': 40, 'y': 40}, 'status': 'rendered', 'error': '',
                           'page': 'page-1', 'bounds': {'x': 40, 'y': 40, 'width': 420, 'height': 120}}]}
    paths.whiteboard.write_text(json.dumps(saved), encoding='utf-8')
    paths.desktop_settings.write_text(json.dumps({'avatar_geometry': {'width': 300, 'screen': 1}}), encoding='utf-8')
    board, settings = paths.whiteboard.read_bytes(), paths.desktop_settings.read_bytes()
    state = create_desktop_state(paths)
    state.add_whiteboard('text', {'text': 'before loading'}); state.update_geometry('whiteboard', x=1)
    assert paths.whiteboard.read_bytes() == board and paths.desktop_settings.read_bytes() == settings  # never over unread user data
    state.board.load()
    assert state.geometry.load_settings() and state.geometry.settings_loaded
    snapshot = state.snapshot()
    assert [item['payload']['text'] for item in snapshot['whiteboard']] == ['saved note'] and snapshot['whiteboard'][0]['status'] == 'queued'
    assert snapshot['whiteboard_geometry']['width'] == 1000 and snapshot['avatar_geometry']['width'] == 300
    state.set_displays([{'index': 0, 'primary': True}, {'index': 1, 'primary': False}])
    assert state.geometry.avatar()['screen'] == 1  # the saved display wins over the primary on the first report
    state.update_geometry('avatar', x=12); state.save_avatar_geometry()
    assert json.loads(paths.desktop_settings.read_text(encoding='utf-8')) == {'avatar_geometry': {'x': 12, 'y': 0, 'width': 300, 'height': 720, 'screen': 1}}
    state.add_whiteboard('text', {'text': 'after loading'})
    assert [item['payload']['text'] for item in json.loads(paths.whiteboard.read_text(encoding='utf-8'))['commands']] == ['saved note', 'after loading']
    assert sorted(path.name for path in paths.memories.iterdir()) == ['desktop_settings.json', 'whiteboard.json']


def test_a_damaged_settings_file_is_ignored_and_kept(tmp_path):
    paths = DataPaths.at(tmp_path)
    paths.memories.mkdir()
    paths.desktop_settings.write_text('{"avatar_geometry": {"width": 0}}', encoding='utf-8')
    state = create_desktop_state(paths)
    assert not state.geometry.load_settings() and not state.geometry.settings_loaded
    state.set_displays([{'index': 0, 'primary': False}, {'index': 1, 'primary': True}])
    assert state.geometry.avatar() == {'x': 0, 'y': 0, 'width': 480, 'height': 720, 'screen': 1}  # nothing restored: the primary display
    assert paths.desktop_settings.read_text(encoding='utf-8') == '{"avatar_geometry": {"width": 0}}'


def test_desktop_tools_change_the_state_they_are_handed_and_two_states_stay_apart():
    first, second = DesktopState(), DesktopState()
    second.subscribe(lambda event, value: second.surface_result('whiteboard', value['id'], 'rendered') if event == 'whiteboard' else None)  # the renderer
    assert "'status': 'rendered'" in WhiteboardTool(DesktopServices.of(second)).execute(action='text', text='hello')
    WhiteboardWindowTool(DesktopServices.of(second)).execute(width=1200)
    assert first.snapshot()['whiteboard'] == [] and first.snapshot()['whiteboard_geometry']['width'] == 900
    assert [item['payload']['text'] for item in second.snapshot()['whiteboard']] == ['hello'] and second.snapshot()['whiteboard_geometry']['width'] == 1200
    assert get_desktop_state() not in (first, second)


def test_a_lone_surrogate_from_a_tool_call_never_stops_the_board_saving(tmp_path):
    """Regression: json.loads turns a model's bare "\\ud83d" escape into a lone surrogate, which UTF-8 cannot encode; the save
    failed silently (not an OSError) and every later save failed with it. Saves escape non-ASCII, and failures are reported."""
    state = create_desktop_state(DataPaths.at(tmp_path))
    state.board.load()  # as the lifespan does: nothing is saved before the file has been read
    state.add_whiteboard('text', {'text': 'Note \ud83d'})
    state.add_whiteboard('text', {'text': 'Later note'})
    saved = json.loads(DataPaths.at(tmp_path).whiteboard.read_text(encoding='utf-8'))
    assert [command['payload']['text'] for command in saved['commands']] == ['Note \ud83d', 'Later note']
    assert state.snapshot()['board_persistence_error'] == ''
