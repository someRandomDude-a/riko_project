import pytest
from process.app_core.desktop.state import DesktopState
from process.app_core.desktop.media import resolve_media
from process.app_core.desktop.tools import WhiteboardTool, EffectTool
from process.app_core.tools.registry import local_definition


def test_render_acknowledgements_reject_stale_commands():
    state = DesktopState()
    command = state.add_whiteboard('text', {'text': 'Hi'})
    assert state.snapshot()['whiteboard'][0]['status'] == 'queued'
    assert state.surface_result('whiteboard', command, 'rendered')
    assert state.snapshot()['whiteboard'][0]['status'] == 'rendered'
    clear = state.clear_whiteboard()
    assert not state.surface_result('whiteboard', command, 'rendered')
    assert state.surface_result('whiteboard', clear, 'rendered')
    first = state.trigger_effect('first')
    second = state.trigger_effect('second')
    assert not state.surface_result('effect', first, 'completed')
    assert state.surface_result('effect', second, 'playing')
    assert state.surface_result('effect', second, 'error', 'Codec unsupported')
    assert state.active_effect is None
    assert state.last_effect['error'] == 'Codec unsupported'


def test_media_access_is_scoped_and_rejects_nonmedia(tmp_path):
    folder = tmp_path / 'character_files'; folder.mkdir()
    image = folder / 'sample.png'; image.write_bytes(b'example')
    assert resolve_media(tmp_path, str(image)) == image
    private = tmp_path / 'private.png'; private.write_bytes(b'secret')
    with pytest.raises(ValueError): resolve_media(tmp_path, str(private))
    with pytest.raises(ValueError): resolve_media(tmp_path, 'character_files/../private.png')
    script = folder / 'code.html'; script.write_text('script')
    with pytest.raises(ValueError): resolve_media(tmp_path, str(script))


def test_whiteboard_tool_validates_and_returns_queued_id(tmp_path):
    tool = WhiteboardTool(); tool.state = DesktopState()
    result = tool.execute(action='draw', points=[[0, 0], {'x': 10, 'y': 20}])
    assert 'queued' in result
    assert tool.state.whiteboard[0].payload['points'] == [[0, 0], [10, 20]]
    with pytest.raises(ValueError): tool.execute(action='draw', points=[[float('nan'), 0]])
    with pytest.raises(ValueError): tool.execute(action='text', text='hi', color='not-a-color')
    schema = local_definition(tool)['inputSchema']
    assert schema['properties']['x']['anyOf'][0]['type'] == 'integer'
    assert schema['properties']['points']['anyOf'][0]['type'] == 'array'


@pytest.mark.parametrize('kwargs', [{'size': None}, {'size': True}, {'width': float('nan')}, {'text': 5}])
def test_whiteboard_invalid_arguments_are_rejected_before_queueing(kwargs):
    tool = WhiteboardTool(); tool.state = DesktopState()
    with pytest.raises(ValueError): tool.execute(action='text', **{'text': 'hello', **kwargs})
    assert tool.state.whiteboard == []


def test_effect_tool_requires_an_approved_video(tmp_path):
    folder = tmp_path / 'effects' / 'greenscreens'; folder.mkdir(parents=True)
    (folder / 'sample.webm').write_bytes(b'video')
    tool = EffectTool(); tool.state = DesktopState()
    tool.state.media_resolver = lambda path: resolve_media(tmp_path, path)
    tool.state.effects_directory = 'effects/greenscreens'
    assert 'queued' in tool.execute(name='sample.webm')
    assert tool.state.active_effect['duration'] == 8
    with pytest.raises(ValueError): tool.execute(name='missing.webm')
    with pytest.raises(ValueError): tool.execute(name='sample.webm', duration=float('nan'))
    tool.execute(action='stop')
    assert tool.state.last_effect['status'] == 'cancelled'


def test_board_auto_layout_measurements_and_pages():
    state = DesktopState()
    first = state.add_whiteboard('text', {'text': '# Heading', 'x': None, 'y': None, 'width': 420})
    second = state.add_whiteboard('text', {'text': 'Second', 'x': None, 'y': None, 'width': 420})
    assert state.whiteboard[1].bounds['y'] > state.whiteboard[0].bounds['y']
    state.surface_result('whiteboard', first, 'rendered', bounds={'x': 999, 'y': 999, 'width': 436, 'height': 400})
    assert state.whiteboard[0].bounds['x'] == 40 # local/user coordinates cannot alter model layout
    assert state.whiteboard[1].bounds['y'] >= 464
    assert state.board_result(first, timeout=0)['bounds']['height'] == 400
    state.board_page('new_page')
    state.add_whiteboard('text', {'text': 'New page', 'x': None, 'y': None})
    assert state.whiteboard[-1].page == 'page-2'
    assert state.whiteboard[-1].bounds['y'] == 40
    state.board_page('previous_page')
    assert state.whiteboard_page == 'page-1'
    assert state.board_result(second, timeout=0)['page'] == 'page-1'
