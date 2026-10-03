import io
from pathlib import Path

from PIL import Image

from process.app_core.desktop.state import DesktopState
from process.app_core.desktop.whiteboard_image import WhiteboardImages, board_revision
from process.app_core.events.bus import EventBus


def test_whiteboard_image_changes_are_source_driven_and_ignore_unrelated_state(tmp_path):
    state, images, bus, events = DesktopState(), WhiteboardImages(), EventBus(), []
    bus.subscribe(events.append)
    state.subscribe(lambda kind, value: images.observe(state.snapshot(), bus) if kind.startswith('whiteboard') else None)
    state.add_whiteboard('text', {'text':'Hello **world**', 'width':420, 'size':18})
    assert events[-1].type == 'whiteboard.changed'
    total = len(events)
    images.observe(state.snapshot(), bus)
    assert len(events) == total
    with Image.open(io.BytesIO(images.image(state.snapshot(), tmp_path))) as image:
        assert image.format == 'PNG' and image.size == (1280, 900)
    state.clear_whiteboard()
    assert len(events) == total + 1


def test_rich_renderer_captures_are_scoped_to_revision_and_deduplicated(tmp_path):
    state, images, bus, events = DesktopState(), WhiteboardImages(), EventBus(), []
    bus.subscribe(events.append)
    revision = board_revision(state.snapshot())
    output = io.BytesIO(); Image.new('RGB', (20, 20), 'white').save(output, 'PNG')
    png = output.getvalue()
    assert images.capture(revision, png, state.snapshot(), bus)
    assert images.image(state.snapshot(), tmp_path) == png
    assert len(events) == 1
    assert images.capture(revision, png, state.snapshot(), bus)
    assert len(events) == 1
    state.add_whiteboard('draw', {'points': [[1,1],[2,2]], 'size':3})
    assert not images.capture(revision, png, state.snapshot(), bus)
    assert images.image(state.snapshot(), tmp_path) != png
