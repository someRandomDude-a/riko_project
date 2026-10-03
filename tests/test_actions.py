import pytest

from process.app_core.runtime.actions import ActionController


def test_replacement_and_shutdown():
    controller = ActionController()
    first = controller.start('emotion', {'primary': 'joy'})
    second = controller.start('emotion', {'primary': 'calm'})
    assert first.status == 'cancelled'
    assert [a['id'] for a in controller.active()] == [second.id]
    controller.close()
    assert second.status == 'cancelled'
    assert controller.active() == []
    with pytest.raises(RuntimeError):
        controller.start('gesture')


def test_completion_is_terminal_and_payload_is_detached():
    controller = ActionController()
    payload = {'weights': {'happy': 0.5}}
    action = controller.start('expression', payload, 60)
    payload['weights']['happy'] = 1
    assert action.as_dict()['payload']['weights']['happy'] == 0.5
    assert controller.complete(action.id)
    assert not controller.cancel(action.id)
    assert not controller.complete(action.id)
    assert controller.active() == []


@pytest.mark.parametrize('duration', [-1, float('inf'), float('nan')])
def test_invalid_duration(duration):
    with pytest.raises(ValueError):
        ActionController().start('gesture', duration=duration)


def test_gesture_validation_and_replacement():
    controller = ActionController()
    with pytest.raises(ValueError):
        controller.gesture('unknown')
    with pytest.raises(ValueError):
        controller.gesture('wave', intensity=2)
    first = controller.gesture('nod')
    second = controller.gesture('wave', intensity=0)
    assert first.status == 'cancelled'
    assert second.payload['intensity'] == 0
    controller.close()
