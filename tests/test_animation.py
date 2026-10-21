import json
import struct
import threading
from types import SimpleNamespace

import pytest

from process.app_core.runtime.actions import ActionController
from process.app_core.animation.library import AnimationLibrary, inspect_vrma, metadata, validate_settings
from process.app_core.animation.policy import AnimationState, eligible_intents, motion_intent
from process.app_core.animation.runtime import AnimationRuntime
from process.app_core.desktop.state import DesktopState
from process.app_core.kernel.turns import RuntimeStatus
from process.app_core.emotion.julia import JuliaEmotionEngine


def glb(document):
    data = json.dumps(document).encode()
    data += b' ' * (-len(data) % 4)
    return struct.pack('<4sIIII', b'glTF', 2, 20 + len(data), len(data), 0x4E4F534A) + data


def vrma_document():
    return {'asset': {'version': '2.0'}, 'nodes': [{}], 'animations': [{}],
            'extensions': {'VRMC_vrm_animation': {'humanoid': {'humanBones': {'head': {'node': 0}}}}}}


def pose_file(tmp_path):
    source = tmp_path / 'test.pose.json'
    source.write_text(json.dumps({'version': 1, 'bones': {'head': [0, .1, 0]}, 'expressions': {}}))
    return source


def test_import_preserves_original_and_deduplicates(tmp_path):
    source = pose_file(tmp_path)
    original = source.read_bytes()
    library = AnimationLibrary(tmp_path)
    entry = library.import_file(source, {'states': ['idle'], 'license': 'test'})
    assert library.path(entry['id']).read_bytes() == original
    assert library.import_file(source)['id'] == entry['id']
    assert len(library.list()) == 1
    library.update(entry['id'], {'mask': ['head'], 'speed': .5})
    restored = AnimationLibrary(tmp_path)
    assert restored.get(entry['id'])['speed'] == .5
    assert source.read_bytes() == original
    entry['states'].append('sleeping')
    assert restored.get(entry['id'])['states'] == ['idle']


def test_vrma_import_and_external_resources_rejected(tmp_path):
    document = vrma_document()
    source = tmp_path / 'clip.vrma'
    source.write_bytes(glb(document))
    assert AnimationLibrary(tmp_path).import_file(source)['bones'] == ['head']
    document['buffers'] = [{'uri': 'https://example.org/secret', 'byteLength': 8}]
    with pytest.raises(ValueError, match='self-contained'): inspect_vrma(glb(document))
    document.pop('buffers')
    document['extensions']['VRMC_vrm_animation']['humanoid']['humanBones']['head']['node'] = 3
    with pytest.raises(ValueError, match='invalid node'): inspect_vrma(glb(document))
    with pytest.raises(ValueError): inspect_vrma(b'bad')


@pytest.mark.parametrize('options', [{'states': ['unknown']}, {'mask': ['tail']}, {'speed': float('nan')}, {'loop': 1}, {'layer': 'other'}, {'bogus': True}])
def test_invalid_metadata(options):
    with pytest.raises(ValueError): metadata(options)


@pytest.mark.parametrize('options', [{'enabled': 1}, {'walk_speed': -1}, {'min_confidence': float('inf')}, {'transition_seconds': 0}])
def test_invalid_animation_settings(options):
    with pytest.raises(ValueError): validate_settings(options)


def test_manifest_traversal_rejected_without_overwriting(tmp_path):
    library = AnimationLibrary(tmp_path)
    entry = library.import_file(pose_file(tmp_path))
    entry['file'] = '../../outside.pose.json'
    raw = json.dumps({'version': 1, 'entries': [entry]})
    library.manifest.write_text(raw)
    with pytest.raises(ValueError): AnimationLibrary(tmp_path)
    assert library.manifest.read_text() == raw


def test_eligibility_and_low_confidence_fallback(tmp_path):
    library = AnimationLibrary(tmp_path)
    entry = library.import_file(pose_file(tmp_path), {'states': ['idle'], 'emotions': ['joy']})
    state = AnimationState(4, 'idle', 'joy', bones=['head'])
    choices = eligible_intents(state, library.list())
    assert choices[0]['id'] == entry['id']
    settings = validate_settings({})
    assert motion_intent(state, choices, settings, {'choice': 'builtin-playful', 'confidence': .8}).source == 'julia_1'
    assert motion_intent(state, choices, settings, {'choice': 'builtin-playful', 'confidence': .1}).intent_id == entry['id']
    state.bones = []
    assert all(c['id'] != entry['id'] for c in eligible_intents(state, library.list()))


def runtime(tmp_path):
    session = SimpleNamespace(config=SimpleNamespace(root=tmp_path, raw={}), chat=SimpleNamespace(),
        state=DesktopState(), actions=ActionController(), status=lambda: RuntimeStatus(listening=True, voice_status='ready'))
    service = AnimationRuntime(session, start=False)
    service.report_capabilities(['head'], [])
    return service, session


def test_animation_worker_parks_until_source_change(tmp_path):
    service, session = runtime(tmp_path)
    calls = []
    stepped = threading.Event()
    def step(): calls.append(True); stepped.set()
    service.step = step
    service.thread.start()
    try:
        assert stepped.wait(1)
        stepped.clear()
        # No elapsed deadline exists in this test; only a state mutation wakes it.
        service.report_capabilities(['head','hips'], [])
        assert stepped.wait(1)
        assert len(calls) == 2
    finally:
        service.close(); session.actions.close(); service.thread.join(1)
    assert not service.thread.is_alive()


def test_runtime_interaction_priority_and_renderer_failure_fallback(tmp_path):
    service, session = runtime(tmp_path)
    try:
        entry = service.library.import_file(pose_file(tmp_path), {'states': ['idle']})
        service.step()
        assert service.current['intent_id'] == entry['id']
        action = session.actions.active()[0]
        session.actions.cancel(action['id'])
        service.renderer_error(action, 'invalid clip')
        service.step()
        assert service.current['intent_id'] == 'builtin-idle'
        service.interact('hold')
        session._generation_active = True
        service.step()
        assert service.state.mode == 'held'
        service.interact('release')
        service.step()
        assert service.state.mode == 'settling'
    finally: service.close(); session.actions.close()


def test_pickup_target_is_retained_and_ragdoll_recovers(tmp_path):
    service, session = runtime(tmp_path)
    try:
        target = {'bone': 'leftHand', 'region': 'arms'}
        service.interact('hold', target=target)
        service.interact('drag', target={'bone': 'head', 'region': 'head'})
        assert service.interaction_state['target'] == target
        service.interact('ragdoll', target=target)
        assert service.interaction_state['ragdoll']
        service.interact('recover', target=target)
        assert not service.interaction_state['held']
        assert not service.interaction_state['ragdoll']
    finally: service.close(); session.actions.close()


def test_hover_and_spring_targets_do_not_start_pickup(tmp_path):
    service, session = runtime(tmp_path)
    try:
        target = {'bone': 'root/0/1/2', 'region': 'spring'}
        for kind in ('hover', 'hoverDelayed', 'leave'):
            service.interact(kind, target=target)
            assert not service.interaction_state['held']
        with pytest.raises(ValueError): service.interact('hover', target={'bone': '../model', 'region': 'spring'})
    finally: service.close(); session.actions.close()


def test_late_julia_result_cannot_override_new_state(tmp_path, monkeypatch):
    from concurrent.futures import Future
    service, session = runtime(tmp_path)
    future = Future()
    service.selector = lambda *args: None
    monkeypatch.setattr(service.executor, 'submit', lambda *args: future)
    try:
        entry = service.library.import_file(pose_file(tmp_path), {'states': ['idle']})
        service.step()
        assert service.pending
        service.interact('hold')
        future.set_result({'choice': entry['id'], 'confidence': 1})
        service.step()
        assert service.current['intent_id'] == 'builtin-held'
    finally: service.close(); session.actions.close()


def test_expired_julia_result_keeps_rule_fallback(tmp_path, monkeypatch):
    from concurrent.futures import Future
    service, session = runtime(tmp_path)
    future = Future()
    service.selector = lambda *args: None
    monkeypatch.setattr(service.executor, 'submit', lambda *args: future)
    now = [100.0]
    monkeypatch.setattr('process.app_core.animation.runtime.time.monotonic', lambda: now[0])
    try:
        entry = service.library.import_file(pose_file(tmp_path), {'states': ['idle']})
        service.step()
        now[0] += 2
        service.step()
        assert service.pending['expired']
        future.set_result({'choice': 'builtin-idle', 'confidence': 1})
        service.step()
        assert service.current['intent_id'] == entry['id']
        assert service.pending is None
    finally: service.close(); session.actions.close()


def test_walk_clamps_and_user_hold_cancels(tmp_path):
    service, session = runtime(tmp_path)
    session.state.set_displays([{'index': 0, 'primary': True, 'bounds': {'width': 1920, 'height': 1080}}])
    try:
        action = service.walk_to(10000, -50)
        assert action.payload['target']['y'] == 0
        assert action.payload['target']['x'] <= 1920
        service.interact('hold')
        assert not service.movement
        assert not any(a['kind'] == 'motion.locomotion' for a in session.actions.active())
        with pytest.raises(ValueError, match='holding'): service.walk_to(20, 20)
    finally: service.close(); session.actions.close()


def test_julia_reuses_model_and_emotion_questions_remain_intact():
    engine = JuliaEmotionEngine(None)
    assert 'emotion' in engine._questions()
    calls = []
    engine._model = SimpleNamespace(predict=lambda **args: calls.append(args) or {'answers': {'intent': {'choice': 'idle', 'probabilities': {'idle': .8}}}})
    assert engine.choose_motion({'mode': 'idle'}, [{'id': 'idle', 'description': 'rest'}]) == {'choice': 'idle', 'confidence': .8}
    assert calls[0]['questions']['intent']['criteria'] == {'idle': 'rest'}
