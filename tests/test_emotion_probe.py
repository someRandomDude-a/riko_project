from types import SimpleNamespace
import threading

import pytest
import torch

from process.app_core.emotion.probe import (
    EmotionProbe, ProbeConfig, build_network, held_out, identity_key,
    latent_features, metrics, qualified,
)
from process.app_core.emotion.julia import JuliaEmotionEngine
from process.app_core.emotion.models import EmotionEvent, EmotionState, EMOTIONS


def small_config(**options):
    return ProbeConfig.from_raw({'hidden_units': [32, 16], 'rank': 8,
        'min_samples': 32, 'retrain_every': 32, 'epochs': 2, **options})


def teacher():
    engine = JuliaEmotionEngine(None)
    engine._load_attempted = True
    engine._model = SimpleNamespace(predict=lambda **kw: {'answers': {
        'emotion': {'choice': 'love', 'max_probability': .9},
        'intensity': {'score': 2}, 'valence': {'score': 3}}})
    return engine


def test_teacher_is_stateless_genuine_and_maps_affection():
    engine = teacher()
    engine.start_turn('unchanged')
    engine._window.append(('user', 'existing'))
    events = []
    engine.on_event = events.append
    label = engine.label_probe('user: hi\nassistant: hello')
    assert label.primary == 'affection'
    assert label.source == 'julia_1'
    assert engine.turn_id == 'unchanged'
    assert engine._transcript() == 'user: existing'
    assert not events


def test_fallback_and_invalid_teacher_never_become_labels():
    engine = JuliaEmotionEngine(None)
    engine._load_attempted = True
    assert engine.label_probe('yay') is None
    engine._model = SimpleNamespace(predict=lambda **kw: {})
    assert engine.label_probe('yay') is None
    engine._model.predict = lambda **kw: {'emotion': {'choice': 'joy'}, 'intensity': {'score': float('nan')}, 'valence': {'score': 2}}
    assert engine.label_probe('yay') is None


@pytest.mark.parametrize('raw', [{'interval_tokens': 0}, {'rank': True}, {'min_agreement': float('nan')},
    {'hidden_units': [100000, 5]}, {'enabled': 'yes'}, {'unknown': 1}, {'min_samples': 1}])
def test_probe_config_rejects_invalid_values(raw):
    with pytest.raises(ValueError): ProbeConfig.from_raw(raw)


def test_default_width_has_tens_of_thousands_of_units_not_dense_connections():
    config = ProbeConfig()
    assert sum(config.hidden_units) == 24576
    network = build_network(config)
    assert sum(p.numel() for p in network.parameters()) < 2_000_000


def test_latents_are_detached_and_nonfinite_rejected():
    hidden = torch.arange(2560., requires_grad=True).reshape(1, 1, -1)
    result = latent_features(hidden)
    assert result.shape == (256,)
    assert not result.requires_grad
    assert result[0] == 4.5
    with pytest.raises(ValueError): latent_features(torch.full((256,), float('nan')))


def test_turn_group_split_and_exact_model_fingerprints():
    assert held_out('same-turn') == held_out('same-turn')
    assert identity_key({'model': 'a', 'revision': 1}) != identity_key({'model': 'a', 'revision': 2})
    assert identity_key({'model': 'a', 'revision': 1}) == identity_key({'revision': 1, 'model': 'a'})


def test_quality_gate_rejects_neutral_only_and_regression_error():
    config = small_config()
    good = dict(agreement=.9, macro_f1=.9, rmse=.1, classes=3, validation_samples=16)
    assert qualified(good, config)
    assert not qualified({**good, 'classes': 1}, config)
    assert not qualified({**good, 'rmse': .4}, config)
    assert not qualified({**good, 'validation_samples': 15}, config)


def test_capture_is_bounded_and_cancelled_work_is_not_queued(tmp_path):
    probe = EmotionProbe(tmp_path, {'model': 'one'}, teacher(), small_config(), idle=lambda: False)
    try:
        probe.capture(torch.ones(256), 'private text', 'one', cancelled=lambda: True)
        assert not probe.pending and not probe.samples
        probe.activate('new')
        assert probe.active_group == 'new'
        assert not probe.ready
    finally: probe.close()
    saved = torch.load(probe.path, weights_only=True)
    assert 'private text' not in repr(saved)


def test_prediction_read_only_confidence_and_cancel_gate(tmp_path):
    probe = EmotionProbe(tmp_path, {'model': 'one'}, teacher(), small_config(min_confidence=.5))
    try:
        probe.network = build_network(probe.config).eval().requires_grad_(False)
        with torch.no_grad():
            for parameter in probe.network.parameters(): parameter.zero_()
            probe.network[-1].bias[EMOTIONS.index('joy')] = 10
        state = probe._predict(torch.zeros(256), 'turn', lambda: False)
        assert state.primary == 'joy' and state.source == 'latent_probe'
        assert state.turn_id == 'turn'
        assert all(parameter.grad is None for parameter in probe.network.parameters())
        assert probe._predict(torch.zeros(256), 'turn', lambda: True) is None
    finally: probe.close()


def test_unqualified_artifact_is_not_loaded_as_ready(tmp_path):
    probe = EmotionProbe(tmp_path, {'model': 'one'}, teacher(), small_config())
    probe.close()
    saved = torch.load(probe.path, weights_only=True)
    saved['weights'] = build_network(probe.config).state_dict()
    saved['validation'] = dict(agreement=1, macro_f1=1, rmse=0, classes=1, validation_samples=32)
    torch.save(saved, probe.path)
    restored = EmotionProbe(tmp_path, {'model': 'one'}, teacher(), small_config())
    other = EmotionProbe(tmp_path, {'model': 'two'}, teacher(), small_config())
    try:
        assert not restored.ready and not other.ready
        assert restored.key != other.key
    finally:
        restored.close()
        other.close()


def test_training_splits_turns_and_persists_no_text(tmp_path):
    probe = EmotionProbe(tmp_path, {'model': 'train'}, teacher(), small_config(min_agreement=0, min_macro_f1=0, max_rmse=1))
    try:
        groups = {'train': [], 'valid': []}
        for i in range(200):
            group = f'turn-{i}'
            groups['valid' if held_out(group) else 'train'].append(group)
        for i, group in enumerate(groups['train'][:32] + groups['valid'][:18]):
            probe.samples.append((torch.full((256,), float(i % 3)), [i % 3, .5, 0., .5], group))
        probe._train()
        assert probe.path.exists()
        assert probe.status()['training'] is False
        saved = torch.load(probe.path, weights_only=True)
        assert saved['validation']['classes'] == 3
        assert saved['validation']['validation_samples'] == 18
        assert set(groups['train']).isdisjoint(groups['valid'])
    finally: probe.close()


def test_probe_is_cpu_even_under_different_torch_default_device():
    with torch.device('meta'):
        network = build_network(small_config())
    assert all(parameter.device.type == 'cpu' for parameter in network.parameters())


def test_low_confidence_restores_aligned_julia_fallback(tmp_path):
    published, finished = [], threading.Event()
    engine = teacher()
    engine.start_turn('teacher-unrelated-turn')
    def fallback(state):
        published.append(state)
        finished.set()
    probe = EmotionProbe(tmp_path, {'model': 'fallback'}, engine, small_config(),
        idle=lambda: False, on_fallback=fallback)
    try:
        probe.activate('active-turn')
        probe.prediction_turn_id = 'active-turn'
        probe.network = build_network(probe.config).eval().requires_grad_(False)
        with torch.no_grad():
            for parameter in probe.network.parameters(): parameter.zero_()
        probe.capture(torch.ones(256), 'user: hi\nassistant: hello', 'active-turn')
        assert finished.wait(5)
        assert probe.prediction_turn_id is None
        assert published[0].turn_id == 'active-turn'
        assert published[0].source == 'julia_1' and published[0].primary == 'affection'
        probe.publish_teacher(EmotionEvent('assistant', 'hello', EmotionState(turn_id='active-turn')), published.append)
        assert len(published) == 2
    finally: probe.close()


def test_late_prediction_and_teacher_event_cannot_override_new_turn(tmp_path):
    published = []
    probe = EmotionProbe(tmp_path, {'model': 'stale'}, teacher(), small_config(), on_prediction=published.append)
    try:
        probe.activate('new')
        probe._publish(EmotionState(turn_id='old'), 'old', lambda: False)
        probe.publish_teacher(EmotionEvent('assistant', '', EmotionState(turn_id='old')), published.append)
        assert not published and probe.prediction_turn_id is None
        probe._publish(EmotionState(source='latent_probe', turn_id='new'), 'new', lambda: False)
        probe.publish_teacher(EmotionEvent('assistant', '', EmotionState(turn_id='new')), published.append)
        assert len(published) == 1 and probe.prediction_turn_id == 'new'
        probe.activate('next')
        probe._publish(None, 'new', lambda: False)
        assert probe.prediction_turn_id is None
    finally: probe.close()


def test_teacher_in_flight_at_turn_change_is_never_published(tmp_path):
    engine = teacher()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    published = []
    original = engine.label_probe
    def delayed(transcript):
        entered.set()
        assert release.wait(5)
        try: return original(transcript)
        finally: finished.set()
    engine.label_probe = delayed
    probe = EmotionProbe(tmp_path, {'model': 'cancel'}, engine, small_config(), idle=lambda: False,
        on_prediction=published.append, on_fallback=published.append)
    try:
        probe.activate('old')
        probe.capture(torch.ones(256), 'old text', 'old', cancelled=lambda: probe.active_group != 'old')
        assert entered.wait(5)
        probe.activate('new')
        release.set()
        assert finished.wait(5)
    finally:
        release.set()
        probe.close()
    assert not published and not probe.samples


def test_local_teacher_mutation_selects_new_artifact(tmp_path):
    source = tmp_path / 'teacher'
    source.mkdir()
    weights = source / 'weights.pt'
    weights.write_bytes(b'first')
    engine = teacher()
    engine._resolved_source = str(source)
    probe = EmotionProbe(tmp_path, {'model': 'same'}, engine, small_config())
    probe.close()
    weights.write_bytes(b'second')
    other = EmotionProbe(tmp_path, {'model': 'same'}, engine, small_config())
    try: assert other.key != probe.key
    finally: other.close()


def test_restored_dataset_keeps_pending_training_count(tmp_path):
    probe = EmotionProbe(tmp_path, {'model': 'resume'}, teacher(), small_config(), idle=lambda: False)
    probe.samples.append((torch.ones(256), [0, .5, 0., .5], 'one'))
    probe.since_train = 1
    probe.close()
    restored = EmotionProbe(tmp_path, {'model': 'resume'}, teacher(), small_config(), idle=lambda: False)
    try: assert restored.since_train == 1 and len(restored.samples) == 1
    finally: restored.close()
