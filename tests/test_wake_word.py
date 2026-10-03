from types import SimpleNamespace

import pytest

from process.app_core.audio.wake_word import WakeWord, profile_key


def config(root, **voice):
    return SimpleNamespace(root=root, character_name='Riko', raw={'voice': voice})


def test_profile_key_changes_for_phrase_or_actual_device():
    assert profile_key('Riko', {'name': 'Mic A'}) == profile_key(' riko ', {'name': 'Mic A'})
    assert profile_key('Riko', {'name': 'Mic A'}) != profile_key('Riko', {'name': 'Mic B'})
    assert profile_key('Riko', {'name': 'Mic A'}) != profile_key('Mita', {'name': 'Mic A'})


def test_enrollment_saved_and_reused_only_for_same_identity(tmp_path):
    wake = WakeWord(config(tmp_path))
    wake.bind_device({'name': 'Mic A'})
    wake.begin_calibration()
    with pytest.raises(ValueError): wake.finish_calibration()
    wake.samples = [[.1, .2] for _ in range(6)]
    wake.finish_calibration()
    assert wake.status()['enrolled']
    reloaded = WakeWord(config(tmp_path))
    reloaded.bind_device({'name': 'Mic A'})
    assert reloaded.status()['enrolled']
    reloaded.bind_device({'name': 'Mic B'})
    assert not reloaded.status()['enrolled']
    wake.close()
    reloaded.close()


def test_followup_expires_and_waits_until_response_finishes(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr('process.app_core.audio.wake_word.time.monotonic', lambda: now[0])
    wake = WakeWord(config(tmp_path))
    assert not wake.active()
    wake.activate()
    assert wake.capture_boundary == (100.0, False)
    assert wake.active()
    now[0] = 111
    assert not wake.active()
    wake.responding()
    now[0] = 200
    assert wake.active()
    wake.response_finished()
    now[0] = 209
    assert wake.active()
    now[0] = 211
    assert not wake.active()
    wake.close()


def test_continuous_and_manual_modes(tmp_path):
    continuous = WakeWord(config(tmp_path, mode='continuous'))
    manual = WakeWord(config(tmp_path, mode='manual'))
    assert continuous.active()
    assert not manual.active()
    manual.activate()
    assert manual.active()
    continuous.close()
    manual.close()


def test_threshold_saved_and_more_samples_allowed(tmp_path):
    wake = WakeWord(config(tmp_path))
    wake.bind_device({'name': 'Mic A'})
    wake.begin_calibration()
    wake.samples = [[.1, .2] for _ in range(6)]
    wake.finish_calibration()
    wake.set_threshold(.72)
    reloaded = WakeWord(config(tmp_path))
    reloaded.bind_device({'name': 'Mic A'})
    assert reloaded.threshold == .72
    assert reloaded.status()['samples'] == 6
    reloaded.begin_calibration()
    assert len(reloaded.samples) == 6
    reloaded.samples.extend([[.3, .4], [.5, .6]])
    reloaded.record()  # Enrollment is not capped at six.
    reloaded.recording = None
    reloaded.finish_calibration()
    assert len(reloaded.embeddings) == 8
    with pytest.raises(ValueError): reloaded.set_threshold(1)
    wake.close()
    reloaded.close()


def test_testing_reports_match_without_activation(tmp_path, monkeypatch):
    import numpy as np
    from process.app_core.events.bus import event_bus
    wake = WakeWord(config(tmp_path))
    wake.bind_device({'name': 'Mic A'})
    wake.embeddings = [[.1, .2] for _ in range(6)]
    wake.threshold = .7
    wake.model = SimpleNamespace(window_frames=24000, audioToVector=lambda _: np.array([[.1, .2]]), scoreVector=lambda *_: .8)
    monkeypatch.setattr('process.app_core.audio.wake_word.prepare_audio', lambda *_: np.zeros(24000))
    received = []
    unsubscribe = event_bus.subscribe(received.append)
    try:
        wake.set_testing(True)
        wake._detect(bytes(1024))
        assert not wake.active()
        assert wake.capture_boundary is None
        assert wake.last_score == .8
        assert any(event.type == 'voice.wake_test' and event.payload['matched'] for event in received)
        wake.set_testing(False)
        wake._detect(bytes(1024))
        assert wake.active()
        assert wake.capture_boundary[1] is True
    finally:
        unsubscribe()
        wake.close()
