from types import SimpleNamespace

import pytest

from process.app_core.audio.wake_word import WakeWord, profile_key, wake_phrase
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths


def config(root, name='Riko', **voice):
    return SimpleNamespace(root=root, paths=DataPaths.at(root), character_name=name, **audio_sections({'voice': voice}))


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


@pytest.mark.parametrize('mode', ['wake_word', 'continuous', 'manual'])
def test_multi_word_companion_name_wakes_on_its_first_word(tmp_path, mode):
    """Regression: the packaged setup accepts any name, and a two-word one stopped the backend after the model loaded."""
    wake = WakeWord(config(tmp_path, 'Ai  Hoshino', mode=mode))
    try:
        assert (wake.phrase, wake.unavailable, wake.status()['error']) == ('Ai', '', '')
        wake.bind_device({'name': 'Mic A'})
        wake.begin_calibration()  # enrollment works as for any one-word name
        assert wake.status()['calibrating']
    finally: wake.close()
    assert wake_phrase({}, '   ') == ('Riko', '') and wake_phrase(None, 'Mita') == ('Mita', '')
    assert wake_phrase({'wake_word': None}, 'Riko Chan') == ('Riko', '')
    assert wake_phrase({'wake_word': ' Mita '}, 'Riko Chan') == ('Mita', '')


@pytest.mark.parametrize('mode', ['wake_word', 'continuous', 'manual'])
@pytest.mark.parametrize('value, shown', [('Riko Chan', 'Riko Chan'), ('', ''), (42, '42'), (True, 'True')])
def test_unusable_wake_word_disables_only_wake_detection(tmp_path, monkeypatch, mode, value, shown):
    """An explicit wake word the detector cannot use, or one PyYAML read as a number or boolean, no longer raises (a
    non-string crashed on .strip()): enrollment and keyword detection are refused with the reason, the rest works."""
    wake = WakeWord(config(tmp_path, mode=mode, wake_word=value))
    detections = []
    monkeypatch.setattr(wake, '_schedule_detection', detections.append)
    try:
        assert wake.phrase == shown and wake.unavailable and wake.status()['error'] == wake.unavailable
        assert ('not text' in wake.unavailable) == (not isinstance(value, str))
        wake.bind_device({'name': 'Mic A'})
        assert wake.status()['error'] == wake.unavailable and not wake.status()['enrolled']
        with pytest.raises(ValueError, match='cannot be used'): wake.begin_calibration()
        with pytest.raises(ValueError): wake.set_testing(True)
        wake.embeddings = [[.1, .2] for _ in range(6)]  # even with samples, keyword detection never runs
        for _ in range(64): wake.feed(bytes(1024), speaking=True)
        for _ in range(64): wake.feed(bytes(1024), speaking=False)
        assert detections == []
        wake.activate()  # Speak now still opens the microphone for one turn
        assert wake.active()
    finally: wake.close()


def test_warmup_skips_the_detector_model_for_an_unusable_wake_word(tmp_path, monkeypatch):
    from process.app_core.runtime import warmup
    jobs = []
    monkeypatch.setattr(warmup, 'warm_components', lambda listed, timeout: jobs.append([name for name, _ in listed]))
    for value in ('Riko', 'Riko Chan'):
        wake = WakeWord(config(tmp_path, wake_word=value))
        session = SimpleNamespace(wake=wake, state=SimpleNamespace(audio_enabled=False), _closed=False,
            config=SimpleNamespace(raw={}, runtime=SimpleNamespace(startup_timeout_seconds=1), **audio_sections({})))
        try: warmup.warm_session(session)
        finally: wake.close()
    assert jobs == [['asr', 'vad', 'wake_detector'], ['asr', 'vad']]
