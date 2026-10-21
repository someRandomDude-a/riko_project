from types import SimpleNamespace

import pytest

from process.app_core.conversation.chat import ChatDeps
from process.app_core.runtime.actions import ActionController
from process.app_core.events.bus import event_bus
from process.app_core.audio.wake_feedback import WakeFeedback, select_rule, validate_settings
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths


def test_rule_selection_prefers_state_and_emotion_over_fallbacks():
    rules = [{'audio': 'default.wav'}, {'emotion': 'joy', 'audio': 'happy.wav'},
             {'state': 'thinking', 'audio': 'busy.wav'},
             {'state': 'thinking', 'emotion': 'joy', 'audio': 'happy-busy.wav'}]
    assert select_rule(rules, 'JOY', 'thinking')['audio'] == 'happy-busy.wav'
    assert select_rule(rules, 'sadness', 'thinking')['audio'] == 'busy.wav'
    assert select_rule(rules, 'joy', 'idle')['audio'] == 'happy.wav'
    assert select_rule(rules, 'sadness', 'idle')['audio'] == 'default.wav'
    assert select_rule([], 'neutral', 'idle') is None


@pytest.mark.parametrize('settings', [None, {'enabled': 'true'}, {'volume': True}, {'volume': 2},
    {'max_clip_seconds': float('inf')}, {'rules': {}}, {'rules': [{'state': [], 'audio': 'clip.wav'}]},
    {'rules': [{'state': 'unknown', 'audio': 'clip.wav'}]}, {'rules': [{}]},
    {'rules': [{'audio': 'clip.exe'}]}, {'rules': [{'animation': 'clip.fbx'}]},
    {'rules': [{'animation': 'clip.vrma', 'duration_seconds': -1}]}])
def test_invalid_feedback_settings_are_rejected(settings):
    with pytest.raises(ValueError): validate_settings(settings)


def test_feedback_uses_local_assets_respects_mute_and_cooldown(tmp_path, monkeypatch):
    directory = tmp_path / 'character_files'
    directory.mkdir()
    (directory / 'cue.wav').touch()
    (directory / 'wake.vrma').touch()
    settings = {'rules': [{'state': 'idle', 'emotion': 'joy', 'audio': 'character_files/cue.wav',
                          'animation': 'character_files/wake.vrma', 'volume': .3}]}
    calls, actions = [], ActionController()
    feedback = WakeFeedback(SimpleNamespace(root=tmp_path, raw={'wake_feedback': settings}),
        SimpleNamespace(submit_clip=lambda *a, **k: calls.append((a, k))), actions)
    now = [100.0]
    monkeypatch.setattr('process.app_core.audio.wake_feedback.time.monotonic', lambda: now[0])
    try:
        assert feedback.trigger('joy', 'idle')
        assert calls[0][1]['volume'] == .3
        assert actions.active()[0]['kind'] == 'wake_animation'
        assert actions.active()[0]['payload']['model_state'] == 'idle'
        assert not feedback.trigger('joy', 'idle')
        now[0] += 1
        assert feedback.trigger('joy', 'idle', audio_enabled=False)
        assert len(calls) == 1
        assert len(actions.active()) == 1
        assert not feedback.trigger('sadness', 'idle')
    finally: actions.close()


def test_missing_or_outside_asset_does_not_suppress_valid_animation(tmp_path):
    directory = tmp_path / 'character_files'
    directory.mkdir()
    (directory / 'wake.vrma').touch()
    outside = tmp_path / 'outside.wav'
    outside.touch()
    actions, events, calls = ActionController(), [], []
    unsubscribe = event_bus.subscribe(events.append)
    config = SimpleNamespace(root=tmp_path, raw={'wake_feedback': {'rules': [
        {'audio': 'outside.wav', 'animation': 'character_files/wake.vrma'}]}})
    feedback = WakeFeedback(config, SimpleNamespace(submit_clip=lambda *a, **k: calls.append(a)), actions)
    try:
        assert feedback.trigger('neutral', 'idle')
        assert not calls and actions.active()
        assert any(event.type == 'wake.feedback.error' for event in events)
    finally:
        unsubscribe()
        actions.close()


def test_only_keyword_activation_triggers_session_feedback(tmp_path):
    from process.app_core.desktop.state import DesktopState
    from process.app_core.runtime.session import SessionManager
    config = SimpleNamespace(root=tmp_path, paths=DataPaths.at(tmp_path), character_name='Riko', raw={}, **audio_sections({}))
    chat = SimpleNamespace(deps=ChatDeps(), provider=BaseProvider())
    session = SessionManager(config, chat, DesktopState())
    calls = []
    session.wake_feedback.trigger = lambda *a, **k: calls.append((a, k))
    try:
        session.wake.activate()
        assert not calls
        session._generation_active = True
        # Inject keyword events to exercise each runtime-dependent selection.
        event_bus.publish('voice.activated', source='keyword')
        assert calls[-1][0] == ('neutral', 'thinking')
        session._playing = {'text': 'speaking'}
        event_bus.publish('voice.activated', source='keyword')
        assert calls[-1][0][1] == 'speaking'
        session.state.set_sleep(True)
        event_bus.publish('voice.activated', source='keyword')
        assert calls[-1][0][1] == 'sleeping'
        session.state.set_mic(False)
        before = len(calls)
        event_bus.publish('voice.activated', source='keyword')
        assert len(calls) == before
        action = session.actions.start('wake_animation', {}, duration=10)
        session.cancel()
        assert action.status == 'cancelled'
    finally:
        session._playing = None
        session.close()


def test_invalid_feedback_settings_leave_no_session_worker_or_listener(tmp_path, monkeypatch):
    """desktop_server keeps Settings up when the session cannot start, so a failed construction must not leak."""
    from process.app_core.desktop.state import DesktopState
    from process.app_core.runtime.session import SessionManager
    closed = []
    class Speech:
        def __init__(self, *args): pass
        def close(self): closed.append('speech')
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    config = SimpleNamespace(root=tmp_path, paths=DataPaths.at(tmp_path), character_name='Riko', raw={'wake_feedback': {'volume': 5}}, **audio_sections({}))
    listeners = len(event_bus._listeners)
    with pytest.raises(ValueError, match='wake_feedback.volume'):
        SessionManager(config, SimpleNamespace(deps=ChatDeps(), provider=BaseProvider()), DesktopState())
    assert closed == ['speech'] and len(event_bus._listeners) == listeners


def test_feedback_defaults_and_rules_are_editable_and_validated_in_settings(tmp_path):
    from process.app_core.configuration.settings_store import SettingsStore
    path = tmp_path / 'character_config.yaml'
    path.write_text('character_name: Riko\n', encoding='utf-8')
    store = SettingsStore(path)
    fields = {item['path']: item for item in store.snapshot()['fields']}
    assert fields['wake_feedback.rules']['section'] == 'Wake acknowledgement'
    assert fields['wake_feedback.rules']['group'] == 'voice'
    _, output, errors = store.prepare({'wake_feedback.rules': [{'emotion': 'joy', 'state': 'idle', 'audio': 'character_files/cue.wav'}]})
    assert output and not errors
    _, output, errors = store.prepare({'wake_feedback.rules': [{'state': 'invalid', 'audio': 'character_files/cue.wav'}]})
    assert errors
    assert path.read_text(encoding='utf-8') == 'character_name: Riko\n'
