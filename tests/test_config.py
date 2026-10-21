import logging
import re

import pytest

from process.app_core.configuration.config import load_config
from process.app_core.kernel.audio_config import SovitsConfig, SpeechConfig, VoiceConfig


def test_missing_config_has_safe_defaults(tmp_path):
    config = load_config(tmp_path / "missing.yaml")
    assert config.character_name == "Riko"
    assert config.memory.history_file.parent == tmp_path / "persistent_memories"
    assert config.runtime.n_ctx == 8192
    assert (config.voice, config.speech, config.sovits, config.unknown_settings) == (VoiceConfig(), SpeechConfig(), SovitsConfig(), ())


def write(tmp_path, text):
    path = tmp_path / 'character_config.yaml'
    path.write_text(text, encoding='utf-8')
    return path


@pytest.mark.parametrize('line, message', [('interruption_seconds: 0', 'voice.interruption_seconds must be a finite number above 0'),
    ('interjection_debounce_seconds: 0', 'voice.interjection_debounce_seconds'), ("interjection_debounce_seconds: 'nan'", 'not .nan.'),
    ('assertive_window_seconds: 61', 'at most 60'), ('playback_words_per_second: .nan', 'playback_words_per_second'),
    ('vad_threshold: 1.5', 'voice.vad_threshold'), ('wake_threshold: 1', 'below 1'), ('follow_up_seconds: yes', 'not True'),
    ('mode: push_to_talk', 'voice.mode must be one of wake_word, continuous, manual'), ('input_device: 1.5', 'whole number'),
    ('transcription_gap_seconds: 2', 'shorter than voice.utterance_end_seconds'), ('live_transcript_interval_seconds: 0.1', 'at least 0.5')])
def test_a_bad_voice_value_stops_load_config_with_its_key_instead_of_failing_every_turn(tmp_path, line, message):
    with pytest.raises(ValueError, match=message): load_config(write(tmp_path, f'voice:\n  {line}\n'))


@pytest.mark.parametrize('text, message', [('sovits_ping_config:\n  sample_rate: 1000\n', 'sovits_ping_config.sample_rate'),
    ('sovits_ping_config:\n  max_in_flight_requests: 0\n', 'max_in_flight_requests'), ('sovits_ping_config:\n  auto_start: "yes"\n', 'auto_start'),
    ('sovits_ping_config:\n  arguments: --port\n', 'list of strings'), ('sovits_ping_config:\n  text_lang: no\n', 'put numbers, yes, no'),
    ('sovits_ping_config:\n  ref_audio_path: null\n', 'ref_audio_path must be text'), ('sovits_ping_config: 5\n', 'must be a mapping'),
    ('speech:\n  max_words: 0\n', 'speech.max_words'), ('voice: [mode]\n', 'voice must be a mapping')])
def test_bad_speech_and_sovits_values_stop_load_config_with_their_key(tmp_path, text, message):
    with pytest.raises(ValueError, match=message): load_config(write(tmp_path, text))


def test_audio_sections_load_typed_once_reading_numeric_text_and_emptied_sections(tmp_path):
    # PyYAML (YAML 1.1) reads a quoted number and 6e-1 as text, where ruamel and Electron (YAML 1.2) read numbers.
    config = load_config(write(tmp_path, "voice:\n  interruption_seconds: '2.5'\n  vad_threshold: 6e-1\n  input_device: 3\n"
        "speech:\nsovits_ping_config:\n  sample_rate: 24000\n  text_lang: ja\n  speed_factor: 1.1\n"))
    assert (config.voice.interruption_seconds, config.voice.vad_threshold, config.voice.input_device) == (2.5, .6, 3)
    assert config.speech == SpeechConfig()  # an emptied section keeps the defaults
    assert config.sovits.sample_rate == 24000 and config.sovits.request() == {'text_lang': 'ja', 'speed_factor': 1.1}
    with pytest.raises(AttributeError): config.voice.interruption_seconds = 0  # frozen: nothing unchecked replaces a value later


def test_unknown_and_obsolete_keys_are_reported_once_and_never_stop_startup(tmp_path, caplog):
    path = write(tmp_path, 'voice:\n  interuption_seconds: 9\n  asr_device: cpu\nspeech:\n  max_word: 5\nsovits_ping_config:\n  media_type: raw\n'
        'animation:\n  walkspeed: 300\nwake_feedback:\n  volumes: 1\ninitiative:\n  enable: true\nemotion:\n  probe:\n    device: cuda\n')
    with caplog.at_level(logging.WARNING, logger='process.app_core.kernel.validation'):
        config = load_config(path)
        assert load_config(path).unknown_settings == config.unknown_settings  # Settings loads it again on every request
    expected = ('voice.interuption_seconds', 'speech.max_word', 'sovits_ping_config.media_type', 'animation.walkspeed',
        'wake_feedback.volumes', 'initiative.enable', 'emotion.probe.device')
    assert config.unknown_settings == expected
    assert config.voice.interruption_seconds == 1.5  # the typo keeps the default, and each is named in the log once
    assert [record.getMessage().split(':')[0] for record in caplog.records] == [f'Ignoring {name}' for name in expected]


@pytest.mark.parametrize('provider', ['openai', 'lm_studio', 'llama_server'])
def test_llama_only_pool_keys_never_break_a_provider_that_has_no_pool(tmp_path, provider):
    from process.app_core.resources.vram_estimate import estimate
    slots = "'3'" if provider != 'llama_server' else '3'  # llama_server checks its slots: it shares them with Riko's scheduler
    config = load_config(write(tmp_path, f'runtime:\n  provider: {provider}\n  parallel_slots: {slots}\n'))
    assert config.runtime.kv_pool_tokens is None
    assert load_config(write(tmp_path, f'runtime:\n  provider: {provider}\n  kv_pool_auto: false\n  kv_pool_tokens: 1\n')).runtime.kv_pool_tokens == 1
    assert estimate(config, {'gpus': []})['kv']['pool_tokens'] is None
    with pytest.raises(ValueError, match='parallel_slots'):  # the in-process provider still checks them
        load_config(write(tmp_path, "runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n  parallel_slots: '3'\n"))


def test_an_unknown_key_seen_before_logging_is_configured_is_logged_by_the_next_load(monkeypatch, caplog):
    import logging as logs
    from process.app_core.kernel.validation import unknown
    with monkeypatch.context() as patch:
        patch.setattr(logs.getLogger(), 'handlers', [])  # run_server's first load_config, before configure_logging
        assert unknown('voice', {'early_typo': 1}, set()) == ['voice.early_typo']
    with caplog.at_level(logs.WARNING, logger='process.app_core.kernel.validation'):
        unknown('voice', {'early_typo': 1}, set())  # desktop_server's load, which reaches logs/debug.log
    assert [record.getMessage().split(':')[0] for record in caplog.records] == ['Ignoring voice.early_typo']


@pytest.mark.parametrize('text, key', [('memory:\n  system1_revision: main\n', 'memory.system1_revision'),
    ('emotion:\n  revision: v1.0\n', 'emotion.revision'), ('emotion:\n  revision: a85b127\n', 'emotion.revision'),
    ('memory:\n  system1_revision: 12345\n', 'memory.system1_revision')])
def test_julia_revisions_must_be_full_commits_because_julia_ships_code_that_runs_here(tmp_path, text, key):
    with pytest.raises(ValueError, match=re.escape(key) + ' must be a full 40-character commit hash'): load_config(write(tmp_path, text))


def test_julia_revisions_default_to_the_pin_and_accept_another_commit(tmp_path):
    from process.app_core.kernel.model_pins import JULIA_REVISION
    defaults = load_config(tmp_path / 'missing.yaml')
    assert defaults.memory.system1_revision == defaults.emotion.revision == JULIA_REVISION
    config = load_config(write(tmp_path, f"memory:\n  system1_revision: '{'b' * 40}'\nemotion:\n  revision: {'c' * 40}\n"))
    assert (config.memory.system1_revision, config.emotion.revision) == ('b' * 40, 'c' * 40)
