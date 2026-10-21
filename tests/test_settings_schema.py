"""The settings schema registry (kernel/schema.py, configuration/schema.py): each package registers its own sections, and
load_config's checks, Settings' fields, rules, restart scopes and live hooks all come from it."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from process.app_core.configuration import schema as settings
from process.app_core.configuration.config import load_config
from process.app_core.configuration.settings_store import SettingsStore
from process.app_core.kernel import schema
from process.app_core.kernel.schema import Registry, Rule, Section, Setting

CODE = Path(__file__).resolve().parents[1] / 'Code'
SECTIONS = ['', 'animation', 'avatar', 'desktop', 'desktop.shortcuts', 'emotion', 'emotion.probe', 'initiative', 'logging', 'memory',
    'presets', 'presets.default.model_params', 'runtime', 'sovits_ping_config', 'speech', 'tasks', 'tools', 'voice', 'wake_feedback']


def write(tmp_path, text, name='character_config.yaml'):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return path


def with_sections(monkeypatch, *extra):
    """The registry as the app has it, plus extra sections, for this test only."""
    registry = Registry()
    registry.register(*schema.REGISTRY.sections(), *extra)
    monkeypatch.setattr(schema, 'REGISTRY', registry)
    return registry


def test_load_config_alone_checks_every_package_section_without_configuration_importing_them(tmp_path):
    """Regression: configuration imported animation, audio, desktop, emotion, inference and runtime for their validators
    (an inverted dependency); now they register their own sections, and load_config, run with nothing else imported (as
    a test or a script would), still checks each of them."""
    cases = {'animation': 'animation:\n  walk_speed: 5000\n', 'wake_feedback': 'wake_feedback:\n  volume: 2\n',
        'probe': 'emotion:\n  probe:\n    enabled: true\n', 'runtime': 'runtime:\n  provider: llama_cpp\n',
        'initiative': 'initiative:\n  max_output_tokens: 9000\n', 'tools': 'tools:\n  best_fit_min_confidence: 2\n',
        'voice': 'voice:\n  mode: push_to_talk\n', 'valid': 'animation:\n  walkspeed: 300\n'}
    paths = {name: str(write(tmp_path, text, f'{name}.yaml')) for name, text in cases.items()}
    code = ('import json, sys\nfrom process.app_core.configuration.config import load_config\nfrom process.app_core.kernel import schema\n'
        'found = {}\nfor name, path in json.loads(sys.argv[1]).items():\n'
        '    try: found[name] = list(load_config(path).unknown_settings)\n    except ValueError as exc: found[name] = str(exc)\n'
        "print(json.dumps({'sections': sorted({s.path for s in schema.REGISTRY.sections()}), 'found': found,\n"
        "    'loaded': sorted(m for m in sys.modules if m.startswith('process.app_core.') and m.split('.')[2] in ('animation', 'desktop', 'emotion', 'runtime'))}))")
    result = subprocess.run([sys.executable, '-c', code, json.dumps(paths)], capture_output=True, text=True, timeout=120,
        env={**os.environ, 'PYTHONPATH': str(CODE)})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['sections'] == SECTIONS
    found = report['found']
    assert 'animation.walk_speed must be a finite number at least 20 and at most 1000' in found['animation']
    assert 'wake_feedback.volume' in found['wake_feedback'] and 'emotion.probe requires' in found['probe']
    assert 'llama_cpp requires runtime.model_path' in found['runtime'] and 'initiative output limit' in found['initiative']
    assert 'tools.best_fit_min_confidence' in found['tools'] and 'voice.mode must be one of' in found['voice']
    assert found['valid'] == ['animation.walkspeed']
    # The feature modules load on first read, through app_core/__init__.py; configuration itself imports none of them.
    assert 'process.app_core.animation.library' in report['loaded'] and 'process.app_core.desktop.avatar_models' in report['loaded']


def test_configuration_imports_no_feature_package():
    from test_package_boundaries import scan
    assert scan()[0]['configuration'] == {'kernel': 'load', 'persistence': 'load'}


def test_a_setting_is_one_registered_spec(monkeypatch, tmp_path):
    """Adding a setting used to take up to seven edits (CLAUDE.md); a registered Setting now gives Settings its field,
    default, range, label, help, restart scope and live hook, and load_config's unknown-key check knows it."""
    with_sections(monkeypatch, Section('demo', group='speech', title='Demo', strict=True, settings=(
        Setting('speed', 2.5, range=(1, 5), restart='none', live='demo_speed', label='Demo speed', help='How fast.'),
        Setting('mode', 'calm', options=('calm', 'busy'), listed=False))))
    path = write(tmp_path, 'runtime:\n  provider: lm_studio\ndemo:\n  mode: busy\n  sped: 3\n')
    store = SettingsStore(path)
    snapshot = store.snapshot()
    fields = {item['path']: item for item in snapshot['fields']}
    assert snapshot['values']['demo.speed'] == 2.5 and snapshot['values']['demo.mode'] == 'busy'
    assert {key: fields['demo.speed'][key] for key in ('label', 'help', 'group', 'section', 'kind', 'integer', 'min', 'max', 'restart', 'restart_scope')} == {
        'label': 'Demo speed', 'help': 'How fast.', 'group': 'speech', 'section': 'Demo', 'kind': 'number', 'integer': False,
        'min': 1, 'max': 5, 'restart': False, 'restart_scope': 'none'}
    assert fields['demo.mode']['options'] == ['calm', 'busy'] and fields['demo.mode']['restart']
    assert snapshot['warnings'] == {'demo.sped': settings.UNKNOWN} and load_config(path).unknown_settings == ('demo.sped',)
    assert 'demo.speed' in store.validate({'demo.speed': 6})['errors'] and store.validate({'demo.speed': 4.5})['valid']
    assert settings.live_hooks({'demo.speed': 3, 'runtime.temperature': .5}) == {'demo_speed': ['demo.speed']}
    result = store.save({'demo.speed': 3}, snapshot['revision'])
    assert result['saved'] and not result['restart_required'] and 'speed: 3' in path.read_text()


def test_registry_refuses_a_setting_or_check_declared_twice_and_loads_its_modules_once():
    registry, calls = Registry(), []
    registry.register(Section('demo', check=dict, settings=(Setting('speed', 1),)))
    with pytest.raises(ValueError, match='declared twice: demo.speed'): registry.register(Section('demo', settings=(Setting('speed', 2),)))
    with pytest.raises(ValueError, match="sets check twice"): registry.register(Section('demo', check=list))
    registry.register(Section('demo', settings=(Setting('size', 2),)))  # another module may add keys to the same section
    registry.loader(lambda: calls.append('load') or registry.register(Section('late', restart='electron')))
    assert calls == [] and [section.path for section in registry.sections()] == ['demo', 'demo', 'late'] and calls == ['load']
    registry.sections(); registry.setting('demo.size')
    assert calls == ['load'] and registry.resolve('late.anything', 'restart') == 'electron' and registry.resolve('demo.size', 'restart') is None


def test_restart_scopes_and_live_hooks_come_from_the_sections_around_a_setting():
    scope = settings.restart_scope
    assert [scope(path) for path in ('avatar.camera.fov', 'runtime.pause_background_on_live', 'emotion.probe.interval_tokens')] == ['none'] * 3
    assert [scope(path) for path in ('desktop.debug', 'desktop.shortcuts.toggle', 'presets.default.name', 'sovits_ping_config.auto_start')] == ['electron'] * 4
    assert [scope(path) for path in ('runtime.n_ctx', 'custom_extension.untouched', 'your_name')] == ['python'] * 3
    assert settings.live_hooks({'avatar.model': 'x', 'avatar.camera.fov': 30, 'runtime.n_ctx': 1}) == {'avatar': ['avatar.model', 'avatar.camera.fov']}
    field = settings.field('desktop.shortcuts.toggle', 'Ctrl+R')
    assert (field['group'], field['restart'], field['restart_scope']) == ('appearance', True, 'electron')


def test_a_float_setting_written_as_a_whole_number_still_takes_fractions(tmp_path):
    """Regression: Settings decided `integer` from the value, so a YAML temperature of 1 (or a volume of 1) refused 0.7 with
    'Enter a whole number'; the declared default's type decides now."""
    store = SettingsStore(write(tmp_path, 'runtime:\n  provider: lm_studio\n  temperature: 1\nwake_feedback:\n  volume: 1\nanimation:\n  walk_speed: 300\n'))
    fields = {item['path']: item for item in store.snapshot()['fields']}
    assert not any(fields[path]['integer'] for path in ('runtime.temperature', 'wake_feedback.volume', 'animation.walk_speed'))
    assert fields['runtime.n_ctx']['integer'] and fields['speech.max_words']['integer']
    assert store.validate({'runtime.temperature': .7, 'wake_feedback.volume': .5, 'animation.walk_speed': 250.5}) == {'valid': True, 'errors': {}}
    assert store.validate({'runtime.n_ctx': 4096.5})['errors'] == {'runtime.n_ctx': 'Enter a whole number'}


@pytest.mark.parametrize('section', ['animation', 'wake_feedback'])
def test_the_range_settings_offers_is_the_range_load_config_accepts(tmp_path, section):
    for item in next(s for s in settings.sections() if s.path == section and s.settings).settings:
        if not item.range: continue
        low, high = item.range
        for value, ok in ((low, True), (high, True), (low - .01, False), (high + .01, False)):
            path = write(tmp_path, f'{section}:\n  {item.key}: {value}\n')
            if ok: load_config(path)
            else:
                with pytest.raises(ValueError, match=f'{section}.{item.key}'): load_config(path)


def test_settings_returns_provider_visibility_and_the_rules_the_backend_enforces(tmp_path):
    """Regression: the renderer hard-coded which provider uses each runtime key and repeated llama_runtime's checks across
    settings; /api/settings now returns both (visible_when, rules), from the objects load_config itself enforces."""
    store = SettingsStore(write(tmp_path, 'runtime:\n  provider: llama_server\n'))
    snapshot = store.snapshot()
    fields = {item['path']: item for item in snapshot['fields']}
    assert fields['runtime.base_url']['visible_when'] == {'runtime.provider': ['llama_server', 'lm_studio', 'openai', 'openai_compatible', 'ollama', 'local_http']}
    assert fields['runtime.n_ubatch']['visible_when'] == {'runtime.provider': ['llama_cpp']}
    assert fields['runtime.parallel_slots']['visible_when'] == {'runtime.provider': ['llama_cpp', 'llama_server']}
    assert 'visible_when' not in fields['runtime.temperature']
    # Every provider's keys arrive, so switching the draft's provider shows them at once (the renderer filters by visible_when).
    assert fields['runtime.model']['visible_when'] == fields['runtime.api_mode']['visible_when'] == {'runtime.provider': ['lm_studio', 'openai', 'openai_compatible', 'ollama', 'local_http']}
    rules = {rule['path']: rule for rule in snapshot['rules']}
    assert rules['runtime.n_ubatch'] == {'path': 'runtime.n_ubatch', 'message': 'runtime.n_ubatch must not exceed n_batch',
        'when': {'runtime.provider': ['llama_cpp']}, 'at_most': 'runtime.n_batch'}
    assert rules['runtime.type_v']['one_of'] == {'runtime.flash_attn': ['auto', 'on']} and 'q8_0' in rules['runtime.type_v']['when']['runtime.type_v']
    assert rules['runtime.hf_repo_id']['any_set'] == [['runtime.model_path'], ['runtime.hf_repo_id', 'runtime.hf_filename']]
    assert rules['voice.transcription_gap_seconds']['below'] == 'voice.utterance_end_seconds' and rules['speech.split_window_words']['at_most'] == 'speech.max_words'
    native = 'runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n'
    broken = {'runtime.n_ubatch': native + '  n_batch: 256\n  n_ubatch: 512\n', 'runtime.type_v': native + '  type_v: q8_0\n  flash_attn: off\n',
        'runtime.hf_repo_id': 'runtime:\n  provider: llama_cpp\n', 'voice.transcription_gap_seconds': 'voice:\n  transcription_gap_seconds: 2\n',
        'speech.split_window_words': 'speech:\n  max_words: 5\n  split_window_words: 10\n'}
    for path, text in broken.items():
        with pytest.raises(ValueError, match=rules[path]['message'].replace('.', r'\.')): load_config(write(tmp_path, text))
    load_config(write(tmp_path, native + '  n_batch: 512\n  n_ubatch: 512\n  type_v: q8_0\n  flash_attn: auto\n'))


def test_rules_hold_unless_every_condition_does_and_the_requirement_fails():
    rule = Rule('a', 'a <= b', when=(('kind', ('x',)),), at_most='b')
    assert rule.broken({'kind': 'x', 'a': 3, 'b': 2}) and not rule.broken({'kind': 'y', 'a': 3, 'b': 2}) and not rule.broken({'kind': 'x', 'a': None, 'b': 2})
    source = Rule('p', 'need one', any_set=(('p',), ('q', 'r')))
    assert source.broken({'p': '', 'q': 'x'}) and not source.broken({'q': 'x', 'r': 'y'}) and not source.broken({'p': 'model.gguf'})
    assert Rule('t', 'needs flash', one_of=(('f', ('on',)),)).broken({'f': 'off'})


def test_snapshot_says_which_saved_settings_still_wait_for_a_restart(tmp_path):
    """Regression: snapshot() always said restart_required: true. Against the config the backend started with it now names
    each saved setting that differs, with the restart that applies it; settings applied on save are never pending."""
    path = write(tmp_path, 'runtime:\n  provider: lm_studio\n  temperature: 0.7\ndesktop:\n  debug: false\npresets:\n  default:\n    name: Riko\n')
    store = SettingsStore(path, running=load_config(path))
    snapshot = store.snapshot()
    assert (snapshot['restart_required'], snapshot['restart_pending']) == (False, {})
    result = store.save({'avatar.enabled': False}, snapshot['revision'])
    assert result['saved'] and not result['restart_required'] and result['restart_pending'] == {}
    result = store.save({'runtime.temperature': .2, 'desktop.debug': True, 'presets.default.name': 'Mita'}, result['revision'])
    assert result['saved'] and result['restart_required']
    assert result['restart_pending'] == {'desktop.debug': 'electron', 'presets.default.name': 'electron', 'runtime.temperature': 'python', 'voice.wake_word': 'python'}
    assert store.snapshot()['restart_required'] and not SettingsStore(path).snapshot()['restart_required']  # no running backend, nothing pending
