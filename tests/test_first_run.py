"""The packaged first run's configuration (configuration/first_run.py, run by electron/release.cjs saveSetup as
`riko-backend --setup-config`) and `--validate-config`. electron/src/release.test.mjs covers Electron's half (the data
folder's rules and never overwriting), and tests/test_settings_store.py runs both halves together."""
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from process.app_core.configuration.config import load_config
from process.app_core.configuration.first_run import command, render, setup_config, setup_text, whole
from process.app_core.configuration.native_backends import backends_for, bundled_library
from process.app_core.configuration.settings_store import SettingsStore

ROOT = Path(__file__).resolve().parents[1]
FORM = {'backend': 'cuda', 'repo': 'owner/model', 'filename': 'model.gguf', 'context': 8192, 'output': 1024, 'threads': 4,
    'memories': 'I like tea.\nMy name is Alex.', 'sovitsAuto': False}


def application(root, backend='cuda', platform='linux'):
    """A packaged app's resources holding one native bundle, under that OS's library name."""
    library = bundled_library(backend, root / 'application', platform)
    library.parent.mkdir(parents=True); library.write_text('test')
    return root / 'application'


@pytest.fixture
def host(tmp_path, monkeypatch):
    """This OS's first shipped bundle, where load_config looks for it (RIKO_BUNDLE_ROOT), and an empty data folder."""
    backend = backends_for()[0]
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(application(tmp_path, backend, sys.platform)))
    (tmp_path / 'data').mkdir()
    return {**FORM, 'backend': backend}, tmp_path / 'data'


def test_the_choices_go_over_what_a_packaged_build_starts_with(tmp_path):
    config = setup_config(FORM, 'linux', application(tmp_path))
    assert config['runtime']['native_library'] == 'bundled:cuda' and config['runtime']['provider'] == 'llama_cpp'
    assert [memory['text'] for memory in config['memory']['default_memories']] == ['I like tea.', 'My name is Alex.']
    assert config['tools'] == {'require_approval': True} and config['emotion']['probe'] == {'enabled': False}
    assert config['voice'] == {'wake_word': 'Riko', 'asr_device': 'cpu', 'asr_compute_type': 'int8'}
    assert config['initiative'] == {'enabled': False} and config['desktop'] == {'setup_on_startup_error': True}
    assert config['runtime']['n_gpu_layers'] == -1 and setup_config({**FORM, 'cpuOnly': True}, 'linux', application(tmp_path / 'cpu'))['runtime']['n_gpu_layers'] == 0


def test_the_form_may_hold_numbers_as_typed_text():
    assert [whole(value) for value in (8192, '8192', ' 8192 ', 8192.0, '2.5', 2.5, True, 'abc', '', None, float('inf'))] == [8192, 8192, 8192, 8192, None, None, None, None, None, None, None]


@pytest.mark.parametrize('name, saved, wake', [('Riko Chan', 'Riko Chan', 'Riko'), ('  Ai  Hoshino ', 'Ai  Hoshino', 'Ai'), ('', 'Riko', 'Riko'), (None, 'Riko', 'Riko'), ('Yes', 'Yes', 'Yes')])
def test_setup_wakes_on_the_first_word_of_any_companion_name(tmp_path, name, saved, wake):
    config = setup_config({**FORM, 'name': name}, 'linux', application(tmp_path))
    assert config['presets']['default']['name'] == saved and config['voice']['wake_word'] == wake


def test_the_yaml_reads_back_as_text_where_pyyaml_would_not():
    """The backend's PyYAML reads YAML 1.1, where an unquoted Yes, on or 42 is not text."""
    config = {'presets': {'default': {'name': 'Yes'}}, 'voice': {'wake_word': 'On'}, 'memory': {'default_memories': [{'text': 'on'}, {'text': '42'}, {'text': '1:30'}]}}
    assert yaml.safe_load(render(config)) == config


def test_setup_validates_budgets_model_and_explicit_speech_consent(tmp_path):
    bundle = application(tmp_path)
    for change, message in (({'output': 8192}, 'budget'), ({'context': 1024}, 'budget'), ({'context': 'many'}, 'budget'), ({'output': 32}, 'budget'),
            ({'threads': 0}, 'thread'), ({'threads': 2.5}, 'thread'), ({'filename': '../secret.gguf'}, 'GGUF'), ({'filename': 'model.bin'}, 'GGUF'),
            ({'repo': ''}, 'GGUF'), ({'modelPath': 'relative.gguf'}, 'Local GGUF'), ({'modelPath': str(tmp_path / 'missing.gguf')}, 'Local GGUF'),
            ({'sovitsAuto': True}, 'GPT-SoVITS'), ({'sovitsAuto': True, 'sovitsExecutable': 'relative.exe'}, 'GPT-SoVITS')):
        with pytest.raises(ValueError, match=message): setup_config({**FORM, **change}, 'linux', bundle)
    with pytest.raises(ValueError, match='must be an object'): setup_config(['not', 'a', 'form'], 'linux', bundle)
    model = tmp_path / 'Model.GGUF'; model.write_text('gguf')
    runtime = setup_config({**FORM, 'repo': '', 'modelPath': str(model)}, 'linux', bundle)['runtime']
    assert (runtime['model_path'], runtime['hf_repo_id']) == (str(model), None)


@pytest.mark.parametrize('context, output', [(8192, 1024), (2048, 2047), (131072, 64)])
def test_setup_leaves_room_for_the_reply_inside_the_live_context_as_settings_requires(host, context, output):
    choices, data = host
    config = yaml.safe_load(setup_text({**choices, 'context': context, 'output': output}, data))
    runtime, memory = config['runtime'], config['memory']
    assert (runtime['n_ctx'], runtime['max_output_tokens']) == (context, output)
    assert memory['context_window_tokens'] >= 1 and memory['context_window_tokens'] + runtime['max_output_tokens'] <= runtime['n_ctx']


def test_setup_writes_bundled_backend_only_for_a_bundle_this_os_ships(tmp_path):
    for index, (platform, backend) in enumerate([('win32', 'cuda'), ('win32', 'vulkan'), ('linux', 'vulkan'), ('darwin', 'metal')]):
        bundle = application(tmp_path / str(index), backend, platform)
        assert setup_config({**FORM, 'backend': backend}, platform, bundle)['runtime']['native_library'] == f'bundled:{backend}'
    windows, mac = application(tmp_path / 'windows', 'cuda', 'win32'), application(tmp_path / 'mac', 'cuda', 'linux')
    with pytest.raises(ValueError, match=r'Packaged native backend is missing: .*libriko-native\.so'): setup_config(FORM, 'linux', windows)
    with pytest.raises(ValueError, match=r'^Choose Metal$'): setup_config(FORM, 'darwin', mac)
    with pytest.raises(ValueError, match=r'^Choose CUDA or Vulkan$'): setup_config({**FORM, 'backend': 'metal'}, 'win32', mac)
    with pytest.raises(ValueError, match='Choose CUDA or Vulkan'): setup_config({**FORM, 'backend': '../native'}, 'linux', mac)
    with pytest.raises(ValueError, match='No native backend ships for freebsd'): setup_config(FORM, 'freebsd', mac)
    # The opt-in hip bundle: written when the installation holds it, refused (by its missing library) when it does not.
    assert setup_config({**FORM, 'backend': 'hip'}, 'linux', application(tmp_path / 'hip', 'hip', 'linux'))['runtime']['native_library'] == 'bundled:hip'
    with pytest.raises(ValueError, match=r'Packaged native backend is missing: .*hip.libriko-native\.so'): setup_config({**FORM, 'backend': 'hip'}, 'linux', mac)
    with pytest.raises(ValueError, match=r'^Choose CUDA or Vulkan$'): setup_config({**FORM, 'backend': 'hip'}, 'win32', mac)


def test_setup_text_is_checked_as_settings_checks_an_edit_and_leaves_nothing_behind(host, monkeypatch):
    choices, data = host
    text = setup_text({**choices, 'name': 'Yes', 'memories': 'on\n42'}, data)
    assert list(data.iterdir()) == []
    path = data / 'character_config.yaml'; path.write_text(text, encoding='utf-8')
    config = load_config(path)
    assert config.character_name == 'Yes' and config.raw['voice']['wake_word'] == 'Yes' and config.memory.default_memories[1]['text'] == '42'
    assert SettingsStore(path).validate_file() == {'valid': True, 'errors': {}}  # nothing a strict section does not read
    with pytest.raises(ValueError, match=r'sovits_ping_config\.url: Use an http:// or https:// URL'): setup_text({**choices, 'sovitsUrl': 'ftp://tts'}, data)
    monkeypatch.delenv('RIKO_BUNDLE_ROOT')
    with pytest.raises(ValueError, match='Bundled native library requires the packaged app'): setup_text(choices, data)
    assert [item.name for item in data.iterdir()] == ['character_config.yaml']


def test_a_file_is_validated_as_the_backend_reads_it(tmp_path):
    """Regression: validating the file through ruamel's round-trip view refused every float (ScalarFloat) and hex number,
    and read desktop.debug: yes as text where the backend's PyYAML reads a boolean."""
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: openai\n  temperature: 0.7\n  request_timeout_seconds: 0x10\n  flash_attn: true\n'
        'desktop:\n  debug: yes\npresets:\n  default:\n    name: Mika\n', encoding='utf-8')
    store = SettingsStore(path)
    assert store.validate_file() == {'valid': True, 'errors': {}}
    path.write_text('runtime:\n  provider: openai\n  max_output_tokens: 0\nlogging:\n  level: LOUD\n  colour: true\n', encoding='utf-8')
    result = store.validate_file()
    assert not result['valid'] and set(result['errors']) == {'runtime.max_output_tokens', 'logging.level'}
    path.write_text('speech:\n  max_words: 5\n  split_window_words: 9\n', encoding='utf-8')
    assert store.validate_file() == {'valid': False, 'errors': {'__all__': 'speech.split_window_words cannot exceed speech.max_words'}}
    path.write_text('initiative:\n  enabeld: true\n', encoding='utf-8')  # reported, as Settings reports it, never an error
    result = store.validate_file()
    assert result['valid'] and list(result['warnings']) == ['initiative.enabeld']
    path.write_text('- a list\n', encoding='utf-8')
    with pytest.raises(ValueError, match='mapping'): store.validate_file()


def test_the_command_answers_one_line_of_json(host):
    choices, data = host
    def run(*argv, source=''):
        stdout = io.StringIO()
        return command(['run_server.py', *argv], io.StringIO(source), stdout), stdout.getvalue()
    status, output = run('--setup-config', str(data), source=json.dumps({**choices, 'name': 'Zoë'}))
    assert status == 0 and output.endswith('\n') and output.count('\n') == 1 and output.isascii()
    assert yaml.safe_load(json.loads(output)['config'])['presets']['default']['name'] == 'Zoë'
    assert run('--setup-config', str(data), source=json.dumps({**choices, 'threads': 0})) == (1, json.dumps({'error': 'Invalid CPU thread count'}) + '\n')
    assert run('--setup-config', str(data), source='{not json')[0] == 1 and run('--setup-config')[1] == json.dumps({'error': '--setup-config needs the data folder'}) + '\n'
    path = data / 'character_config.yaml'; path.write_text('runtime:\n  provider: openai\n', encoding='utf-8')
    assert run('--validate-config', str(path)) == (0, json.dumps({'valid': True, 'errors': {}}) + '\n')
    path.write_text('runtime:\n  provider: openai\n  n_ctx: -1\n', encoding='utf-8')
    status, output = run('--validate-config', str(path))
    assert status == 1 and 'runtime.n_ctx' in json.loads(output)['errors']
    assert run('--validate-config', str(data / 'missing.yaml'))[0] == 1


def test_the_real_command_line_routes_through_run_server(host, tmp_path):
    """As Electron runs it in development (python Code/run_server.py) and frozen (riko-backend): stdin, one line, the exit status."""
    choices, data = host
    server = [sys.executable, str(ROOT / 'Code' / 'run_server.py')]
    setup = subprocess.run([*server, '--setup-config', str(data)], input=json.dumps(choices), capture_output=True, text=True, encoding='utf-8', cwd=tmp_path, timeout=300)
    assert setup.returncode == 0, setup.stdout + setup.stderr
    config = yaml.safe_load(json.loads(setup.stdout.splitlines()[-1])['config'])
    assert config['runtime']['native_library'] == f'bundled:{choices["backend"]}' and list(data.iterdir()) == []
    (data / 'character_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    check = subprocess.run([*server, '--validate-config', str(data / 'character_config.yaml')], capture_output=True, text=True, encoding='utf-8', cwd=tmp_path, timeout=300)
    assert (check.returncode, json.loads(check.stdout.splitlines()[-1])) == (0, {'valid': True, 'errors': {}}), check.stdout + check.stderr
    assert sorted(item.name for item in tmp_path.iterdir()) == ['application', 'data']  # no logs or data root in the working directory
