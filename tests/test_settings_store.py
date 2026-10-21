from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest

from process.app_core.configuration.config import load_config
from process.app_core.configuration.native_backends import backends_for, bundled_library
from process.app_core.configuration.settings_store import SettingsStore, SettingsConflict


@pytest.fixture
def store(tmp_path):
    path = tmp_path/'character_config.yaml'
    path.write_text('''# keep this explanation
your_name: User
runtime:
  provider: lm_studio # keep this comment
  n_ctx: 8192
  n_batch: 512
  n_ubatch: 512
  model_path: null
  hf_repo_id: null
  hf_filename: null
  n_threads: null
  type_v: f16
  flash_attn: false
voice:
  mode: wake_word
  wake_word: Riko
  wake_threshold: 0.9
presets:
  default:
    model_params:
      context_window_token_limit: 7168
      max_output_tokens: 1024
    memories: []
custom_extension:
  untouched: true
''', encoding='utf-8')
    return SettingsStore(path)


def test_round_trip_save_preserves_comments_and_unknown_settings(store):
    before = store.path.read_text()
    snapshot = store.snapshot()
    result = store.save({'your_name':'Senpai', 'voice.wake_threshold':.85}, snapshot['revision'])
    assert result['saved'] and result['restart_required']
    assert '# keep this explanation' in store.path.read_text()
    assert '# keep this comment' in store.path.read_text()
    assert result['values']['custom_extension.untouched'] is True
    assert store.path.with_suffix('.yaml.previous').read_text() == before


def test_unknown_keys_are_warnings_that_never_block_validation_or_saving(store):
    from process.app_core.configuration.settings_store import UNKNOWN
    assert 'warnings' not in store.validate({}) and 'warnings' not in store.snapshot()  # unchanged for a file without any
    store.path.write_text(store.path.read_text().replace('  wake_threshold: 0.9\n', '  wake_threshold: 0.9\n  wake_treshold: 0.5\n'))
    snapshot = store.snapshot()
    fields = {item['path']: item for item in snapshot['fields']}
    assert snapshot['warnings'] == {'voice.wake_treshold': UNKNOWN}
    assert fields['voice.wake_treshold']['unknown'] and fields['voice.wake_treshold']['help'] == UNKNOWN
    assert 'unknown' not in fields['voice.wake_threshold'] and 'unknown' not in fields['custom_extension.untouched']  # only the typed sections
    assert store.validate({}) == {'valid': True, 'errors': {}, 'warnings': {'voice.wake_treshold': UNKNOWN}}
    result = store.save({'voice.wake_threshold': .8}, snapshot['revision'])
    assert result['saved'] and result['warnings'] == {'voice.wake_treshold': UNKNOWN} and 'wake_treshold: 0.5' in store.path.read_text()


def test_wake_threshold_says_an_enrolled_wake_word_keeps_its_own(store):
    help = {item['path']: item for item in store.snapshot()['fields']}['voice.wake_threshold']['help']
    assert 'enrolled wake word keeps the threshold saved with it' in help and 'overrides this value' in help


def test_every_numeric_voice_speech_and_sovits_setting_has_an_editing_range_load_config_accepts():
    from dataclasses import fields
    from process.app_core.configuration.schema import setting
    from process.app_core.kernel.audio_config import SovitsConfig, SpeechConfig, VoiceConfig
    partners = {'voice.transcription_gap_seconds': {'utterance_end_seconds': 10}, 'voice.utterance_end_seconds': {'transcription_gap_seconds': .0005},
        'speech.split_window_words': {'max_words': 1000}, 'speech.max_words': {'split_window_words': 0}}  # load_config also checks each pair
    for section, cls in (('voice', VoiceConfig), ('speech', SpeechConfig), ('sovits_ping_config', SovitsConfig)):
        for item in fields(cls):
            path = f'{section}.{item.name}'
            if type(item.default) not in (int, float) and path != 'voice.input_device': continue
            assert setting(path).range, path
            for value in setting(path).range:  # None: no upper bound, as the loader has none
                if value is not None: cls.from_raw({item.name: value, **partners.get(path, {})})


def test_other_providers_neither_size_nor_write_a_kv_pool(store):
    store.path.write_text(store.path.read_text().replace('  n_ctx: 8192\n', "  n_ctx: 8192\n  parallel_slots: '3'\n"))
    snapshot = store.snapshot()
    assert snapshot['values']['runtime.kv_pool_tokens'] is None
    result = store.save({'memory.reflection_context_window_tokens': 8192}, snapshot['revision'])
    assert result['saved'] and 'kv_pool_tokens' not in store.path.read_text()


def test_background_budgets_are_edited_under_their_sections_not_as_runtime_fields(store):
    """RuntimeConfig declares initiative_n_ctx and the rest, but Settings keeps showing them as the YAML keys they come from."""
    snapshot = store.snapshot()
    names = {*snapshot['values'], *(item['path'] for item in snapshot['fields'])}
    assert not names & {'runtime.initiative_n_ctx', 'runtime.initiative_max_output_tokens', 'runtime.reflection_n_ctx'}
    assert snapshot['values']['initiative.context_window_tokens'] == 4096 and snapshot['values']['initiative.max_output_tokens'] == 1024


def test_validation_does_not_write_config_or_start_models(store):
    before = store.path.read_bytes()
    result = store.validate({'runtime.provider':'llama_cpp'})
    assert not result['valid']
    assert store.path.read_bytes() == before
    assert not list(store.path.parent.glob('.settings-validation-*'))


def test_replaced_preset_fields_are_not_editable_and_independent_values_are_preserved(store):
    snapshot=store.snapshot()
    paths={item['path'] for item in snapshot['fields']}
    assert not any(path.startswith('presets.default.model_params.') for path in paths)
    result=store.save({'runtime.temperature':.23},snapshot['revision'])
    assert result['saved']
    assert result['values']['runtime.temperature']==.23
    assert result['values']['presets.default.model_params.max_output_tokens']==1024
    assert result['values']['memory.context_window_tokens']==snapshot['values']['memory.context_window_tokens']
    result=store.save({'runtime.pause_background_on_live':False},result['revision'])
    assert result['saved'] and not result['restart_required']


@pytest.mark.parametrize('changes', [
    {'runtime.flash_attn':'false'}, {'runtime.n_ctx':-1}, {'runtime.n_batch':12.5},
    {'voice.wake_word':'two names'}, {'voice.wake_threshold':2}, {'unknown.setting':True},
    {'runtime.type_v':'not_a_type'}, {'presets.default.memories':[True]},
])
def test_invalid_drafts_never_replace_config(store, changes):
    before = store.path.read_bytes()
    result = store.save(changes, store.snapshot()['revision'])
    assert not result['saved'] and result['errors']
    assert store.path.read_bytes() == before


def test_conflicting_external_edit_is_preserved(store):
    snapshot = store.snapshot()
    store.path.write_text(store.path.read_text()+'\n# user edited outside UI\n')
    with pytest.raises(SettingsConflict): store.save({'your_name':'New name'}, snapshot['revision'])
    assert '# user edited outside UI' in store.path.read_text()


def test_nullable_threads_can_be_reset_after_saving(store):
    result = store.save({'runtime.n_threads':4}, store.snapshot()['revision'])
    spec = next(field for field in result['fields'] if field['path']=='runtime.n_threads')
    assert spec['nullable']
    assert store.save({'runtime.n_threads':None}, result['revision'])['saved']


def test_repository_model_and_cache_validation(store):
    valid = {'runtime.provider':'llama_cpp', 'runtime.hf_repo_id':'owner/model', 'runtime.hf_filename':'model.gguf'}
    assert store.validate(valid)['valid']
    result = store.validate({**valid,'runtime.type_v':'q8_0'})
    assert not result['valid']


def test_fixture_character_configuration_has_valid_settings(store):
    assert store.validate({}) == {'valid':True,'errors':{}}
    assert 'runtime.provider' in store.snapshot()['values']


def test_current_character_configuration_has_valid_settings():
    if os.environ.get('RIKO_RELEASE_BUILD') == '1':
        pytest.skip('Release builds exclude the private local configuration')
    path=Path(__file__).resolve().parents[1]/'character_config.yaml'
    if not path.is_file(): pytest.skip('No local character_config.yaml: it is private and gitignored, so a fresh clone has none')
    store=SettingsStore(path)
    assert store.validate({}) == {'valid':True,'errors':{}}
    assert 'runtime.provider' in store.snapshot()['values']


def test_background_budgets_are_exposed_and_validated(store):
    values = store.snapshot()['values']
    assert values['initiative.max_output_tokens'] == 1024
    assert values['memory.reflection_context_window_tokens'] == 4096
    assert store.validate({'initiative.max_output_tokens': 4096})['valid'] is False
    assert store.validate({'memory.reflection_max_output_tokens': 4096})['valid'] is False
    assert store.validate({'memory.reflection_max_output_tokens': 1536, 'memory.reflection_context_window_tokens': 8192})['valid']


def test_speech_recognition_pair_defaults_to_auto_and_is_checked_only_when_edited(store, monkeypatch):
    from process.app_core.audio import asr
    monkeypatch.setattr(asr, 'supported_types', lambda: {'cpu': frozenset({'float32', 'int8', 'int8_float32'})})
    values = store.snapshot()['values']
    assert (values['voice.asr_device'], values['voice.asr_compute_type']) == ('auto', 'default')
    assert store.validate({'voice.asr_device': 'cpu', 'voice.asr_compute_type': 'int8'})['valid']
    assert 'voice.asr_device' in store.validate({'voice.asr_device': 'cuda'})['errors']
    assert 'voice.asr_compute_type' in store.validate({'voice.asr_compute_type': 'int8_float16'})['errors']
    # A pair already in the file falls back at startup; it must not block saving anything else.
    store.path.write_text(store.path.read_text().replace('  wake_threshold: 0.9\n', '  wake_threshold: 0.9\n  asr_device: cuda\n  asr_compute_type: int8_float16\n'))
    assert store.save({'voice.wake_threshold': .8}, store.snapshot()['revision'])['saved']


def test_resource_fields_are_grouped_with_models(store):
    fields = {field['path']:field for field in store.snapshot()['fields']}
    for path in ('voice.asr_device','memory.embedding_model','emotion.max_length','memory.reflection_max_output_tokens','initiative.context_window_tokens','memory.system1_model_id'):
        assert fields[path]['group'] == 'models'
    assert fields['voice.wake_threshold']['group'] == 'voice'


def test_julia_revisions_are_listed_as_commits_that_need_a_restart(store):
    from process.app_core.kernel.model_pins import JULIA_REVISION
    snapshot = store.snapshot()
    fields = {field['path']: field for field in snapshot['fields']}
    for path in ('memory.system1_revision', 'emotion.revision'):
        assert fields[path]['group'] == 'models' and fields[path]['kind'] == 'text' and fields[path]['restart_scope'] == 'python'
        assert snapshot['values'][path] == JULIA_REVISION and 'never a branch' in fields[path]['help']
        assert 'full 40-character commit hash' in str(store.validate({path: 'main'}))


def test_barge_in_settings_are_listed_with_their_options_and_shown_only_where_they_matter(store):
    snapshot = store.snapshot()
    fields, values = {field['path']: field for field in snapshot['fields']}, snapshot['values']
    assert fields['voice.barge_in']['options'] == ['full_duplex', 'echo_gate', 'stop_only'] and values['voice.barge_in'] == 'full_duplex'
    assert (values['voice.barge_in_vad_threshold'], values['voice.echo_coupling'], values['voice.echo_tail_seconds']) == (.85, None, .3)
    assert fields['voice.echo_coupling']['visible_when'] == fields['voice.barge_in_vad_threshold']['visible_when'] == {'voice.barge_in': ['echo_gate']}
    assert fields['voice.echo_tail_seconds']['visible_when'] == {'voice.barge_in': ['echo_gate', 'stop_only']}
    assert all(fields[f'voice.{key}']['restart_scope'] == 'python' for key in ('barge_in', 'barge_in_vad_threshold', 'echo_coupling', 'echo_tail_seconds'))
    assert 'headphones' in fields['voice.barge_in']['help'] and fields['voice.echo_coupling']['nullable']
    assert store.validate({'voice.barge_in': 'echo_gate', 'voice.echo_coupling': None})['valid']
    assert 'voice.barge_in' in str(store.validate({'voice.barge_in': 'half_duplex'})['errors'])


def native(store):  # only llama_cpp allocates a KV pool, so only it has one to size
    store.path.write_text(store.path.read_text().replace('provider: lm_studio', 'provider: llama_cpp').replace('  model_path: null\n', '  model_path: test.gguf\n'))
    return store


def test_automatic_pool_is_recalculated_and_saved_with_other_settings(store):
    native(store)
    before = store.snapshot()
    result = store.save({'memory.reflection_context_window_tokens':8192,'runtime.kv_pool_auto':True}, before['revision'])
    assert result['saved']
    assert result['values']['runtime.kv_pool_tokens'] == 16384
    assert 'kv_pool_tokens: 16384' in store.path.read_text()
    assert 'kv_pool_auto: true' in store.path.read_text()


def test_saved_pool_and_initiative_budget_preserve_live_preferences(store):
    import json
    preferences = store.path.parent / 'persistent_memories' / 'initiative_settings.json'
    preferences.parent.mkdir()
    preferences.write_text(json.dumps({'enabled':True,'context_window_tokens':8192,'max_output_tokens':1536}))
    before = native(store).snapshot()
    assert before['values']['initiative.context_window_tokens'] == 8192
    result = store.save({'initiative.context_window_tokens':6144}, before['revision'])
    assert result['saved']
    assert result['values']['runtime.kv_pool_tokens'] == 14336
    assert result['values']['initiative.context_window_tokens'] == 6144
    assert json.loads(preferences.read_text())['enabled'] is True
    assert json.loads(preferences.read_text())['max_output_tokens'] == 1536


def test_missing_default_settings_can_be_saved_into_new_yaml_sections(store):
    snapshot=store.snapshot()
    assert snapshot['values']['memory.embeddings_enabled'] is True
    result=store.save({'memory.embeddings_enabled':False},snapshot['revision'])
    assert result['saved'] and result['values']['memory.embeddings_enabled'] is False


def test_explicit_path_check_never_reads_file_contents(store):
    result=store.check_path('memory.store_file', str(store.path))
    assert result['exists'] and not result['directory']
    assert 'contents' not in result


def test_path_check_is_only_for_file_settings_and_never_contacts_a_network_path(store, monkeypatch):
    with pytest.raises(ValueError, match='only for file and folder settings'): store.check_path('runtime.n_ctx', str(store.path))
    with pytest.raises(ValueError, match='only for file and folder settings'): store.check_path('no.such_file', str(store.path))
    touched = []
    for name in ('resolve', 'exists', 'is_dir', 'stat'):
        original = getattr(Path, name)
        def spy(self, *args, _original=original, **kwargs):
            if 'attacker' in str(self): touched.append(str(self)); raise AssertionError('a network path reached the filesystem')
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(Path, name, spy)
    for value in ('//attacker/share/x.gguf', r'\\attacker\share\x.gguf'):
        with pytest.raises(ValueError, match='Network and device paths'): store.check_path('memory.store_file', value)
    assert not touched
    calls = []
    monkeypatch.setattr(shutil, 'which', lambda value, *args, **kwargs: calls.append(value) or '/usr/bin/' + value)
    assert store.check_path('sovits_ping_config.executable', 'sovits') == {'resolved': '/usr/bin/sovits', 'exists': True, 'directory': False, 'executable': True}
    store.check_path('memory.store_file', 'memory_store.json'); store.check_path('sovits_ping_config.executable', 'bin/sovits')
    assert calls == ['sovits']  # only a bare command of an executable setting is looked up on PATH
    relative = store.check_path('memory.store_file', 'persistent_memories/../memory_store.json')
    assert relative == {'resolved': str(store.path.parent / 'memory_store.json'), 'exists': False, 'directory': False, 'executable': False}


def test_secrets_are_masked_and_never_sent_back(store, monkeypatch):
    from process.app_core.configuration.settings_store import SECRET_MASK
    monkeypatch.delenv('RIKO_API_KEY', raising=False)
    assert store.snapshot()['values']['runtime.api_key'] == 'local'  # the default is no secret
    store.path.write_text(store.path.read_text().replace('runtime:\n', 'runtime:\n  api_key: sk-saved-secret-123456\n'))
    snapshot = store.snapshot()
    field = next(item for item in snapshot['fields'] if item['path'] == 'runtime.api_key')
    assert snapshot['values']['runtime.api_key'] == SECRET_MASK and field['secret'] and field['secret_state'] == 'saved'
    assert 'sk-saved-secret' not in json.dumps(snapshot) and 'path' not in snapshot  # nor the config's absolute path
    assert store.stored_values()['runtime.api_key'] == 'sk-saved-secret-123456'  # the backend's own comparisons
    # Saving other settings with the mask untouched keeps the key; the result is masked too.
    result = store.save({'runtime.api_key': SECRET_MASK, 'runtime.n_ctx': 4096}, snapshot['revision'])
    assert result['saved'] and 'api_key: sk-saved-secret-123456' in store.path.read_text() and 'sk-saved' not in json.dumps(result)
    assert not result['restart_required'] or 'runtime.api_key' not in result['restart_pending']
    assert store.validate({'runtime.api_key': SECRET_MASK})['valid']
    refused = store.save({'runtime.api_key': SECRET_MASK + 'x'}, result['revision'])
    assert not refused['saved'] and refused['errors']['runtime.api_key'] == 'Type the whole new value, or clear the field'
    replaced = store.save({'runtime.api_key': 'sk-new-key-654321'}, result['revision'])
    assert replaced['saved'] and 'sk-new-key-654321' in store.path.read_text() and replaced['values']['runtime.api_key'] == SECRET_MASK
    cleared = store.save({'runtime.api_key': ''}, replaced['revision'])
    assert cleared['saved'] and "api_key: ''" in store.path.read_text()


def test_a_secret_from_the_environment_or_a_top_level_key_is_never_sent(store, monkeypatch):
    from process.app_core.configuration.settings_store import EXTERNAL, SECRET_MASK
    monkeypatch.setenv('RIKO_API_KEY', 'sk-environment-secret-42')
    snapshot = store.snapshot()
    field = next(item for item in snapshot['fields'] if item['path'] == 'runtime.api_key')
    assert snapshot['values']['runtime.api_key'] == '' and field['secret_state'] == 'external' and EXTERNAL in field['help']
    assert 'sk-environment-secret' not in json.dumps(snapshot)
    assert store.save({'runtime.n_ctx': 4096}, snapshot['revision'])['saved'] and 'api_key' not in store.path.read_text()
    monkeypatch.delenv('RIKO_API_KEY')
    store.path.write_text('api_key: sk-legacy-top-level-77\n' + store.path.read_text())
    snapshot = store.snapshot()
    assert snapshot['values']['api_key'] == SECRET_MASK and snapshot['values']['runtime.api_key'] == ''
    assert 'sk-legacy-top-level' not in json.dumps(snapshot)


def test_a_secret_pyyaml_reads_as_a_number_is_masked_and_kept(store, monkeypatch):
    from process.app_core.configuration.settings_store import SECRET_MASK
    monkeypatch.delenv('RIKO_API_KEY', raising=False)
    store.path.write_text(store.path.read_text().replace('runtime:\n', 'runtime:\n  api_key: 123456789012345\n') + 'discord:\n  password: 87654321\n')
    snapshot = store.snapshot()
    field = next(item for item in snapshot['fields'] if item['path'] == 'runtime.api_key')
    assert snapshot['values']['runtime.api_key'] == SECRET_MASK and field['kind'] == 'text' and field['secret'] and field['secret_state'] == 'saved'
    assert snapshot['values']['discord.password'] == SECRET_MASK  # an unknown secret-like key too
    assert '123456789012345' not in json.dumps(snapshot) and '87654321' not in json.dumps(snapshot)
    result = store.save({'runtime.api_key': SECRET_MASK, 'runtime.n_ctx': 4096}, snapshot['revision'])
    assert result['saved'] and 'api_key: 123456789012345' in store.path.read_text()
    replaced = store.save({'runtime.api_key': '555'}, result['revision'])
    assert replaced['saved'] and "api_key: '555'" in store.path.read_text()


def test_previous_config_backup_is_owner_only(store):
    if os.name == 'nt': pytest.skip('POSIX permission bits')
    backup = store.path.with_suffix('.yaml.previous')
    backup.write_text('old'); backup.chmod(0o644)
    assert store.save({'runtime.n_ctx': 4096}, store.snapshot()['revision'])['saved']
    assert backup.stat().st_mode & 0o777 == 0o600 and 'n_ctx: 8192' in backup.read_text()


def test_concurrent_same_name_tool_results_update_the_correct_activity():
    from process.app_core.desktop.state import DesktopState
    state=DesktopState()
    first=state.tool_started('lookup', {'id':'first'})
    second=state.tool_started('lookup', {'id':'second'})
    state.tool_finished('lookup','first result',activity_id=first)
    activities={item['id']:item for item in state.snapshot()['tools']}
    assert activities[first]['status']=='complete'
    assert activities[second]['status']=='running'
    assert activities[first]['duration_ms']>=0


def test_legacy_faiss_index_file_is_not_offered(store):
    assert 'memory.index_file' not in {field['path'] for field in store.snapshot()['fields']}


def test_memory_device_is_a_background_model_setting(store):
    snapshot = store.snapshot()
    spec = next(item for item in snapshot['fields'] if item['path'] == 'memory.device')
    assert snapshot['values']['memory.device'] == 'cpu' and spec['section'] == 'Background models' and spec['restart']
    assert store.validate({'memory.device': 'mps', 'emotion.device': 'auto'})['valid']
    assert not store.validate({'memory.device': 'vulkan'})['valid']
    assert not store.validate({'emotion.device': 'mps'})['valid']


def test_flash_attention_is_a_choice_that_keeps_old_boolean_files(store):
    snapshot = store.snapshot()
    spec = next(item for item in snapshot['fields'] if item['path'] == 'runtime.flash_attn')
    assert snapshot['values']['runtime.flash_attn'] == 'off' and spec['kind'] == 'text' and spec['options'] == ['auto', 'on', 'off']
    result = store.save({'runtime.flash_attn': 'on'}, snapshot['revision'])
    assert result['saved'] and result['values']['runtime.flash_attn'] == 'on'
    assert 'flash_attn: on' in store.path.read_text()  # ruamel writes it bare, so PyYAML reads true
    from process.app_core.configuration.config import load_config
    native = store.path.with_name('native.yaml')
    native.write_text('runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n  flash_attn: on\n')
    assert load_config(native).runtime.flash_attn == 'on'
    assert not store.save({'runtime.flash_attn': True}, store.snapshot()['revision'])['saved']


def test_layer_offload_and_split_choices_match_llama_cpp(store):
    assert store.validate({'runtime.n_gpu_layers': -2})['valid']
    assert not store.validate({'runtime.n_gpu_layers': -3})['valid']
    assert not store.validate({'runtime.split_mode': 'row'})['valid']
    store.path.write_text(store.path.read_text().replace('provider: lm_studio', 'provider: llama_cpp')
        .replace('  model_path: null', '  model_path: model.gguf').replace('  flash_attn: false', '  flash_attn: false\n  split_mode: row'))
    assert store.snapshot()['values']['runtime.split_mode'] == 'row'  # an old file still opens
    assert 'runtime.split_mode' in store.validate({})['errors']


def test_speech_recognition_error_names_the_value_that_cannot_run(store, monkeypatch):
    from process.app_core.audio import asr
    monkeypatch.setattr(asr, 'supported_types', lambda: {'cpu': frozenset({'float32', 'int8'}), 'cuda': frozenset({'float32', 'int8', 'int8_float16'})})
    store.path.write_text(store.path.read_text().replace('  wake_threshold: 0.9\n', '  wake_threshold: 0.9\n  asr_device: cuda\n  asr_compute_type: int8_float16\n'))
    errors = store.validate({'voice.asr_device': 'cpu'})['errors']  # CPU is fine; the stored precision is what cannot run there
    assert 'voice.asr_device' not in errors and 'cpu cannot run int8_float16' in errors['voice.asr_compute_type']
    assert store.validate({'voice.asr_device': 'cpu', 'voice.asr_compute_type': 'default'})['valid']


def test_wake_word_is_the_phrase_the_runtime_uses_and_stays_editable_text(tmp_path):
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\npresets:\n  default:\n    name: Riko Chan\n')
    store = SettingsStore(path)
    snapshot = store.snapshot()  # a two-word name wakes on its first word, which Settings shows and can change
    assert snapshot['values']['voice.wake_word'] == 'Riko'
    assert {item['path']: item for item in snapshot['fields']}['voice.wake_word']['kind'] == 'text'
    assert store.validate({'speech.max_words': 20}) == {'valid': True, 'errors': {}}
    assert 'one short wake name' in store.validate({'voice.wake_word': 'Riko Chan'})['errors']['voice.wake_word']
    # PyYAML, which the runtime uses, reads an unquoted number or yes as a number or boolean, not a word.
    for value in ('42', 'yes'):
        path.write_text(f'runtime:\n  provider: lm_studio\nvoice:\n  wake_word: {value}\n')
        snapshot = store.snapshot()
        assert {item['path']: item for item in snapshot['fields']}['voice.wake_word']['kind'] == 'text'
        assert 'not text' in store.validate({'speech.max_words': 20})['errors']['voice.wake_word']
        result = store.save({'voice.wake_word': 'Mita'}, snapshot['revision'])
        assert result['saved'] and load_config(path).raw['voice']['wake_word'] == 'Mita'
    result = store.save({'voice.wake_word': 'Yes'}, result['revision'])  # written quoted, so it stays a word
    assert result['saved'] and load_config(path).raw['voice']['wake_word'] == 'Yes'


def test_live_budget_is_checked_only_when_edited_so_older_setups_can_save(tmp_path, monkeypatch):
    # The setup wizard used to write memory.context_window_tokens equal to n_ctx, which never leaves room for the reply.
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(tmp_path / 'application'))
    path = tmp_path / 'character_config.yaml'
    path.write_text(f'runtime:\n  provider: llama_cpp\n  native_library: bundled:{backends_for()[0]}\n  hf_repo_id: owner/model\n  hf_filename: model.gguf\n'
        '  n_ctx: 8192\n  max_output_tokens: 1024\nmemory:\n  context_window_tokens: 8192\n')
    store = SettingsStore(path)
    assert store.validate({'speech.max_words': 20}) == {'valid': True, 'errors': {}}
    assert 'Context must fit' in store.validate({'runtime.max_output_tokens': 512})['errors']['runtime.n_ctx']
    assert store.validate({'memory.context_window_tokens': 7168})['valid']
    path.write_text(path.read_text().replace('memory:\n  context_window_tokens: 8192\n', ''))
    assert store.snapshot()['values']['memory.context_window_tokens'] == 7168  # the prompt gets what the reply leaves
    assert store.validate({'runtime.n_ctx': 4096})['valid']
    assert 'runtime.n_ctx' in store.validate({'runtime.max_output_tokens': 8192})['errors']


ROOT = Path(__file__).resolve().parents[1]


def packaged_setup(data, form, resources):
    """The packaged first run end to end: Electron's saveSetup (electron/release.cjs) asks this tree's backend, as it asks
    the frozen one (`--setup-config`), and writes its answer."""
    script = ("require('./release.cjs').saveSetup(process.argv[1], JSON.parse(process.argv[2]), process.argv[3], process.argv[3], process.platform,"
        " JSON.parse(process.argv[4])).catch(error => {console.error(error.message); process.exit(1);})")
    command = [sys.executable, str(ROOT / 'Code' / 'run_server.py')]
    subprocess.run(['node', '-e', script, str(data), json.dumps(form), str(resources), json.dumps(command)], cwd=ROOT / 'electron', check=True, timeout=300)


@pytest.mark.skipif(not shutil.which('node') or not (ROOT / 'electron' / 'node_modules' / 'yaml').is_dir(), reason='needs node and the Electron dependencies')
def test_packaged_setup_writes_a_config_settings_can_save(tmp_path, monkeypatch):
    # The backend refuses a bundle whose library is not where bundled:<backend> resolves, and Electron passes it the
    # resources it will later start the backend with, so this also proves both use the same file for this OS.
    resources, backend = tmp_path / 'application', backends_for()[0]
    library = bundled_library(backend, resources)
    library.parent.mkdir(parents=True); library.write_text('test')
    form = {'backend': backend, 'repo': 'owner/model', 'filename': 'model.gguf', 'context': 8192, 'output': 1024, 'threads': 4}
    packaged_setup(tmp_path / 'data', form, resources)
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(resources))
    store = SettingsStore(tmp_path / 'data' / 'character_config.yaml')
    assert load_config(store.path).runtime.native_library == library
    assert store.validate({'runtime.n_ctx': 8192}) == {'valid': True, 'errors': {}}  # the wizard's budget fits
    assert store.validate({'speech.max_words': 20}) == {'valid': True, 'errors': {}}


@pytest.mark.skipif(not shutil.which('node') or not (ROOT / 'electron' / 'node_modules' / 'yaml').is_dir(), reason='needs node and the Electron dependencies')
@pytest.mark.parametrize('name, wake', [('Riko Chan', 'Riko'), ('Yes', 'Yes'), ('', 'Riko')])
def test_packaged_setup_name_gives_a_usable_wake_word(tmp_path, monkeypatch, name, wake):
    """Regression: setup accepts any companion name, and a two-word one (or Yes, which PyYAML read as a boolean) stopped
    the backend in WakeWord after the model had loaded."""
    from process.app_core.audio.wake_word import WakeWord
    resources, backend = tmp_path / 'application', backends_for()[0]
    library = bundled_library(backend, resources)
    library.parent.mkdir(parents=True); library.write_text('test')
    form = {'backend': backend, 'repo': 'owner/model', 'filename': 'model.gguf', 'context': 8192, 'output': 1024, 'threads': 4, 'name': name}
    packaged_setup(tmp_path / 'data', form, resources)
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(resources))
    store = SettingsStore(tmp_path / 'data' / 'character_config.yaml')
    config = load_config(store.path)
    assert config.character_name == (name or 'Riko') and config.raw['voice']['wake_word'] == wake
    detector = WakeWord(config)
    try: assert (detector.phrase, detector.unavailable) == (wake, '')
    finally: detector.close()
    # Saving another setting keeps the quotes setup wrote, so a companion named Yes stays text for PyYAML.
    assert store.save({'speech.max_words': 20}, store.snapshot()['revision'])['saved']
    config = load_config(store.path)
    assert config.character_name == (name or 'Riko') and config.raw['voice']['wake_word'] == wake


def test_legacy_preset_context_leaves_room_for_the_reply(tmp_path):
    from process.app_core.configuration.config import load_config
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n  max_output_tokens: 1024\npresets:\n  default:\n    model_params:\n      context_window_token_limit: 8192\n')
    config = load_config(path)
    assert config.runtime.n_ctx == 8192 and config.memory.context_window_tokens == 7168
    assert SettingsStore(path).validate({'runtime.max_output_tokens': 512})['valid']


def test_stored_values_the_backend_accepts_never_block_saving_other_settings(tmp_path):
    """Regression: Settings ranges narrower than load_config's, or integer inferred from a whole-number default, made a
    working stored value an error the Settings page shows on open, which disables Save for every setting."""
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\nvoice:\n  max_segment_seconds: 180\n  vad_threshold: 0.97\n'
        '  interjection_debounce_seconds: 0.05\n  playback_words_per_second: 12\ninitiative:\n  interval_seconds: 90.5\n', encoding='utf-8')
    store = SettingsStore(path)
    fields = {field['path']: field for field in store.snapshot()['fields']}
    for key, value in (('voice.max_segment_seconds', 180), ('voice.vad_threshold', .97), ('voice.interjection_debounce_seconds', .05), ('voice.playback_words_per_second', 12)):
        field = fields[key]
        assert (field.get('min') is None or field['min'] <= value) and (field.get('max') is None or value <= field['max']), key
    assert fields['initiative.interval_seconds']['integer'] is False
    assert store.validate({'runtime.temperature': .5})['valid'] and store.validate({'initiative.interval_seconds': 45.5})['valid']


def test_a_section_whose_keys_are_all_commented_out_can_be_edited(tmp_path):
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\nspeech:\n#  max_words: 40\n', encoding='utf-8')  # PyYAML reads speech: null
    store = SettingsStore(path)
    assert store.validate({'speech.max_words': 30}) == {'valid': True, 'errors': {}}
    store.save({'speech.max_words': 30}, store.snapshot()['revision'])
    assert 'max_words: 30' in path.read_text(encoding='utf-8')


def test_settings_lists_the_avatar_model_above_the_animation_engine(tmp_path):
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\n', encoding='utf-8')  # neither section in the file, as after first setup
    sections = [field['section'] for field in SettingsStore(path).snapshot()['fields'] if field['section'] in ('Avatar model', 'Animation engine')]
    assert sections.index('Avatar model') < sections.index('Animation engine')


def test_a_key_the_files_provider_does_not_use_is_not_judged(tmp_path):
    """A leftover runtime.api_mode from an OpenAI setup loads and runs on llama_cpp; checking the whole file (first run,
    --validate-config) must not reject it, as Settings no longer does for a field its provider hides."""
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: llama_server\n  api_mode: chat\n', encoding='utf-8')
    assert SettingsStore(path).validate_file() == {'valid': True, 'errors': {}}
