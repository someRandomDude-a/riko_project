from pathlib import Path
import os

import pytest

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


def test_resource_fields_are_grouped_with_models(store):
    fields = {field['path']:field for field in store.snapshot()['fields']}
    for path in ('voice.asr_device','memory.embedding_model','emotion.max_length','memory.reflection_max_output_tokens','initiative.context_window_tokens','memory.system1_model_id'):
        assert fields[path]['group'] == 'models'
    assert fields['voice.wake_threshold']['group'] == 'voice'


def test_automatic_pool_is_recalculated_and_saved_with_other_settings(store):
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
    before = store.snapshot()
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
    result=store.check_path(str(store.path))
    assert result['exists'] and not result['directory']
    assert 'contents' not in result


def test_concurrent_same_name_tool_results_update_the_correct_activity():
    from process.app_core.desktop.state import DesktopState
    state=DesktopState()
    first=state.tool_started('lookup', {'id':'first'})
    second=state.tool_started('lookup', {'id':'second'})
    state.tool_finished('lookup','first result',activity_id=first)
    activities={item['id']:item for item in state.tool_activity}
    assert activities[first]['status']=='complete'
    assert activities[second]['status']=='running'
    assert activities[first]['duration_ms']>=0
