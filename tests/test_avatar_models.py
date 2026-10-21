from pathlib import Path

import pytest

from process.app_core.desktop.avatar_models import AvatarModels, inspect_model
from process.app_core.configuration.settings_store import SettingsStore
from conftest import vrm_bytes  # shared with the desktop API tests


def test_vrm_import_copies_originals_and_never_overwrites_colliding_names(tmp_path):
    library = AvatarModels(tmp_path)
    source = tmp_path/'my avatar.vrm'; source.write_bytes(vrm_bytes())
    first = library.import_model(source)
    second = library.import_model(source)
    assert first == {'path':'character_files/models/my avatar.vrm', 'format':'vrm1'}
    assert second['path'] == 'character_files/models/my avatar-1.vrm'
    assert source.read_bytes() == (tmp_path/first['path']).read_bytes() == (tmp_path/second['path']).read_bytes()
    assert library.import_model(first['path']) == first
    assert len(library.listing()['entries']) == 2
    assert library.validate(first['path'], 'auto')[1] == 'vrm1'
    with pytest.raises(ValueError, match='matching format'): library.validate(first['path'], 'vrm0')


@pytest.mark.parametrize('version', ['vrm0','vrm1'])
def test_vrm_formats_are_detected_from_file_contents(tmp_path, version):
    source = tmp_path/'model.VRM'; source.write_bytes(vrm_bytes(version))
    assert inspect_model(source) == version


@pytest.mark.parametrize('data', [b'not a model', vrm_bytes(uri='https://example.com/texture.png'), vrm_bytes(uri='../private.png'), vrm_bytes()[:-1]])
def test_invalid_or_external_resource_models_are_not_imported(tmp_path, data):
    source = tmp_path/'bad.vrm'; source.write_bytes(data)
    library = AvatarModels(tmp_path)
    with pytest.raises(ValueError): library.import_model(source)
    assert not library.directory.exists()
    assert source.read_bytes() == data


def test_vrm_selection_cannot_escape_character_files(tmp_path):
    source = tmp_path/'outside.vrm'; source.write_bytes(vrm_bytes())
    with pytest.raises(ValueError, match='Import'): AvatarModels(tmp_path).validate(str(source))


def test_avatar_settings_validate_and_persist_format_without_modifying_models(tmp_path):
    config = tmp_path/'character_config.yaml'
    config.write_text('runtime:\n  provider: lm_studio\n')
    source = tmp_path/'source.vrm'; source.write_bytes(vrm_bytes('vrm0'))
    model = AvatarModels(tmp_path).import_model(source)['path']
    store = SettingsStore(config)
    snapshot = store.snapshot()
    assert snapshot['values']['avatar.format'] == 'auto'
    fields = {item['path']:item for item in snapshot['fields']}
    assert fields['avatar.model']['group'] == 'appearance'
    assert fields['avatar.format']['options'] == ['auto','vrm0','vrm1']
    invalid = store.save({'avatar.model':model,'avatar.format':'vrm1'},snapshot['revision'])
    assert not invalid['saved'] and 'avatar.model' in invalid['errors']
    result = store.save({'avatar.model':model,'avatar.format':'auto'},snapshot['revision'])
    assert result['saved'] and not result['restart_required'] and result['values']['avatar.model'] == model
    assert (tmp_path/model).read_bytes() == source.read_bytes()
