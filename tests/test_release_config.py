import sys
import pytest
from process.app_core.configuration.config import load_config


def test_bundled_backend_resolves_current_install_location(tmp_path, monkeypatch):
    config = tmp_path / 'data' / 'character_config.yaml'
    config.parent.mkdir()
    config.write_text('runtime:\n  provider: llama_cpp\n  native_library: bundled:cuda\n  hf_repo_id: owner/model\n  hf_filename: model.gguf\n')
    resources = tmp_path / 'application'
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(resources))
    loaded = load_config(config)
    assert loaded.root == config.parent
    assert loaded.runtime.native_library == resources / 'native/cuda' / ('riko-native.dll' if sys.platform == 'win32' else 'libriko-native.so')
    assert loaded.memory.store_file.is_relative_to(config.parent)
    moved = tmp_path / 'upgraded-app'
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(moved))
    assert load_config(config).runtime.native_library.is_relative_to(moved)


def test_bundled_backend_rejects_untrusted_names(tmp_path, monkeypatch):
    config = tmp_path / 'character_config.yaml'
    config.write_text('runtime:\n  native_library: bundled:../../other\n')
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='Bundled native'): load_config(config)
