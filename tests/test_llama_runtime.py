from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from process.app_core.configuration.config import RuntimeConfig, load_config
from process.app_core.inference.llama_runtime import validate_runtime, resolve_model


def native(**kwargs):
    return RuntimeConfig(provider='llama_cpp', model_path=Path('model.gguf'), **kwargs)


@pytest.mark.parametrize('settings', [
    {'n_ctx': -1}, {'n_threads': 0}, {'flash_attn': 'true'}, {'n_ubatch': 1024},
    {'type_k': 'q4_k'}, {'type_v': 'q8_0'}, {'tensor_split': [float('nan')]},
    {'tensor_split': [0, 0]}, {'cache_size_mb': -1}, {'split_mode': 'invalid'},
])
def test_invalid_native_settings(settings):
    with pytest.raises(ValueError): validate_runtime(native(**settings))


def test_yaml_native_settings_and_paths(tmp_path):
    config = tmp_path / 'character_config.yaml'
    config.write_text('''runtime:
  provider: llama_cpp
  model_path: models/example.gguf
  n_ctx: 16384
  n_batch: 256
  flash_attn: true
  type_k: q8_0
  type_v: q8_0
  n_threads: 4
  cache_size_mb: 256
''')
    runtime = load_config(config).runtime
    assert runtime.model_path == tmp_path/'models/example.gguf'
    assert runtime.n_ctx == 16384 and runtime.n_ubatch == 256
    assert runtime.flash_attn and runtime.type_v == 'q8_0'
    assert runtime.cache_size_mb == 256


def test_hf_download_uses_normal_cache_and_explicit_revision(monkeypatch):
    calls = []
    def download(**kwargs): calls.append(kwargs); return '/cache/model.gguf'
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    config = replace(native(), model_path=None, hf_repo_id='owner/repo', hf_filename='model.gguf', hf_revision='commit', hf_local_files_only=True)
    validate_runtime(config)
    assert resolve_model(config) == Path('/cache/model.gguf')
    assert calls == [dict(repo_id='owner/repo', filename='model.gguf', revision='commit', local_files_only=True)]
    assert resolve_model(native()) == Path('model.gguf')
    assert len(calls) == 1 # Local path never requests HF.


def test_split_gguf_downloads_all_shards_at_same_commit(monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(Path('/cache/snapshots/abc123') / kwargs['filename'])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    config = replace(native(), model_path=None, hf_repo_id='owner/repo', hf_filename='sub/model-00001-of-00003.gguf')
    resolve_model(config)
    assert len(calls) == 3
    assert calls[1]['revision'] == 'abc123'
    assert calls[2]['filename'] == 'sub/model-00003-of-00003.gguf'


@pytest.mark.parametrize('filename', ['*.gguf', '../model.gguf', '/model.gguf', 'model.bin'])
def test_hf_requires_unambiguous_filename(filename):
    with pytest.raises(ValueError):
        validate_runtime(replace(native(), model_path=None, hf_repo_id='owner/repo', hf_filename=filename))
