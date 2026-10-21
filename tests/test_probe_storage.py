from types import SimpleNamespace

from process.app_core.configuration.paths import DataPaths
from process.app_core.emotion.probe_storage import corpus_files, probe_directory, training_directory


def test_model_weights_and_training_have_separate_per_model_paths(tmp_path):
    runtime = SimpleNamespace(model_path=None, hf_repo_id='publisher/model-name')
    paths = DataPaths.at(tmp_path)
    assert probe_directory(paths, runtime) == tmp_path / 'models/model-name/expression probe'
    assert training_directory(paths, runtime) == tmp_path / 'models/training/expression/model-name'
    runtime.model_path = tmp_path / 'local-model.gguf'
    assert training_directory(paths, runtime).name == 'local-model'


def test_model_names_cannot_escape_storage_root(tmp_path):
    runtime = SimpleNamespace(model_path=None, hf_repo_id='publisher/..')
    assert training_directory(DataPaths.at(tmp_path), runtime) == tmp_path / 'models/training/expression/unknown-model'


def test_corpus_discovery_prefers_training_and_preserves_legacy(tmp_path):
    key = 'a' * 64
    new = tmp_path / 'models/training/expression/model-name' / key / 'examples.json'
    old = tmp_path / 'persistent_memories/emotion_probes' / key / 'examples.json'
    for path in (new, old):
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    assert list(corpus_files(DataPaths.at(tmp_path))) == [new]
    assert old.exists()
