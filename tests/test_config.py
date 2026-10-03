from process.app_core.configuration.config import load_config


def test_missing_config_has_safe_defaults(tmp_path):
    config = load_config(tmp_path / "missing.yaml")
    assert config.character_name == "Riko"
    assert config.memory.history_file.parent == tmp_path / "persistent_memories"
    assert config.runtime.n_ctx == 8192
