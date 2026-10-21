"""Validated configuration with stable paths and environment overrides."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..kernel.audio_config import SovitsConfig, SpeechConfig, VoiceConfig
from ..kernel.background_budget import validate_budget
from ..kernel.code_paths import CodePaths
from ..kernel.model_pins import JULIA_MODEL_ID, JULIA_REVISION
from . import schema
from .paths import DataPaths, resolve

try:
    import yaml
except ImportError:  # Keep core imports usable for tooling/tests before dependencies are installed.
    yaml = None


@dataclass(slots=True)  # an undeclared field fails when set (tests/test_declared_attributes.py)
class RuntimeConfig:
    provider: str = "openai"
    model: str = ""
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "local"
    model_path: Path | None = None
    tokenizer_model: str | None = None
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    n_batch: int = 512
    temperature: float = 0.7
    max_output_tokens: int = 1024
    seed: int = -1
    api_mode: str = "auto"
    reuse_response_ids: bool = True
    request_timeout_seconds: float = 120.0
    hf_repo_id: str | None = None
    hf_filename: str | None = None
    hf_revision: str = 'main'
    hf_local_files_only: bool = False
    n_ubatch: int = 512
    n_threads: int | None = None
    n_threads_batch: int | None = None
    flash_attn: str | bool = 'auto'  # auto, on or off; YAML true/false still mean on/off
    type_k: str = 'f16'
    type_v: str = 'f16'
    offload_kqv: bool = True
    use_mmap: bool = True
    use_mlock: bool = False
    main_gpu: int = 0
    split_mode: str = 'layer'
    tensor_split: list[float] | None = None
    chat_format: str | None = None
    cache_size_mb: int = 0
    verbose: bool = False
    parallel_slots: int = 2
    kv_unified: bool = True
    kv_pool_auto: bool = True
    kv_pool_tokens: int | None = None
    startup_timeout_seconds: float = 600.0
    warmup: bool = True
    pause_background_on_live: bool = True
    native_library: Path | None = None
    # Background lanes' budgets, set by load_config from initiative.* (initiative_settings.json overrides them) and
    # memory.reflection_context_window_tokens; Settings edits them under those keys, not as runtime.* (configuration/schema.py TYPED).
    initiative_n_ctx: int = 4096
    initiative_max_output_tokens: int = 1024
    reflection_n_ctx: int = 4096


@dataclass
class ToolConfig:
    mcp_config: Path | None = None
    max_iterations: int = 8
    timeout_seconds: float = 30.0
    require_approval: bool = False
    best_fit_inputs: bool = True
    best_fit_timeout_seconds: float = .4
    best_fit_min_confidence: float = .85
    disabled: tuple = ()  # tool names Riko never sees or calls (tools.disabled): fewer prompt tokens on every turn


@dataclass
class MemoryConfig:
    # The three files are AppConfig.paths' chat_history, memory_store and memory_index once in an AppConfig (__post_init__).
    history_file: Path = Path("persistent_memories/chat_history.json")
    context_window_tokens: int = 8192
    store_file: Path = Path("persistent_memories/memory_store.json")
    index_file: Path = Path("persistent_memories/faiss_index.index")  # legacy: no longer written or read
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    max_results: int = 8
    token_budget: int = 1200
    reflection_enabled: bool = True
    reflection_min_importance: float = 0.5
    reflection_max_output_tokens: int = 1024
    reflection_context_window_tokens: int = 4096
    embeddings_enabled: bool = True
    system1_enabled: bool = True
    system1_model_id: str = JULIA_MODEL_ID
    system1_revision: str = JULIA_REVISION  # a commit: Julia-1 ships code that runs here (kernel/model_pins.py)
    system1_cache_dir: Path | None = None
    system1_max_length: int = 8192
    device: str = "cpu"
    minimum_importance: float = 0.35
    default_memories: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EmotionConfig:
    enabled: bool = False
    model_path: Path | None = None
    model_id: str = JULIA_MODEL_ID
    revision: str = JULIA_REVISION  # a commit, as memory.system1_revision
    cache_dir: Path | None = None
    device: str = "cpu"
    strict_encoding: bool = True
    max_length: int = 8192
    context_tokens: int = 1024
    update_interval_tokens: int = 16
    temperature: float = 0.2
    fallback: bool = True
    probe: dict = field(default_factory=dict)


@dataclass
class AppConfig:
    root: Path
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    sovits: SovitsConfig = field(default_factory=SovitsConfig)  # sovits_ping_config
    unknown_settings: tuple = ()  # section.key paths no typed section declares: logged once and flagged in Settings
    load_errors: tuple = ()  # sections loaded with their defaults by load_config(recover=...): the lifespan starts in setup mode
    character_name: str = "Riko"
    system_prompt: str = "You are a helpful local assistant."
    raw: dict[str, Any] = field(default_factory=dict)
    avatar: dict[str, Any] = field(default_factory=dict)
    paths: DataPaths | None = None  # every data location (configuration/paths.py); load_config builds it from the YAML

    def __post_init__(self):
        if self.paths is None:  # built directly (tests, tools): the default layout under root, with the memory section's files
            files = {'history_file': self.memory.history_file, 'store_file': self.memory.store_file, 'index_file': self.memory.index_file}
            self.paths = DataPaths.build(Path(self.root) / 'character_config.yaml', {'memory': files})
        # One location per store: the memory section's files are the paths' (relative defaults would follow the cwd).
        self.memory = replace(self.memory, history_file=self.paths.chat_history, store_file=self.paths.memory_store, index_file=self.paths.memory_index)


def load_config(path: str | Path | None = None, *, recover: bool | str = False) -> AppConfig:
    """recover: True (Settings' snapshot) or 'setup' (startup, only with desktop.setup_on_startup_error) loads a section
    whose check fails with its defaults and lists the message in load_errors, so the backend stays up to repair it."""
    config_path = Path(path or os.getenv("RIKO_CONFIG", "character_config.yaml")).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    raw: dict[str, Any] = {}
    if config_path.exists():
        if yaml is None:
            raise RuntimeError("PyYAML is required to load character_config.yaml")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = config_path.parent
    # Every data location, once. RIKO_DATA_DIR: the backend's working directory (run_server sets it), where earlier
    # releases kept the to-do list.
    paths = DataPaths.build(config_path, raw, working=os.getenv('RIKO_DATA_DIR'))
    # Every registered section's check (configuration/schema.py): voice, speech and sovits_ping_config become AppConfig's
    # typed sections, checked now instead of at every turn or frame; animation, wake_feedback, emotion.probe, initiative,
    # memory and tools are checked here too.
    errors = [] if recover is True or recover == 'setup' and (raw.get('desktop') or {}).get('setup_on_startup_error') else None
    checked = schema.checked(raw, errors)
    preset = raw.get("presets", {}).get("default", {})
    params = preset.get("model_params", {})
    runtime_raw = raw.get("runtime", {})
    library = runtime_raw.get('native_library')
    if isinstance(library, str) and library.startswith('bundled:'):
        from .native_backends import bundled_library  # electron/native_backends.json: shipped backends, library names
        library = bundled_library(library.split(':', 1)[1], CodePaths.current().bundle)
    runtime = RuntimeConfig(
        provider=runtime_raw.get("provider", "openai"),
        model=runtime_raw.get("model", raw.get("model", "")),
        base_url=runtime_raw.get("base_url", raw.get("base_url", "http://127.0.0.1:8080"
            if str(runtime_raw.get("provider", "")).lower().replace("-", "_") == "llama_server" else "http://localhost:1234/v1")),
        api_key=runtime_raw.get("api_key", raw.get("api_key", os.getenv("RIKO_API_KEY", "local"))),
        model_path=resolve(root, runtime_raw.get("model_path")),
        native_library=resolve(root, library),
        tokenizer_model=runtime_raw.get("tokenizer_model", raw.get("tokenizer_model")),
        n_ctx=int(runtime_raw.get("n_ctx", params.get("context_window_token_limit", 8192))),
        n_gpu_layers=int(runtime_raw.get("n_gpu_layers", -1)),
        n_batch=int(runtime_raw.get("n_batch", 512)),
        temperature=float(runtime_raw.get("temperature", params.get("temperature", 0.7))),
        max_output_tokens=int(runtime_raw.get("max_output_tokens", params.get("max_output_tokens", 1024))),
        seed=int(runtime_raw.get("seed", -1)),
        api_mode=str(runtime_raw.get("api_mode", "auto")),
        reuse_response_ids=bool(runtime_raw.get("reuse_response_ids", True)),
        request_timeout_seconds=float(runtime_raw.get('request_timeout_seconds', 120)),
    )
    if not math.isfinite(runtime.request_timeout_seconds) or runtime.request_timeout_seconds <= 0:
        raise ValueError('runtime.request_timeout_seconds must be positive and finite')
    tools_raw = raw.get("tools", {})
    memory_raw = raw.get("memory", {})
    emotion_raw = raw.get("emotion", {})
    initiative_raw = checked['initiative']  # runtime/initiative.py: its defaults and the YAML's, with the budgets checked
    runtime.initiative_n_ctx = initiative_raw['context_window_tokens']
    runtime.initiative_max_output_tokens = initiative_raw['max_output_tokens']
    # Live initiative preferences survive restart and override YAML defaults.
    import json
    try:
        persisted = json.loads(paths.initiative_settings.read_text(encoding='utf-8'))
        persisted_context = persisted.get('context_window_tokens', initiative_raw['context_window_tokens'])
        persisted_output = persisted.get('max_output_tokens', initiative_raw['max_output_tokens'])
        validate_budget(persisted_context, persisted_output, 'initiative')
        runtime.initiative_n_ctx = persisted_context
        runtime.initiative_max_output_tokens = persisted_output
    except (OSError, ValueError, TypeError, AttributeError): pass
    runtime.reflection_n_ctx = memory_raw.get('reflection_context_window_tokens', 4096)
    # The in-process context holds the prompt and the reply, so by default the prompt gets what the reply leaves.
    native = str(runtime.provider).lower().replace('-', '_') == 'llama_cpp'
    live_prompt = max(1, runtime.n_ctx - runtime.max_output_tokens) if native else 8192
    # The legacy preset key sets n_ctx; as a prompt budget it would leave no room for the reply.
    legacy_prompt = params.get('context_window_token_limit', live_prompt)
    config = AppConfig(
        root=root,
        paths=paths,
        load_errors=tuple(errors or ()),
        runtime=runtime,
        tools=ToolConfig(
            mcp_config=resolve(root, tools_raw.get("mcp_config")),
            max_iterations=int(tools_raw.get("max_iterations", 8)),
            timeout_seconds=float(tools_raw.get("timeout_seconds", 30)),
            require_approval=bool(tools_raw.get("require_approval", False)),
            best_fit_inputs=tools_raw.get('best_fit_inputs', True),
            best_fit_timeout_seconds=float(tools_raw.get('best_fit_timeout_seconds', .4)),
            best_fit_min_confidence=float(tools_raw.get('best_fit_min_confidence', .85)),
            disabled=tuple(tools_raw.get('disabled') or ()),
        ),
        memory=MemoryConfig(
            history_file=paths.chat_history,
            context_window_tokens=int(memory_raw.get("context_window_tokens", min(legacy_prompt, live_prompt) if native else legacy_prompt)),
            store_file=paths.memory_store, index_file=paths.memory_index,
            embedding_model=str(memory_raw.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            embedding_dimension=int(memory_raw.get("embedding_dimension", 384)),
            max_results=int(memory_raw.get("max_results", 8)),
            token_budget=int(memory_raw.get("token_budget", 1200)),
            reflection_enabled=bool(memory_raw.get("reflection_enabled", True)),
            reflection_min_importance=float(memory_raw.get("reflection_min_importance", 0.5)),
            reflection_max_output_tokens=memory_raw.get('reflection_max_output_tokens', 1024),
            reflection_context_window_tokens=memory_raw.get('reflection_context_window_tokens', 4096),
            embeddings_enabled=bool(memory_raw.get("embeddings_enabled", True)),
            system1_enabled=bool(memory_raw.get("system1_enabled", True)),
            system1_model_id=str(memory_raw.get("system1_model_id", JULIA_MODEL_ID)),
            system1_revision=memory_raw.get("system1_revision", JULIA_REVISION),
            system1_cache_dir=resolve(root, memory_raw.get("system1_cache_dir")),
            system1_max_length=int(memory_raw.get("system1_max_length", 8192)),
            device=str(memory_raw.get("device", "cpu")),
            minimum_importance=float(memory_raw.get("minimum_importance", 0.35)),
            default_memories=list(memory_raw.get("default_memories", preset.get("memories", []))),
        ),
        emotion=EmotionConfig(
            enabled=bool(emotion_raw.get("enabled", False)),
            model_path=resolve(root, emotion_raw.get("model_path")),
            model_id=str(emotion_raw.get("model_id", JULIA_MODEL_ID)),
            revision=emotion_raw.get("revision", JULIA_REVISION),
            cache_dir=resolve(root, emotion_raw.get("cache_dir")),
            device=str(emotion_raw.get("device", "cpu")),
            strict_encoding=bool(emotion_raw.get("strict_encoding", True)),
            max_length=int(emotion_raw.get("max_length", 8192)),
            context_tokens=int(emotion_raw.get("context_tokens", 1024)),
            update_interval_tokens=int(emotion_raw.get("update_interval_tokens", 16)),
            temperature=float(emotion_raw.get("temperature", 0.2)),
            fallback=bool(emotion_raw.get("fallback", True)),
            probe=dict(emotion_raw.get('probe', {})),
        ),
        character_name=preset.get("name", raw.get("character_name", "Riko")),
        system_prompt=preset.get("system_prompt", raw.get("system_prompt", "You are a helpful local assistant.")),
        voice=checked['voice'], speech=checked['speech'], sovits=checked['sovits_ping_config'],
        unknown_settings=schema.unknown_settings(raw),
        raw=raw,
        avatar=dict(raw.get("avatar", {})),
    )
    # Checks across sections on the assembled config: the provider's own (inference/settings.py), which also sizes the
    # in-process KV pool from the background budgets, and the emotion probe's needs (emotion/probe.py).
    schema.configure(config, raw)
    return config
