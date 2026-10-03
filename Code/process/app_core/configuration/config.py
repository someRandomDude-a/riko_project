"""Validated configuration with stable paths and environment overrides."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # Keep core imports usable for tooling/tests before dependencies are installed.
    yaml = None


@dataclass
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
    flash_attn: bool = False
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
    server_path: str = 'llama-server'
    startup_timeout_seconds: float = 600.0
    warmup: bool = True
    pause_background_on_live: bool = True


@dataclass
class ToolConfig:
    mcp_config: Path | None = None
    max_iterations: int = 8
    timeout_seconds: float = 30.0
    require_approval: bool = False
    best_fit_inputs: bool = True
    best_fit_timeout_seconds: float = .4
    best_fit_min_confidence: float = .85


@dataclass
class MemoryConfig:
    history_file: Path = Path("persistent_memories/chat_history.json")
    context_window_tokens: int = 8192
    store_file: Path = Path("persistent_memories/memory_store.json")
    index_file: Path = Path("persistent_memories/faiss_index.index")
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
    system1_model_id: str = "SupersonicLabs/Julia-1"
    system1_cache_dir: Path | None = None
    system1_max_length: int = 8192
    minimum_importance: float = 0.35
    default_memories: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EmotionConfig:
    enabled: bool = False
    model_path: Path | None = None
    model_id: str = "SupersonicLabs/Julia-1"
    cache_dir: Path | None = None
    device: str = "cpu"
    strict_encoding: bool = True
    max_length: int = 8192
    context_tokens: int = 1024
    update_interval_tokens: int = 16
    temperature: float = 0.2
    fallback: bool = True


@dataclass
class AppConfig:
    root: Path
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    character_name: str = "Riko"
    system_prompt: str = "You are a helpful local assistant."
    raw: dict[str, Any] = field(default_factory=dict)
    avatar: dict[str, Any] = field(default_factory=dict)


def _path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or os.getenv("RIKO_CONFIG", "character_config.yaml")).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    raw: dict[str, Any] = {}
    if config_path.exists():
        if yaml is None:
            raise RuntimeError("PyYAML is required to load character_config.yaml")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = config_path.parent
    from ..audio.speech_chunks import validate_settings
    validate_settings(raw.get('speech', {}))
    from ..audio.wake_feedback import validate_settings as validate_wake_feedback
    validate_wake_feedback(raw.get('wake_feedback', {}))
    from ..animation.library import validate_settings as validate_animation
    validate_animation(raw.get('animation', {}))
    preset = raw.get("presets", {}).get("default", {})
    params = preset.get("model_params", {})
    runtime_raw = raw.get("runtime", {})
    runtime = RuntimeConfig(
        provider=runtime_raw.get("provider", "openai"),
        model=runtime_raw.get("model", raw.get("model", "")),
        base_url=runtime_raw.get("base_url", raw.get("base_url", "http://localhost:1234/v1")),
        api_key=runtime_raw.get("api_key", raw.get("api_key", os.getenv("RIKO_API_KEY", "local"))),
        model_path=_path(root, runtime_raw.get("model_path")),
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
    from ..inference.llama_runtime import configure_runtime
    configure_runtime(runtime, runtime_raw)
    if '/' in runtime.server_path or '\\' in runtime.server_path:
        runtime.server_path = str(_path(root, runtime.server_path))
    tools_raw = raw.get("tools", {})
    if type(tools_raw.get('best_fit_inputs', True)) is not bool: raise ValueError('tools.best_fit_inputs must be boolean')
    for key, default, low, high in [('best_fit_timeout_seconds', .4, .05, 2), ('best_fit_min_confidence', .85, .5, 1)]:
        value = tools_raw.get(key, default)
        if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high: raise ValueError(f'tools.{key} must be between {low} and {high}')
    memory_raw = raw.get("memory", {})
    emotion_raw = raw.get("emotion", {})
    from ..inference.background_budget import validate_budget
    from ..runtime.initiative import DEFAULTS as INITIATIVE_DEFAULTS
    initiative_raw = {**INITIATIVE_DEFAULTS, **raw.get('initiative', {})}
    validate_budget(initiative_raw['context_window_tokens'], initiative_raw['max_output_tokens'], 'initiative')
    validate_budget(memory_raw.get('reflection_context_window_tokens', 4096), memory_raw.get('reflection_max_output_tokens', 1024), 'reflection')
    runtime.initiative_n_ctx = initiative_raw['context_window_tokens']
    runtime.initiative_max_output_tokens = initiative_raw['max_output_tokens']
    # Live initiative preferences survive restart and override YAML defaults.
    import json
    try:
        persisted = json.loads((root / 'persistent_memories' / 'initiative_settings.json').read_text(encoding='utf-8'))
        persisted_context = persisted.get('context_window_tokens', initiative_raw['context_window_tokens'])
        persisted_output = persisted.get('max_output_tokens', initiative_raw['max_output_tokens'])
        validate_budget(persisted_context, persisted_output, 'initiative')
        runtime.initiative_n_ctx = persisted_context
        runtime.initiative_max_output_tokens = persisted_output
    except (OSError, ValueError, TypeError, AttributeError): pass
    runtime.reflection_n_ctx = memory_raw.get('reflection_context_window_tokens', 4096)
    from ..inference.kv_budget import pool_capacity
    if runtime.kv_pool_auto: runtime.kv_pool_tokens = pool_capacity(runtime)
    return AppConfig(
        root=root,
        runtime=runtime,
        tools=ToolConfig(
            mcp_config=_path(root, tools_raw.get("mcp_config")),
            max_iterations=int(tools_raw.get("max_iterations", 8)),
            timeout_seconds=float(tools_raw.get("timeout_seconds", 30)),
            require_approval=bool(tools_raw.get("require_approval", False)),
            best_fit_inputs=tools_raw.get('best_fit_inputs', True),
            best_fit_timeout_seconds=float(tools_raw.get('best_fit_timeout_seconds', .4)),
            best_fit_min_confidence=float(tools_raw.get('best_fit_min_confidence', .85)),
        ),
        memory=MemoryConfig(
            history_file=_path(root, raw.get("history_file", memory_raw.get("history_file", "persistent_memories/chat_history.json"))) or root / "persistent_memories/chat_history.json",
            context_window_tokens=int(memory_raw.get("context_window_tokens", params.get("context_window_token_limit", 8192))),
            store_file=_path(root, memory_raw.get("store_file", "persistent_memories/memory_store.json")) or root / "persistent_memories/memory_store.json",
            index_file=_path(root, memory_raw.get("index_file", "persistent_memories/faiss_index.index")) or root / "persistent_memories/faiss_index.index",
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
            system1_model_id=str(memory_raw.get("system1_model_id", "SupersonicLabs/Julia-1")),
            system1_cache_dir=_path(root, memory_raw.get("system1_cache_dir")),
            system1_max_length=int(memory_raw.get("system1_max_length", 8192)),
            minimum_importance=float(memory_raw.get("minimum_importance", 0.35)),
            default_memories=list(memory_raw.get("default_memories", preset.get("memories", []))),
        ),
        emotion=EmotionConfig(
            enabled=bool(emotion_raw.get("enabled", False)),
            model_path=_path(root, emotion_raw.get("model_path")),
            model_id=str(emotion_raw.get("model_id", "SupersonicLabs/Julia-1")),
            cache_dir=_path(root, emotion_raw.get("cache_dir")),
            device=str(emotion_raw.get("device", "cpu")),
            strict_encoding=bool(emotion_raw.get("strict_encoding", True)),
            max_length=int(emotion_raw.get("max_length", 8192)),
            context_tokens=int(emotion_raw.get("context_tokens", 1024)),
            update_interval_tokens=int(emotion_raw.get("update_interval_tokens", 16)),
            temperature=float(emotion_raw.get("temperature", 0.2)),
            fallback=bool(emotion_raw.get("fallback", True)),
        ),
        character_name=preset.get("name", raw.get("character_name", "Riko")),
        system_prompt=preset.get("system_prompt", raw.get("system_prompt", "You are a helpful local assistant.")),
        raw=raw,
        avatar=dict(raw.get("avatar", {})),
    )
