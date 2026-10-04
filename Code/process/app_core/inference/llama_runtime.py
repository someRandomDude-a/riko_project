"""Validated native settings and reproducible GGUF resolution (no model imports)."""
import math
from pathlib import Path, PurePosixPath
import re

KV_TYPES = {'f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl'}
EXTRA_SETTINGS = ('hf_repo_id', 'hf_filename', 'hf_revision', 'hf_local_files_only',
    'n_ubatch', 'n_threads', 'n_threads_batch', 'flash_attn', 'type_k', 'type_v',
    'offload_kqv', 'use_mmap', 'use_mlock', 'main_gpu', 'split_mode', 'tensor_split',
    'chat_format', 'cache_size_mb', 'verbose', 'parallel_slots',
    'startup_timeout_seconds', 'warmup', 'kv_unified', 'kv_pool_auto', 'kv_pool_tokens', 'pause_background_on_live')


def configure_runtime(config, raw):
    for key in EXTRA_SETTINGS:
        if key in raw: setattr(config, key, raw[key])
    if config.provider.lower().replace('-', '_') == 'llama_cpp':
        for key in ('n_ctx', 'n_batch', 'n_gpu_layers'):
            if key in raw and type(raw[key]) is not int: raise ValueError(f'runtime.{key} must be an integer')
        if 'n_ubatch' not in raw: config.n_ubatch = min(512, config.n_batch)
        validate_runtime(config)


def validate_runtime(config):
    if type(config.parallel_slots) is not int or not 2 <= config.parallel_slots <= 4:
        raise ValueError('runtime.parallel_slots must be an integer from 2 to 4')
    if type(config.warmup) is not bool: raise ValueError('runtime.warmup must be boolean')
    if type(config.pause_background_on_live) is not bool: raise ValueError('runtime.pause_background_on_live must be boolean')
    if type(config.kv_unified) is not bool: raise ValueError('runtime.kv_unified must be boolean')
    if type(config.kv_pool_auto) is not bool: raise ValueError('runtime.kv_pool_auto must be boolean')
    if config.kv_pool_tokens is not None and (type(config.kv_pool_tokens) is not int or not 1 <= config.kv_pool_tokens <= 4194304):
        raise ValueError('runtime.kv_pool_tokens must be null or an integer from 1 to 4194304')
    if type(config.startup_timeout_seconds) not in (int, float) or not math.isfinite(config.startup_timeout_seconds) or config.startup_timeout_seconds <= 0:
        raise ValueError('runtime.startup_timeout_seconds must be positive and finite')
    for key in ('n_ctx', 'n_batch', 'n_ubatch', 'main_gpu', 'cache_size_mb'):
        value = getattr(config, key)
        minimum = 1 if key in {'n_batch', 'n_ubatch'} else 0
        if type(value) is not int or value < minimum: raise ValueError(f'runtime.{key} must be an integer >= {minimum}')
    if type(config.n_gpu_layers) is not int or config.n_gpu_layers < -1:
        raise ValueError('runtime.n_gpu_layers must be -1 or a nonnegative integer')
    if config.n_ubatch > config.n_batch: raise ValueError('runtime.n_ubatch must not exceed n_batch')
    for key in ('n_threads', 'n_threads_batch'):
        value = getattr(config, key)
        if value is not None and (type(value) is not int or value < 1): raise ValueError(f'runtime.{key} must be null or a positive integer')
    for key in ('flash_attn', 'offload_kqv', 'use_mmap', 'use_mlock', 'hf_local_files_only', 'verbose'):
        if type(getattr(config, key)) is not bool: raise ValueError(f'runtime.{key} must be boolean')
    for key in ('type_k', 'type_v'):
        if getattr(config, key) not in KV_TYPES: raise ValueError(f'runtime.{key} must be one of {sorted(KV_TYPES)}')
    if config.type_v not in {'f16', 'f32', 'bf16'} and not config.flash_attn:
        raise ValueError('Quantized runtime.type_v requires flash_attn: true')
    if config.split_mode not in {'none', 'layer', 'row'}: raise ValueError('runtime.split_mode must be none, layer or row')
    if config.tensor_split is not None:
        values = config.tensor_split
        if not isinstance(values, list) or not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
            raise ValueError('runtime.tensor_split must contain nonnegative finite weights with a positive sum')
    for key in ('hf_repo_id', 'hf_filename', 'chat_format'):
        value = getattr(config, key)
        if value is not None and (not isinstance(value, str) or not value.strip()): raise ValueError(f'runtime.{key} must be null or a nonempty string')
    if not isinstance(config.hf_revision, str) or not config.hf_revision.strip(): raise ValueError('runtime.hf_revision must be a nonempty string')
    if not config.model_path and not (config.hf_repo_id and config.hf_filename):
        raise ValueError('llama_cpp requires runtime.model_path or both hf_repo_id and hf_filename')
    if config.hf_filename:
        filename = PurePosixPath(config.hf_filename)
        if filename.is_absolute() or '..' in filename.parts or '\\' in config.hf_filename or any(c in config.hf_filename for c in '*?[]') or filename.suffix.lower() != '.gguf':
            raise ValueError('runtime.hf_filename must be an exact relative GGUF filename, not a glob')
        split = re.fullmatch(r'(.*)-(\d{5})-of-(\d{5})\.gguf', config.hf_filename)
        if split and (int(split[2]) != 1 or not 1 <= int(split[3]) <= 1024):
            raise ValueError('Split GGUF requires its first shard and a valid shard count')


def resolve_model(config):
    if config.model_path: return Path(config.model_path) # Explicit path wins; no network.
    from huggingface_hub import hf_hub_download
    options = dict(repo_id=config.hf_repo_id, revision=config.hf_revision,
                   local_files_only=config.hf_local_files_only)
    filename = config.hf_filename
    # llama.cpp discovers sibling split shards by name; ensure all are in the HF snapshot.
    split = re.fullmatch(r'(.*)-(\d{5})-of-(\d{5})\.gguf', filename)
    if split:
        prefix, part, total = split.groups()
        if int(part) != 1 or not 1 <= int(total) <= 1024:
            raise ValueError('Split GGUF requires the first shard (-00001-of-NNNNN.gguf)')
        # Pin all shard downloads to the same resolved snapshot, even if main moves.
        first = Path(hf_hub_download(filename=filename, **options))
        options['revision'] = first.parent.name if '/' not in filename else first.parents[len(PurePosixPath(filename).parts)-1].name
        for index in range(2, int(total)+1):
            hf_hub_download(filename=f'{prefix}-{index:05d}-of-{total}.gguf', **options)
        return first
    return Path(hf_hub_download(filename=filename, **options))
