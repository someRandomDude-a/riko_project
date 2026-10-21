"""Validated llama.cpp settings and reproducible GGUF resolution (no model imports)."""
import math
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit

from ..kernel.schema import Rule

KV_TYPES = ('f16', 'f32', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl')  # in the order Settings offers them
FULL_PRECISION = ('f16', 'f32', 'bf16')
EXTRA_SETTINGS = ('hf_repo_id', 'hf_filename', 'hf_revision', 'hf_local_files_only',
    'n_ubatch', 'n_threads', 'n_threads_batch', 'flash_attn', 'type_k', 'type_v',
    'offload_kqv', 'use_mmap', 'use_mlock', 'main_gpu', 'split_mode', 'tensor_split',
    'chat_format', 'cache_size_mb', 'verbose', 'parallel_slots',
    'startup_timeout_seconds', 'warmup', 'kv_unified', 'kv_pool_auto', 'kv_pool_tokens', 'pause_background_on_live')
FLASH_ATTENTION = ('auto', 'on', 'off')
NATIVE = (('runtime.provider', ('llama_cpp',)),)
# The checks across llama.cpp settings that validate_runtime enforces and Settings returns to the UI (/api/settings 'rules').
RULES = (Rule('runtime.n_ubatch', 'runtime.n_ubatch must not exceed n_batch', when=NATIVE, at_most='runtime.n_batch'),
    Rule('runtime.type_v', 'Quantized runtime.type_v requires flash_attn auto or on', when=(*NATIVE, ('runtime.type_v', tuple(t for t in KV_TYPES if t not in FULL_PRECISION))),
        one_of=(('runtime.flash_attn', ('auto', 'on')),)),
    Rule('runtime.hf_repo_id', 'llama_cpp requires runtime.model_path or both hf_repo_id and hf_filename', when=NATIVE,
        any_set=(('runtime.model_path',), ('runtime.hf_repo_id', 'runtime.hf_filename'))))


def flash_attention(value):
    """runtime.flash_attn as llama.cpp's --flash-attn value. A YAML boolean keeps its old meaning; PyYAML also
    reads the bare on/off that ruamel writes back as booleans."""
    if type(value) is bool: return 'on' if value else 'off'
    if value in FLASH_ATTENTION: return value
    raise ValueError('runtime.flash_attn must be auto, on or off')


def check_native_build(config):
    """Settings the pinned llama.cpp cannot load. Checked when the provider starts, not when the YAML loads,
    so desktop.setup_on_startup_error keeps Settings open to repair them."""
    if config.split_mode == 'row':
        raise ValueError("runtime.split_mode 'row' is not available: llama.cpp removed it from CUDA and ROCm, and Metal "
            "and Vulkan never had it. Choose layer (or none for one GPU) in Settings.")


def configure_runtime(config, raw):
    for key in EXTRA_SETTINGS:
        if key in raw: setattr(config, key, raw[key])
    provider = config.provider.lower().replace('-', '_')
    if provider == 'llama_cpp':
        for key in ('n_ctx', 'n_batch', 'n_gpu_layers'):
            if key in raw and type(raw[key]) is not int: raise ValueError(f'runtime.{key} must be an integer')
        if 'n_ubatch' not in raw: config.n_ubatch = min(512, config.n_batch)
        validate_runtime(config)
    elif provider == 'llama_server':
        if 'n_ctx' in raw and type(raw['n_ctx']) is not int: raise ValueError('runtime.n_ctx must be an integer')
        if config.base_url is None: config.base_url = 'http://127.0.0.1:8080'
        server_address(config.base_url)
        if config.api_key is not None and not isinstance(config.api_key, str):
            raise ValueError('runtime.api_key must be text; put it in quotes in character_config.yaml')
        validate_slots(config)


def server_address(url):
    """The llama-server root for runtime.base_url; it serves /tokenize, /slots and /health beside /v1."""
    problem = ValueError('runtime.base_url must be the llama-server address, such as http://127.0.0.1:8080')
    if not isinstance(url, str): raise problem
    root = url.strip().rstrip('/')
    root = root[:-3] if root.endswith('/v1') else root
    parts = urlsplit(root)
    try: parts.port
    except ValueError: raise problem from None
    if parts.scheme not in {'http', 'https'} or not parts.hostname or parts.username or parts.password or parts.path or parts.query or parts.fragment:
        raise problem
    return root


def validate_slots(config):
    """Settings shared by every llama.cpp transport: Riko schedules the slots and packs each request."""
    if type(config.parallel_slots) is not int or not 2 <= config.parallel_slots <= 4:
        raise ValueError('runtime.parallel_slots must be an integer from 2 to 4')
    if type(config.warmup) is not bool: raise ValueError('runtime.warmup must be boolean')
    if type(config.pause_background_on_live) is not bool: raise ValueError('runtime.pause_background_on_live must be boolean')
    if type(config.startup_timeout_seconds) not in (int, float) or not math.isfinite(config.startup_timeout_seconds) or config.startup_timeout_seconds <= 0:
        raise ValueError('runtime.startup_timeout_seconds must be positive and finite')
    if type(config.n_ctx) is not int or config.n_ctx < 0: raise ValueError('runtime.n_ctx must be an integer >= 0')


def validate_runtime(config):
    validate_slots(config)
    if type(config.kv_unified) is not bool: raise ValueError('runtime.kv_unified must be boolean')
    if type(config.kv_pool_auto) is not bool: raise ValueError('runtime.kv_pool_auto must be boolean')
    if config.kv_pool_tokens is not None and (type(config.kv_pool_tokens) is not int or not 1 <= config.kv_pool_tokens <= 4194304):
        raise ValueError('runtime.kv_pool_tokens must be null or an integer from 1 to 4194304')
    for key in ('n_ctx', 'n_batch', 'n_ubatch', 'main_gpu', 'cache_size_mb'):
        value = getattr(config, key)
        minimum = 1 if key in {'n_batch', 'n_ubatch'} else 0
        if type(value) is not int or value < minimum: raise ValueError(f'runtime.{key} must be an integer >= {minimum}')
    if type(config.n_gpu_layers) is not int or config.n_gpu_layers < -2:
        raise ValueError('runtime.n_gpu_layers must be -1 (fit to free GPU memory), -2 (all layers) or a layer count >= 0')
    for key in ('n_threads', 'n_threads_batch'):
        value = getattr(config, key)
        if value is not None and (type(value) is not int or value < 1): raise ValueError(f'runtime.{key} must be null or a positive integer')
    for key in ('offload_kqv', 'use_mmap', 'use_mlock', 'hf_local_files_only', 'verbose'):
        if type(getattr(config, key)) is not bool: raise ValueError(f'runtime.{key} must be boolean')
    config.flash_attn = flash_attention(config.flash_attn)
    for key in ('type_k', 'type_v'):
        if getattr(config, key) not in KV_TYPES: raise ValueError(f'runtime.{key} must be one of {sorted(KV_TYPES)}')
    # 'row' still loads from YAML so Settings can open and fix it; check_native_build rejects it at startup.
    if config.split_mode not in {'none', 'layer', 'row'}: raise ValueError('runtime.split_mode must be layer or none')
    if config.tensor_split is not None:
        values = config.tensor_split
        if not isinstance(values, list) or not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
            raise ValueError('runtime.tensor_split must contain nonnegative finite weights with a positive sum')
    for key in ('hf_repo_id', 'hf_filename', 'chat_format'):
        value = getattr(config, key)
        if value is not None and (not isinstance(value, str) or not value.strip()): raise ValueError(f'runtime.{key} must be null or a nonempty string')
    if not isinstance(config.hf_revision, str) or not config.hf_revision.strip(): raise ValueError('runtime.hf_revision must be a nonempty string')
    settings = {'runtime.provider': 'llama_cpp', **{f'runtime.{key}': getattr(config, key) for key in ('n_ubatch', 'n_batch', 'type_v', 'flash_attn', 'model_path', 'hf_repo_id', 'hf_filename')}}
    for rule in RULES:
        if rule.broken(settings): raise ValueError(rule.message)
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
