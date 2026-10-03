"""Settings-aware estimates with provenance/unknowns, not OOM guarantees."""
from functools import lru_cache
import json
from pathlib import Path
import struct

from .gpu_memory import MIB
from ..inference.llama_server import context_capacity

KV_BYTES = {'f32': 4, 'f16': 2, 'bf16': 2, 'q8_0': 34 / 32, 'q4_0': 18 / 32,
            'q4_1': 20 / 32, 'q5_0': 22 / 32, 'q5_1': 24 / 32, 'iq4_nl': 18 / 32}


def read_gguf(path):
    """Read only metadata, never tensors. Bound all untrusted lengths."""
    scalar = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
    with Path(path).open('rb') as file:
        def unpack(fmt):
            data = file.read(struct.calcsize(fmt))
            if len(data) != struct.calcsize(fmt): raise ValueError('Truncated GGUF metadata')
            return struct.unpack(fmt, data)[0]
        def string():
            length = unpack('<Q')
            if length > 4 * MIB: raise ValueError('Oversized GGUF metadata string')
            data = file.read(length)
            if len(data) != length: raise ValueError('Truncated GGUF string')
            return data.decode('utf-8')
        def value(kind, keep=True, depth=0):
            if depth > 2: raise ValueError('Nested GGUF metadata exceeds limit')
            if kind in scalar: return unpack(scalar[kind])
            if kind == 8: return string()
            if kind == 9:
                subtype, count = unpack('<I'), unpack('<Q')
                if count > 2000000: raise ValueError('Oversized GGUF array')
                if subtype in scalar and not keep:
                    file.seek(struct.calcsize(scalar[subtype]) * count, 1)
                    return None
                result = []
                for _ in range(count):
                    item = value(subtype, False, depth + 1)
                    if keep and count <= 4096: result.append(item)
                return result if keep and count <= 4096 else None
            raise ValueError('Unknown GGUF metadata type')
        if file.read(4) != b'GGUF' or unpack('<I') not in (2, 3): raise ValueError('Not GGUF v2/v3')
        unpack('<Q')
        count = unpack('<Q')
        if count > 10000: raise ValueError('Oversized GGUF metadata')
        result = {}
        for _ in range(count):
            key, kind = string(), unpack('<I')
            keep = not key.startswith(('tokenizer.', 'general.description', 'general.tags'))
            item = value(kind, keep)
            if keep: result[key] = item
        return result


@lru_cache(maxsize=16)
def local_metadata(path, size, modified):
    raw = read_gguf(path)
    architecture = raw.get('general.architecture', '')
    def get(key, default=None): return raw.get(architecture + '.' + key, default)
    return {'weight_bytes': size, 'layers': get('block_count'), 'embedding': get('embedding_length'),
        'kv_heads': get('attention.head_count_kv'), 'heads': get('attention.head_count'),
        'key_length': get('attention.key_length'), 'value_length': get('attention.value_length'),
        'context_length': get('context_length'), 'architecture': architecture,
        'full_attention_interval': get('full_attention_interval'),
        'ssm_state_size': get('ssm.state_size'), 'source': 'local GGUF metadata + file size'}


@lru_cache(maxsize=16)
def remote_metadata(repo, revision, filename):
    # Repository metadata/config only; never download model weights for an estimate.
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True, timeout=5)
    siblings = info.siblings or []
    exact = next((s for s in siblings if s.rfilename == filename), None)
    if exact is None: raise ValueError('Selected GGUF absent from repository')
    size = exact.size
    import re
    split = re.fullmatch(r'(.*)-00001-of-(\d{5})\.gguf', filename)
    if split:
        parts = [next((s for s in siblings if s.rfilename == f'{split[1]}-{i:05d}-of-{int(split[2]):05d}.gguf'), None) for i in range(1, int(split[2]) + 1)]
        size = sum(s.size for s in parts) if all(s and s.size for s in parts) else None
    result = {'weight_bytes': size, 'source': 'Hugging Face file sizes; architecture unavailable'}
    import requests
    url = f'https://huggingface.co/{repo}/resolve/{revision}/config.json'
    with requests.get(url, timeout=5, stream=True) as response:
        if response.status_code != 200: return result
        data = bytearray()
        for chunk in response.iter_content(16384):
            data.extend(chunk)
            if len(data) > MIB: return result
        config = json.loads(data)
    text = config.get('text_config', config)
    result.update(layers=text.get('num_hidden_layers'), embedding=text.get('hidden_size'),
        heads=text.get('num_attention_heads'), kv_heads=text.get('num_key_value_heads'),
        key_length=text.get('head_dim'), value_length=text.get('head_dim'),
        context_length=text.get('max_position_embeddings'), architecture=text.get('model_type', ''),
        full_attention_interval=text.get('full_attention_interval'), source='Hugging Face file sizes + transformer config (approximate)')
    return result


def model_metadata(runtime):
    if runtime.model_path:
        path = Path(runtime.model_path)
        stat = path.stat()
        data = dict(local_metadata(str(path), stat.st_size, stat.st_mtime_ns))
        import re
        split = re.fullmatch(r'(.*)-00001-of-(\d{5})\.gguf', path.name)
        if split:
            data['weight_bytes'] = sum(path.with_name(f'{split[1]}-{i:05d}-of-{int(split[2]):05d}.gguf').stat().st_size for i in range(1, int(split[2]) + 1))
        return data
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(runtime.hf_repo_id, runtime.hf_filename, revision=runtime.hf_revision)
        if isinstance(cached, str):
            from copy import copy
            local = copy(runtime)
            local.model_path = Path(cached)
            return model_metadata(local)
    except (ImportError, ValueError): pass
    if runtime.hf_local_files_only: raise ValueError('Model metadata unavailable in local cache (offline mode)')
    return dict(remote_metadata(runtime.hf_repo_id, runtime.hf_revision, runtime.hf_filename))


def estimate(config, telemetry, metadata=None):
    runtime = config.runtime
    components, warnings = [], []
    def add(key, label, low, high, basis):
        components.append({'id': key, 'label': label, 'low_mib': low, 'high_mib': high, 'basis': basis})
    managed = runtime.provider.lower().replace('-', '_') == 'llama_cpp'
    meta = metadata or {}
    if managed and metadata is None:
        try: meta = model_metadata(runtime)
        except Exception as exc: warnings.append('LLM metadata unavailable: ' + str(exc))
    pool = context_capacity(runtime)
    kv_per_token = None
    if managed and runtime.n_gpu_layers != 0:
        layers, embedding, heads, kv_heads = (meta.get(k) for k in ('layers', 'embedding', 'heads', 'kv_heads'))
        fraction = min(1, runtime.n_gpu_layers / max(1, layers)) if layers and runtime.n_gpu_layers >= 0 else 1
        weight = meta.get('weight_bytes')
        add('llm_weights', 'Shared LLM weights (one copy)', weight / MIB * fraction if weight else None,
            weight / MIB * fraction * 1.1 if weight else None, meta.get('source', 'Unknown model metadata') + '; partial offload approximated by layer fraction')
        if all(type(v) in (int, float) and v > 0 for v in (layers, embedding, heads, kv_heads)):
            key_dim = meta.get('key_length') or embedding / heads
            value_dim = meta.get('value_length') or embedding / heads
            attention_layers = layers
            interval = meta.get('full_attention_interval')
            if interval:
                attention_layers = max(1, layers // interval)
                warnings.append('Hybrid recurrent model: attention KV estimated from full-attention interval; recurrent state uses an additional reserve, not dense KV for every layer.')
                add('recurrent', 'Hybrid recurrent state reserve', 32 * runtime.parallel_slots, 256 * runtime.parallel_slots, 'Heuristic per-slot state/checkpoint reserve; backend layout/build can differ')
            elif any(name in str(meta.get('architecture', '')).lower() for name in ('qwen35', 'qwen3_5', 'mamba', 'rwkv', 'jamba')):
                warnings.append('Hybrid architecture without full-attention metadata: dense KV upper bound; no reliable maximum-context suggestion.')
            if runtime.offload_kqv:
                kv_per_token = attention_layers * kv_heads * (key_dim * KV_BYTES[runtime.type_k] + value_dim * KV_BYTES[runtime.type_v]) * fraction / MIB
                add('kv', 'KV pool — all slots occupied', pool * kv_per_token, pool * kv_per_token * 1.15, 'Attention dimensions × pooled tokens × selected K/V tensor formats; 15% padding allowance')
            else: add('kv', 'KV cache (CPU)', 0, 0, 'GPU KV offload disabled')
            batch = (max(128, runtime.n_ubatch) * embedding * max(4, heads) * 4 + runtime.n_batch * embedding * 8) / MIB
            add('llm_compute', 'LLM compute / CUDA graphs', max(128, batch), max(384, batch * (1.25 if runtime.flash_attn else 2)), 'Heuristic workspace affected by batch, microbatch and flash attention')
        else:
            add('kv', 'KV pool — all slots occupied', None, None, 'Attention metadata unavailable; cannot compute KV bytes per token')
            add('llm_compute', 'LLM compute workspace', 128, 768, 'Heuristic; missing architecture details')
    else:
        add('llm_weights', 'LLM', 0, 0, 'CPU-only owned model' if managed else 'External provider excluded (any local GPU use appears in Other)')
    voice = config.raw.get('voice', {})
    device = voice.get('asr_device', 'cuda')
    model = str(voice.get('asr_model', 'distil-small.en')).lower()
    base = next((size for name, size in [('large', 3100), ('medium', 1600), ('small', 600), ('base', 200), ('tiny', 100)] if name in model), None)
    if device == 'cpu': add('asr', 'Whisper / faster-whisper ASR', 0, 0, 'CPU configured')
    elif base:
        precision = voice.get('asr_compute_type', 'int8_float16')
        factor = .65 if precision.startswith('int8') else 2 if precision == 'float32' else 1
        add('asr', 'Whisper / faster-whisper ASR', base * factor + 128, base * factor * 1.4 + 384,
            f'{model}, {precision}, {device}; heuristic resident weights + decoder workspace, included even if microphone is idle')
    else: add('asr', 'Whisper / faster-whisper ASR', None, None, f'Unknown custom ASR model {model}; no invented size')
    add('julia', 'Julia emotion / memory classifier / animation policy', 0, 0, 'Current implementations force CPU; no separate animation model')
    add('embedding', 'Memory embeddings, wake detector and VAD', 0, 0, 'CPU sentence embeddings, ONNX CPU wake detector and CPU Silero VAD; no separate flow model owned by this application')
    add('electron', 'Electron / VRM / whiteboard GPU surfaces', 96, 384, 'Heuristic graphics textures/framebuffers; driver and assets affect this')
    add('cuda', 'Shared Python CUDA/runtime overhead', 128 if device != 'cpu' else 0, 384 if device != 'cpu' else 0, 'One reserve, not duplicated per ASR subtask')
    add('tts', 'TTS / external providers', 0, 0, 'External processes excluded from owned estimates; their real GPU use remains in Other')
    unknown = [c['label'] for c in components if c['high_mib'] is None]
    low = sum(c['low_mib'] or 0 for c in components)
    high = sum(c['high_mib'] or 0 for c in components)
    gpu = next((g for g in telemetry.get('gpus', []) if g['index'] == runtime.main_gpu), None)
    suggestion = None
    if gpu and not unknown and kv_per_token and not any('no reliable' in w for w in warnings):
        fixed = high - next(c['high_mib'] for c in components if c['id'] == 'kv')
        reserved = gpu['other_mib'] if gpu.get('other_mib') is not None else gpu['used_mib']
        available = gpu['total_mib'] - reserved - fixed - max(256, gpu['total_mib'] * .05)
        maximum_pool = max(0, int(available / (kv_per_token * 1.15)))
        if runtime.kv_unified:
            suggested_live = max(0, maximum_pool - (pool - runtime.n_ctx))
        else:
            suggested_live = maximum_pool // runtime.parallel_slots
            if suggested_live < max(getattr(runtime, 'initiative_n_ctx', 4096), getattr(runtime, 'reflection_n_ctx', 4096)): suggested_live = 0
        model_limit = meta.get('context_length')
        if model_limit: suggested_live = min(suggested_live, model_limit)
        suggested_live = min(suggested_live, 1048576)
        if suggested_live < runtime.max_output_tokens + 256: suggested_live = 0
        suggestion = {'live_context_tokens': suggested_live // 256 * 256, 'maximum_pool_tokens': maximum_pool,
            'basis': 'Upper estimate + 5%/256 MiB safety reserve; background budgets fixed; advisory only'
                + ('; attribution unavailable: conservatively reserves ALL currently used VRAM, which may double-count already-running owned components' if gpu.get('other_mib') is None else '')}
    if runtime.tensor_split: warnings.append('Multi-GPU split estimates are aggregate, not per-device; recommendation suppressed.'); suggestion = None
    projection = None
    if gpu and gpu.get('other_mib') is not None and not unknown and not runtime.tensor_split:
        projection = {'low_mib': low + gpu['other_mib'], 'high_mib': high + gpu['other_mib'], 'total_mib': gpu['total_mib'],
            'fits_upper_estimate': high + gpu['other_mib'] <= gpu['total_mib']}
    return {'components': components, 'low_mib': low, 'high_mib': high, 'complete': not unknown, 'unknown_components': unknown,
        'confidence': 'low',
        'confidence_basis': 'Model/KV metadata is used where available; compute, recurrent state, ASR and graphics reserves remain heuristic, not a native backend dry-run allocation report.',
        'warnings': warnings, 'suggestion': suggestion, 'kv': {'unified': runtime.kv_unified, 'pool_tokens': pool,
        'live_context_tokens': runtime.n_ctx, 'initiative_context_tokens': getattr(runtime, 'initiative_n_ctx', 4096),
        'reflection_context_tokens': getattr(runtime, 'reflection_n_ctx', 4096), 'slots': runtime.parallel_slots},
        'managed': managed,
        'projection': projection,
        'note': 'Estimated peak residency, not measured allocations. Ranges and unknowns are intentional. Recommendations never modify settings.'}
