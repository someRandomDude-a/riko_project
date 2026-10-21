"""Settings-aware estimates with provenance/unknowns, not OOM guarantees."""
from functools import lru_cache
import json
from pathlib import Path
import sys

from .gpu_memory import MIB
from ..inference.gguf import read_gguf  # defined here before inference/gguf.py: old imports still work
from ..inference.llama_context import context_capacity

KV_BYTES = {'f32': 4, 'f16': 2, 'bf16': 2, 'q8_0': 34 / 32, 'q4_0': 18 / 32,
            'q4_1': 20 / 32, 'q5_0': 22 / 32, 'q5_1': 24 / 32, 'iq4_nl': 18 / 32}


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


def host_platform(): return sys.platform


def torch_target(setting, *, mps=True):
    """(device, exact) a torch model with this device setting loads on: auto is exact only once torch is imported (resolving
    it imports torch); before that, on macOS it can only become MPS or the CPU, never CUDA. mps=False: the model maps MPS to
    the CPU (Julia moves its inputs only for CUDA)."""
    from ..kernel.torch_device import resolve, validate
    device, exact = validate(setting), True
    if device == 'auto':
        if 'torch' in sys.modules:
            try: device = resolve(device)
            except Exception: exact = False
        else: exact = False
        if not exact and host_platform() == 'darwin': device, exact = ('mps', False) if mps else ('cpu', True)
    if device == 'mps' and not mps: device = 'cpu'
    return device.split(':')[0], exact


def weights_mib(model_id, cache_dir=None, path=None, revision=None):
    """A torch model's weights on disk (MiB), from its local folder or the Hugging Face cache (at revision: a pinned commit's
    snapshot has no refs/main), never downloaded; None when neither has them."""
    def size(files): return sum(file.stat().st_size for file in files) / MIB if files else None
    try:
        if path and Path(path).is_dir():
            files = [file for file in Path(path).rglob('*') if file.suffix in ('.safetensors', '.bin') and file.is_file()]
            return size([file for file in files if file.suffix == '.safetensors'] or files)
        from huggingface_hub import try_to_load_from_cache
        def cached(name):
            found = try_to_load_from_cache(model_id, name, cache_dir=str(cache_dir) if cache_dir else None, revision=revision)
            return Path(found) if isinstance(found, str) else None
        for name in ('model.safetensors', 'pytorch_model.bin'):
            if (found := cached(name)): return size([found])
        for index in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
            if (found := cached(index)):
                shards = [cached(name) for name in sorted(set(json.loads(found.read_text(encoding='utf-8')).get('weight_map', {}).values()))]
                return size(shards) if shards and all(shards) else None
    except (ImportError, OSError, ValueError, TypeError): return None
    return None


def torch_models(config):
    """The in-process torch models as (row id, label, [(name, device setting, mps allowed, model id, cache dir, local path,
    revision)])."""
    emotion, memory = config.emotion, config.memory
    julia = [('Julia emotion', emotion.device, False, emotion.model_id, emotion.cache_dir, emotion.model_path, emotion.revision)] if emotion.enabled else []
    remember = ([('Julia memory classifier', memory.device, False, memory.system1_model_id, memory.system1_cache_dir, None, memory.system1_revision)] if memory.system1_enabled else []) \
        + ([('sentence embeddings', memory.device, True, memory.embedding_model, None, None, None)] if memory.embeddings_enabled else [])
    return [('emotion_model', 'Julia emotion model', julia, 'emotion disabled'),
            ('memory_models', 'Memory classifier and embeddings', remember, 'memory classifier and embeddings disabled')]


def torch_rows(config, add, used):
    """One row per torch model group: 0 on the CPU, weights x1.0-1.3 + 128 MiB on a GPU, 0..that for an unresolved auto, and
    unknown when the weights are not on disk yet. `used` collects the GPU kinds (cuda, mps, auto) they need."""
    for key, label, models, off in torch_models(config):
        low = high = 0
        notes = []
        for name, setting, mps, model_id, cache_dir, path, revision in models:
            device, exact = torch_target(setting, mps=mps)
            if device == 'cpu':
                notes.append(f'{name} on the CPU')
                continue
            used.add(device if exact or device == 'mps' else 'auto')  # an unresolved auto on macOS: MPS or the CPU, no CUDA context
            size = weights_mib(model_id, cache_dir, path, revision=revision)
            top = None if size is None else size * 1.3 + 128
            low = None if low is None or (exact and size is None) else low + (size + 128 if exact else 0)
            high = None if high is None or top is None else high + top
            notes.append(f'{name} on {device}' + (' (auto: torch not loaded yet, so maybe the CPU)' if not exact else '')
                + (f': {size:,.0f} MiB of weights x1.0-1.3 + 128 MiB' if size is not None else ': weights not downloaded yet, size unknown'))
        add(key, label, low, high, '; '.join(notes) if notes else off.capitalize())


def aggregate(rows):
    def total(key):
        values = [row.get(key) for row in rows]
        return None if any(value is None for value in values) else sum(values)
    return {'key': '+'.join(str(row.get('key')) for row in rows), 'aggregate': True, 'total_mib': total('total_mib'), 'used_mib': total('used_mib'),
        'other_mib': total('other_mib'), 'backend': rows[0].get('backend'),
        'memory_kind': 'unified' if all(row.get('memory_kind') == 'unified' for row in rows) else 'dedicated',
        'memory_scope': 'process' if any(row.get('memory_scope') == 'process' for row in rows) else 'device'}


def draft_telemetry(telemetry, running, draft):
    """The telemetry a draft is held against. riko_devices numbers devices as the running build does (CUDA fastest first,
    Vulkan its own order), so its list maps a draft's main_gpu only when the draft loads the same native library through
    the same provider; otherwise the estimate falls back to the driver's numbering and says main_gpu is a guess."""
    def same(runtime): return runtime.provider.lower().replace('-', '_'), runtime.native_library
    if telemetry.get('ggml_devices') is None or (running is not None and same(running) == same(draft)): return telemetry
    return {**telemetry, 'ggml_devices': None, 'ggml_devices_note': 'the draft loads another native library than the running backend'}


def target_gpu(runtime, telemetry, managed, warnings):
    """The telemetry row the estimate is held against: llama.cpp's own device choice from riko_devices (several rows summed
    when the model spans them); without it the only GPU, else the row numbered main_gpu (a guess, said so)."""
    gpus = [gpu for gpu in telemetry.get('gpus') or [] if gpu.get('total_mib')]
    if not gpus: return None
    devices = telemetry.get('ggml_devices')
    if managed and devices is not None:
        from ..inference.devices import model_devices
        chosen = model_devices(devices, split_mode=runtime.split_mode, main_gpu=runtime.main_gpu)
        if chosen is None:
            warnings.append(f'runtime.main_gpu {runtime.main_gpu} names no GPU llama.cpp can use here: llama.cpp will refuse to load the model.')
            return None
        if not chosen:
            warnings.append('This llama.cpp build sees no GPU, so the model runs on the CPU: GPU estimates do not apply to it.')
            return None
        rows = [row for row in (next((gpu for gpu in gpus if gpu.get('ggml_index') == device['index']), None) for device in chosen) if row]
        if len(rows) != len(chosen):
            warnings.append('GPU telemetry does not cover every device llama.cpp would use: no projection.')
            return None
        if len(rows) == 1: return rows[0]
        warnings.append(f'The model spans {len(rows)} GPUs: their memory is summed and the fit of each GPU is not checked.')
        return aggregate(rows)
    if not managed:
        return next((gpu for gpu in gpus if gpu.get('key') and gpu['key'] == telemetry.get('main_key')), gpus[0] if len(gpus) == 1 else None)
    if len(gpus) == 1: return gpus[0]
    why = telemetry.get('ggml_devices_note') or 'an older riko-native, or the model not loaded yet'
    warnings.append(f'runtime.main_gpu is a llama.cpp device index; without riko_devices ({why}) '
        f'Riko assumes it is GPU {runtime.main_gpu} as the driver numbers them, which may be another GPU.')
    return next((gpu for gpu in gpus if gpu.get('index') == runtime.main_gpu), None)


def system_ram_mib(telemetry):
    """Physical RAM from ggml's CPU device (it reports RAM as its total), or None."""
    return next((device['memory_total'] / MIB for device in telemetry.get('ggml_devices') or [] if device.get('type') == 'cpu' and device.get('memory_total')), None)


def estimate(config, telemetry, metadata=None):
    runtime = config.runtime
    components, warnings = [], []
    def add(key, label, low, high, basis):
        components.append({'id': key, 'label': label, 'low_mib': low, 'high_mib': high, 'basis': basis})
    managed = runtime.provider.lower().replace('-', '_') == 'llama_cpp'
    gpu = target_gpu(runtime, telemetry, managed, warnings)
    unified, process_scope = bool(gpu) and gpu.get('memory_kind') == 'unified', bool(gpu) and gpu.get('memory_scope') == 'process'
    meta = metadata or {}
    if managed and metadata is None:
        try: meta = model_metadata(runtime)
        except Exception as exc: warnings.append('LLM metadata unavailable: ' + str(exc))
    pool = context_capacity(runtime) if managed else None  # only the in-process provider allocates a KV pool
    kv_per_token = None
    if managed and runtime.n_gpu_layers != 0:
        layers, embedding, heads, kv_heads = (meta.get(k) for k in ('layers', 'embedding', 'heads', 'kv_heads'))
        fraction = min(1, runtime.n_gpu_layers / max(1, layers)) if layers and runtime.n_gpu_layers >= 0 else 1
        weight = meta.get('weight_bytes')
        add('llm_weights', 'Shared LLM weights (one copy)', weight / MIB * fraction if weight else None,
            weight / MIB * fraction * 1.1 if weight else None, meta.get('source', 'Unknown model metadata') + '; partial offload approximated by layer fraction' + ('; -1 assumes the automatic fit offloads every layer' if runtime.n_gpu_layers == -1 else ''))
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
            add('llm_compute', 'LLM compute buffers', max(128, batch), max(384, batch * (1.25 if runtime.flash_attn in ('on', True) else 2)),
                'Heuristic workspace affected by batch, microbatch and flash attention' + ('; includes CUDA graphs' if gpu and gpu.get('backend') == 'CUDA' else ''))
        else:
            add('kv', 'KV pool — all slots occupied', None, None, 'Attention metadata unavailable; cannot compute KV bytes per token')
            add('llm_compute', 'LLM compute buffers', 128, 768, 'Heuristic; missing architecture details')
    else:
        add('llm_weights', 'LLM', 0, 0, 'CPU-only owned model' if managed else 'External provider excluded (any local GPU use appears in Other)')
    voice = config.raw.get('voice') or {}
    from ..audio.asr import MODEL, resolve
    device, precision, note = resolve(voice)
    used = {'cuda'} if device != 'cpu' else set()  # CTranslate2 runs on the CPU or CUDA (a HIP build reports cuda too)
    model = str(voice.get('asr_model', MODEL)).lower()
    base = next((size for name, size in [('large', 3100), ('medium', 1600), ('small', 600), ('base', 200), ('tiny', 100)] if name in model), None)
    if note: warnings.append(f'Speech recognition runs as {device}/{precision} here: {note}.')
    if device == 'cpu': add('asr', 'Whisper / faster-whisper ASR', 0, 0, f'CPU ({precision})')
    elif base:
        factor = .65 if precision.startswith('int8') else 2 if precision == 'float32' else 1
        add('asr', 'Whisper / faster-whisper ASR', base * factor + 128, base * factor * 1.4 + 384,
            f'{model}, {precision}, {device}; heuristic resident weights + decoder workspace, included even if microphone is idle')
    else: add('asr', 'Whisper / faster-whisper ASR', None, None, f'Unknown custom ASR model {model}; no invented size')
    torch_rows(config, add, used)
    add('wake_vad', 'Wake detector and VAD', 0, 0, 'ONNX CPU wake detector and CPU Silero VAD; no separate flow model owned by this application')
    add('electron', 'Electron / VRM / whiteboard GPU surfaces', 96, 384, 'Heuristic graphics textures/framebuffers; driver and assets affect this')
    if 'cuda' in used: add('gpu_runtime', 'Python GPU runtime (CUDA/ROCm context)', 128, 384, 'One CUDA/HIP context reserve shared by the Python models on the GPU, not one per model')
    elif 'auto' in used: add('gpu_runtime', 'Python GPU runtime (CUDA/ROCm context)', 0, 384, 'A model set to auto may land on CUDA/ROCm (one context reserve) or elsewhere (none)')
    else: add('gpu_runtime', 'Python GPU runtime (CUDA/ROCm context)', 0, 0, 'Metal (MPS): no separate reserve' if 'mps' in used else 'No Python model on a CUDA/ROCm GPU')
    add('tts', 'TTS / external providers', 0, 0, 'External processes excluded from owned estimates; their real GPU use remains in Other')
    unknown = [c['label'] for c in components if c['high_mib'] is None]
    low = sum(c['low_mib'] or 0 for c in components)
    high = sum(c['high_mib'] or 0 for c in components)
    # Metal's working set is this process's own: other processes (Electron's surfaces too) never count against it.
    electron = next(c for c in components if c['id'] == 'electron')
    own_low, own_high = (low - electron['low_mib'], high - electron['high_mib']) if process_scope else (low, high)
    other = 0 if process_scope else gpu.get('other_mib') if gpu else None
    ram = system_ram_mib(telemetry)
    if unified and process_scope:
        warnings.append(f"Apple unified memory: the budget is Metal's recommended working set for this process ({gpu['total_mib']:,.0f} MiB"
            + (f' of {ram:,.0f} MiB RAM' if ram else '') + '); other apps share that RAM and are not counted, and Electron draws in its own process.')
    elif unified: warnings.append("Integrated GPU: it shares system RAM with every other program; the budget is the shared pool llama.cpp's backend reports for it.")
    cpu_model = unified and managed and runtime.n_gpu_layers == 0
    if cpu_model: warnings.append('With GPU layer offload 0 the model runs on the CPU: on unified memory its weights and KV cache still take '
        'system RAM, outside the GPU budget, so no GPU projection is shown.')
    suggestion = None
    reserved = None if not gpu else other if other is not None else gpu.get('used_mib')
    budget = gpu['total_mib'] if gpu else None
    if unified and process_scope:
        # Metal's working set (about 3/4 of RAM on large Macs) counts this process alone, so a ceiling that filled it would
        # leave other apps the rest: keep a quarter of RAM for macOS and them, and Electron's surfaces, out of the suggestion.
        budget, reserved = min(budget, ram * .75) if ram else budget, electron['high_mib']
    if gpu and reserved is not None and not gpu.get('aggregate') and not unknown and kv_per_token and not any('no reliable' in w for w in warnings):
        fixed = own_high - next(c['high_mib'] for c in components if c['id'] == 'kv')
        available = budget - reserved - fixed - max(256, budget * .05)
        maximum_pool = max(0, int(available / (kv_per_token * 1.15)))
        if runtime.kv_unified:
            suggested_live = max(0, maximum_pool - (pool - runtime.n_ctx))
        else:
            suggested_live = maximum_pool // runtime.parallel_slots
            if suggested_live < max(runtime.initiative_n_ctx, runtime.reflection_n_ctx): suggested_live = 0
        model_limit = meta.get('context_length')
        if model_limit: suggested_live = min(suggested_live, model_limit)
        suggested_live = min(suggested_live, 1048576)
        if suggested_live < runtime.max_output_tokens + 256: suggested_live = 0
        suggestion = {'live_context_tokens': suggested_live // 256 * 256, 'maximum_pool_tokens': maximum_pool,
            'basis': 'Upper estimate + 5%/256 MiB safety reserve; background budgets fixed; advisory only'
                + ("; Metal's working set counts this process only: capped at 75% of RAM with Electron's surfaces reserved, an upper "
                   "bound, since other apps' current RAM use is not measured" if process_scope else '')
                + ('; attribution unavailable: conservatively reserves ALL currently used VRAM, which may double-count already-running owned components' if other is None else '')}
    if runtime.tensor_split: warnings.append('Multi-GPU split estimates are aggregate, not per-device; recommendation suppressed.'); suggestion = None
    projection = None
    if gpu and other is not None and not unknown and not runtime.tensor_split and not cpu_model:
        projection = {'low_mib': own_low + other, 'high_mib': own_high + other, 'total_mib': gpu['total_mib'],
            'fits_upper_estimate': own_high + other <= gpu['total_mib'], 'unified': unified, 'scope': gpu.get('memory_scope', 'device'), 'gpu_key': gpu.get('key')}
    if managed and runtime.n_gpu_layers == -1 and projection and projection['high_mib'] + 1024 > projection['total_mib']:
        # llama.cpp's --fit offloads every layer only while 1 GiB of device memory stays free at load time.
        warnings.append('With GPU layer offload -1, llama.cpp keeps 1 GiB of GPU memory free and would probably run some layers on the CPU (slower) at these settings; -2 requires every layer on the GPU.')
    return {'components': components, 'low_mib': low, 'high_mib': high, 'complete': not unknown, 'unknown_components': unknown,
        'confidence': 'low',
        'confidence_basis': 'Model/KV metadata is used where available; compute, recurrent state, ASR and graphics reserves remain heuristic, not a native backend dry-run allocation report.',
        'warnings': warnings, 'suggestion': suggestion, 'kv': {'unified': runtime.kv_unified, 'pool_tokens': pool,
        'live_context_tokens': runtime.n_ctx, 'initiative_context_tokens': runtime.initiative_n_ctx,
        'reflection_context_tokens': runtime.reflection_n_ctx, 'slots': runtime.parallel_slots},
        'managed': managed, 'gpu_key': gpu.get('key') if gpu else None,  # the telemetry row (or rows, joined by +) it is held against
        'projection': projection,
        'note': 'Estimated peak residency, not measured allocations. Ranges and unknowns are intentional. Recommendations never modify settings.'}
