"""The runtime section's settings (kernel/schema.py): which provider uses each key, its Settings metadata, and what
load_config and Settings check for the llama.cpp providers (llama_runtime.py) and the in-process KV pool (kv_budget.py).
RuntimeConfig itself is configuration's: Settings lists its effective values."""
from ..kernel.schema import Section, Setting, register
from .kv_budget import pool_capacity
from .llama_runtime import FLASH_ATTENTION, KV_TYPES, RULES, configure_runtime, flash_attention

OPENAI_COMPATIBLE = ('lm_studio', 'openai', 'openai_compatible', 'ollama', 'local_http')  # OpenAIProvider (providers.py)
PROVIDERS = ('llama_cpp', 'llama_server', *OPENAI_COMPATIBLE)
NATIVE = (('runtime.provider', ('llama_cpp',)),)  # the in-process library
SLOTS = (('runtime.provider', ('llama_cpp', 'llama_server')),)  # Riko schedules llama.cpp's slots
SERVED = (('runtime.provider', ('llama_server', *OPENAI_COMPATIBLE)),)  # a server at runtime.base_url
OPENAI = (('runtime.provider', OPENAI_COMPATIBLE),)
# Settings that change the live conversation budget (memory.context_window_tokens + runtime.max_output_tokens <= n_ctx).
BUDGET_KEYS = {'runtime.provider', 'runtime.n_ctx', 'runtime.max_output_tokens', 'memory.context_window_tokens',
    'presets.default.model_params.context_window_token_limit', 'presets.default.model_params.max_output_tokens'}


def configure(config, raw):
    """load_config: the provider's own checks, then the in-process KV pool, sized from the background budgets."""
    runtime = config.runtime
    configure_runtime(runtime, raw.get('runtime', {}))
    if str(runtime.provider).lower().replace('-', '_') == 'llama_cpp' and runtime.kv_pool_auto:  # only it allocates a pool
        runtime.kv_pool_tokens = pool_capacity(runtime)


def effective(config, values):
    if type(values.get('runtime.flash_attn')) is bool: values['runtime.flash_attn'] = flash_attention(values['runtime.flash_attn'])
    if config.runtime.kv_pool_auto: values['runtime.kv_pool_tokens'] = config.runtime.kv_pool_tokens


def review(candidate, draft, changes):
    runtime, errors = candidate.runtime, {}
    if runtime.provider != 'llama_cpp': return errors
    if runtime.n_ctx <= 0: errors['runtime.n_ctx'] = 'Managed server requires a positive context'
    if runtime.split_mode == 'row': errors['runtime.split_mode'] = 'Row split is not available in this llama.cpp build; choose layer or none'
    # Check the live budget only when one of its inputs is edited: the runtime clamps it (factory.py), so a config that
    # does not fit, such as one an older setup wizard wrote, never blocks saving other settings.
    budget = candidate.memory.context_window_tokens + runtime.max_output_tokens
    if budget > runtime.n_ctx and BUDGET_KEYS & changes.keys(): errors['runtime.n_ctx'] = f'Context must fit conversation budget + response ({budget} tokens)'
    return errors


def derive(candidate):
    """The KV pool Settings writes with every save while it is automatic; a manual pool too small for the budgets fails."""
    runtime = candidate.runtime
    if runtime.provider != 'llama_cpp': return {}
    pool = pool_capacity(runtime)
    return {'runtime.kv_pool_auto': True, 'runtime.kv_pool_tokens': pool} if runtime.kv_pool_auto else {}


SOURCE, BUDGETS, GENERATION, SCHEDULING = (dict(section=name) for name in ('Model source', 'Token budgets', 'Generation', 'Scheduling & startup'))
PYTHON = ' Save and restart Python.'
register(Section('runtime', group='models', title='Compute & cache', configure=configure, effective=effective, review=review, derive=derive,
    rules=RULES, settings=(
        Setting('provider', options=PROVIDERS, label='Backend', **SOURCE,
            help='llama_cpp runs llama.cpp inside Python with a riko-native library. llama_server connects to a llama-server you run, built for any backend (CUDA, ROCm, Metal, Vulkan or CPU), and keeps slots, exact token counts and streaming. The others use OpenAI-compatible APIs.' + PYTHON),
        Setting('native_library', nullable=True, file=True, label='In-process llama.cpp library', visible_when=NATIVE, **SOURCE,
            help='Required for llama_cpp: a compatible riko-native library. Runs llama.cpp inside Python, including Responses, tools, streaming, cancellation and probe capture. No server process or HTTP listener. Use a CUDA-enabled build for GPU acceleration.' + PYTHON),
        Setting('model_path', nullable=True, label='Local GGUF file', visible_when=NATIVE, **SOURCE, help='An explicit local GGUF takes precedence over Hugging Face.'),
        Setting('hf_repo_id', nullable=True, label='Hugging Face model', visible_when=NATIVE, **SOURCE,
            help='Search Hugging Face, or paste an owner/repository ID. No download occurs until backend restart.'),
        Setting('hf_filename', nullable=True, label='GGUF file', visible_when=NATIVE, **SOURCE,
            help='Choose an exact GGUF. For split models choose the first shard; not the mmproj file.'),
        Setting('hf_revision', label='Model revision', visible_when=NATIVE, **SOURCE),
        Setting('hf_local_files_only', label='Offline mode', visible_when=NATIVE, **SOURCE),
        # OpenAIProvider's alone.
        Setting('tokenizer_model', nullable=True, advanced=True, visible_when=OPENAI, **SOURCE),
        Setting('model', visible_when=OPENAI, **SOURCE),
        Setting('api_mode', options=('auto', 'responses', 'chat_completions'), visible_when=OPENAI, **SOURCE),
        Setting('reuse_response_ids', visible_when=OPENAI, **SOURCE),
        Setting('base_url', visible_when=SERVED, **SOURCE,
            help='Server address. For llama_server, the llama-server address such as http://127.0.0.1:8080; its --parallel must equal Parallel inference slots and each slot needs at least the live, initiative and reflection context. Conversations are sent to this address.' + PYTHON),
        Setting('api_key', label='API key', visible_when=SERVED, **SOURCE),
        Setting('n_ctx', range=(1, 1048576), label='Live context length (tokens)', **BUDGETS,
            help='Live conversation context including output. With unified KV, the pool adds the worst concurrent initiative/reflection budgets instead of duplicating this size per slot.'),
        Setting('max_output_tokens', range=(1, 1048576), label='Response token limit', **BUDGETS),
        Setting('temperature', **GENERATION), Setting('seed', advanced=True, visible_when=NATIVE, **GENERATION),
        Setting('parallel_slots', range=(2, 4), label='Parallel inference slots', visible_when=SLOTS, **SCHEDULING,
            help='One slot reserved for live replies. Other slots prioritize initiative over reflection.'),
        Setting('warmup', **SCHEDULING, help='Preload models and silently test TTS. Does not open your microphone.'),
        Setting('startup_timeout_seconds', range=(1, 86400), label='Startup timeout (seconds)', visible_when=SLOTS, **SCHEDULING,
            help='Deadline for optional component warmup (ASR, TTS and auxiliary models), and for an external llama-server to finish loading its model; not a timeout for synchronous native model loading. Native request waits use Request timeout. Restart Python.'),
        Setting('request_timeout_seconds', range=(.1, 86400), label='Request inactivity timeout (seconds)', **SCHEDULING),
        Setting('pause_background_on_live', restart='none', live='background_pause', label='Pause background inference during foreground turns',
            visible_when=SLOTS, **SCHEDULING,
            help='Pause/preempt managed initiative and reflection throughout foreground turns, including tool waits. Turn off to allow parallel background inference. Applies immediately when saved; does not shrink allocated KV capacity.'),
        Setting('n_gpu_layers', range=(-2, 1000), label='GPU layer offload', visible_when=NATIVE,
            help='-1 fits as many layers as free GPU memory allows at load time, keeping 1 GiB free (layers left over run on the CPU, slower). -2 requires every layer on the GPU. 0 or more is an exact layer count.'),
        Setting('n_batch', range=(1, 65536), label='Prompt batch size', visible_when=NATIVE),
        Setting('n_ubatch', range=(1, 65536), label='Physical batch size', visible_when=NATIVE),
        Setting('n_threads', kind='number', integer=True, nullable=True, range=(1, 1024), advanced=True, label='CPU generation threads', visible_when=NATIVE),
        Setting('n_threads_batch', kind='number', integer=True, nullable=True, range=(1, 1024), advanced=True, label='CPU prompt threads', visible_when=NATIVE),
        Setting('flash_attn', options=FLASH_ATTENTION, label='Flash attention', visible_when=NATIVE,
            help='auto lets llama.cpp use flash attention where the GPU backend supports it for this model. Quantized V cache needs auto or on (it forces flash attention on).'),
        Setting('type_k', options=KV_TYPES, label='Key cache precision', visible_when=NATIVE, help='KV tensor quantization, independent of model weight quantization.'),
        Setting('type_v', options=KV_TYPES, label='Value cache precision', visible_when=NATIVE),
        Setting('offload_kqv', label='GPU KV offload', visible_when=NATIVE), Setting('use_mmap', label='Memory-mapped weights', visible_when=NATIVE),
        Setting('use_mlock', advanced=True, label='Lock weights in RAM', visible_when=NATIVE),
        Setting('main_gpu', range=(0, 128), advanced=True, label='Primary GPU', visible_when=NATIVE),
        Setting('split_mode', options=('none', 'layer'), advanced=True, label='GPU split strategy', visible_when=NATIVE,
            help='How a model is spread over several GPUs: layer gives each GPU whole layers; none uses only the primary GPU. Row split is not available in this llama.cpp build.'),
        Setting('tensor_split', kind='json', nullable=True, advanced=True, label='GPU allocation weights', visible_when=NATIVE),
        Setting('chat_format', nullable=True, advanced=True, label='Chat template override', visible_when=NATIVE),
        Setting('cache_size_mb', range=(0, 1048576), advanced=True, label='RAM prompt cache (MiB)', visible_when=NATIVE,
            help='Optional extra RAM prompt cache; 0 still preserves per-slot live prefix reuse.'),
        Setting('verbose', advanced=True, visible_when=NATIVE),
        Setting('kv_unified', label='Shared KV pool', visible_when=NATIVE,
            help='Use one native KV pool sized for simultaneous live and background demands. Disabled: each slot allocates the largest task context.'),
        Setting('kv_pool_auto', visible_when=NATIVE, help='Keep the shared pool at the calculated maximum concurrent token demand; saved with all other settings.'),
        Setting('kv_pool_tokens', kind='number', integer=True, nullable=True, range=(1, 4194304), visible_when=NATIVE,
            help='Total KV token capacity across all slots. Automatic mode recalculates this from task budgets; manual mode must be at least the suggested size.'),
        Setting('server_path', obsolete=True))))
