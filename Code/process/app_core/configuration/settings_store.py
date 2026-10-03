"""Round-trip YAML settings with typed forms, validation and conflict-safe saves."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading

from .config import load_config

LOCK = threading.RLock()
OBSOLETE = {'avatar.camera.distance', 'avatar.expression_engine', 'avatar.view', 'desktop.shortcuts.effects',
            'emotion.temperature', 'emotion.device', 'sovits_ping_config.media_type', 'your_name',
            'model','base_url','api_key','tokenizer_model'}
ENUMS = {
    'runtime.provider': ['llama_cpp', 'lm_studio', 'openai', 'openai_compatible', 'ollama', 'local_http'],
    'runtime.api_mode': ['auto', 'responses', 'chat_completions'],
    'runtime.type_k': ['f16', 'f32', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl'],
    'runtime.type_v': ['f16', 'f32', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl'],
    'runtime.split_mode': ['none', 'layer', 'row'], 'voice.mode': ['wake_word', 'continuous', 'manual'],
    'voice.asr_device': ['cuda', 'cpu', 'auto'], 'voice.asr_compute_type': ['int8_float16', 'float16', 'int8', 'float32', 'default'],
    'emotion.device': ['cpu'], 'avatar.format': ['auto', 'vrm0', 'vrm1'], 'avatar.view': ['full_body'],
    'sovits_ping_config.text_lang': ['en', 'zh', 'ja', 'ko', 'yue', 'auto'],
    'sovits_ping_config.prompt_lang': ['en', 'zh', 'ja', 'ko', 'yue', 'auto'],
    'sovits_ping_config.media_type': ['raw'],
}
RANGES = {
    'runtime.parallel_slots': (2, 4), 'runtime.n_ctx': (1, 1048576),
    'runtime.n_gpu_layers': (-1, 1000), 'runtime.n_batch': (1, 65536), 'runtime.n_ubatch': (1, 65536),
    'runtime.n_threads': (1, 1024), 'runtime.n_threads_batch': (1, 1024),
    'runtime.request_timeout_seconds': (.1, 86400), 'runtime.startup_timeout_seconds': (1, 86400),
    'runtime.cache_size_mb': (0, 1048576), 'runtime.main_gpu': (0, 128),
    'voice.wake_threshold': (.01, .99), 'voice.follow_up_seconds': (.1, 300),
    'voice.interruption_seconds': (.1, 60), 'voice.input_device': (0, 10000),
    'voice.live_transcript_interval_seconds': (.5, 10),
    'tools.timeout_seconds': (.1, 3600), 'tools.max_iterations': (1, 100),
    'tools.best_fit_timeout_seconds': (.05, 2), 'tools.best_fit_min_confidence': (.5, 1),
    'memory.minimum_importance': (0, 1), 'memory.reflection_min_importance': (0, 1),
    'sovits_ping_config.max_in_flight_requests': (1, 32),
    'speech.max_words': (1, 1000), 'speech.split_window_words': (0, 1000),
    'wake_feedback.volume': (0, 1), 'wake_feedback.max_clip_seconds': (.1, 10),
    'wake_feedback.cooldown_seconds': (0, 30),
    'animation.policy_timeout_seconds': (.05, 10), 'animation.min_dwell_seconds': (0, 30),
    'animation.transition_seconds': (.05, 3), 'animation.min_confidence': (0, 1), 'animation.walk_speed': (20, 1000),
    'avatar.camera.fov': (10, 80),
    'runtime.max_output_tokens': (1, 1048576),
    'runtime.kv_pool_tokens': (1, 4194304),
    'initiative.max_output_tokens': (1, 1048575), 'initiative.context_window_tokens': (256, 1048576),
    'memory.reflection_max_output_tokens': (1, 1048575), 'memory.reflection_context_window_tokens': (256, 1048576),
    'memory.context_window_tokens': (1, 1048576),
    'presets.default.model_params.context_window_token_limit': (1, 1048576),
    'presets.default.model_params.max_output_tokens': (1, 1048576),
}
INTEGER_NULLS = {'runtime.n_threads', 'runtime.n_threads_batch', 'runtime.kv_pool_tokens', 'voice.input_device'}
JSON_NULLS = {'runtime.tensor_split'}
OPTIONAL_TEXT = {'runtime.model_path', 'runtime.hf_repo_id', 'runtime.hf_filename', 'runtime.chat_format'}
OPTIONAL_TEXT |= {'runtime.tokenizer_model', 'emotion.model_path', 'emotion.cache_dir', 'memory.system1_cache_dir'}
HELP = {
    'runtime.server_path': 'Path to a recent llama-server executable. Riko manages one model instance.',
    'runtime.parallel_slots': 'One slot reserved for live replies. Other slots prioritize initiative over reflection.',
    'runtime.pause_background_on_live': 'Pause/preempt managed initiative and reflection throughout foreground turns, including tool waits. Turn off to allow parallel background inference. Applies immediately when saved; does not shrink allocated KV capacity.',
    'runtime.n_ctx': 'Live conversation context including output. With unified KV, the pool adds the worst concurrent initiative/reflection budgets instead of duplicating this size per slot.',
    'runtime.kv_unified': 'Use one shared KV pool sized for simultaneous live and background demands. Requires a recent llama-server. Disabled: each slot allocates the largest task context.',
    'runtime.kv_pool_auto': 'Keep the shared pool at the calculated maximum concurrent token demand; saved with all other settings.',
    'runtime.kv_pool_tokens': 'Total KV token capacity across all slots. Automatic mode recalculates this from task budgets; manual mode must be at least the suggested size.',
    'initiative.max_output_tokens': 'Output budget including reasoning. Invalid/empty/truncated decisions fail the attempt; no repair inference.',
    'initiative.context_window_tokens': 'Total initiative context including output/tools. Older recent dialogue is token-trimmed while character/emotion state stays. Restart to resize the managed KV pool.',
    'memory.reflection_max_output_tokens': 'Reflection output budget including reasoning. Invalid/empty/truncated output errors that job without a format-repair retry.',
    'memory.reflection_context_window_tokens': 'Total reflection context including evidence/instructions/output. Optional related evidence and formation history are token-trimmed; required focal content stays.',
    'runtime.hf_repo_id': 'Search Hugging Face, or paste an owner/repository ID. No download occurs until backend restart.',
    'runtime.hf_filename': 'Choose an exact GGUF. For split models choose the first shard; not the mmproj file.',
    'runtime.model_path': 'An explicit local GGUF takes precedence over Hugging Face.',
    'runtime.flash_attn': 'Hardware/build-dependent. Quantized V cache requires flash attention.',
    'runtime.type_k': 'KV tensor quantization, independent of model weight quantization.',
    'runtime.cache_size_mb': 'Optional extra RAM prompt cache; 0 still preserves per-slot live prefix reuse.',
    'runtime.warmup': 'Preload models and silently test TTS. Does not open your microphone.',
    'voice.input_device': 'Automatic uses the system microphone. Changes apply on backend restart.',
    'voice.wake_word': 'One short name supported by EfficientWord-Net. Changing name/device requires its matching enrollment.',
    'voice.live_transcript_interval_seconds': 'Rolling provisional ASR updates while you are still speaking. One partial job at a time; final transcription replaces partial text. Smaller intervals increase ASR work. Applies on microphone restart.',
    'tools.require_approval': 'Default for new/unconfigured tools. Per-tool live permissions override this default; approval bubbles expire after two minutes without executing.',
    'tools.best_fit_inputs': 'Correct finite-choice desktop inputs before approval. Reuses the enabled, already-loaded CPU Julia model; safe spelling normalization is the fallback. Never repairs file permissions, numeric values or arbitrary paths.',
    'desktop.shortcuts': 'Shortcut configuration is saved; some legacy shortcuts are not implemented.',
    'desktop.shortcuts.effects': 'Legacy effect editor shortcut; currently not registered.',
    'avatar.camera.distance': 'Legacy fixed distance; current renderer frames the whole model automatically.',
    'avatar.camera.fov': 'Perspective field of view; automatic fitting still keeps the whole model visible.',
    'avatar.model': 'Choose or import a VRM in Desktop & avatar. Saved avatar changes apply immediately.',
    'avatar.format': 'Auto detects VRM 0.x/1.0. Explicit formats must match the model; this does not convert it.',
    'avatar.view': 'Current renderer supports full-body framing only.',
    'avatar.expression_engine': 'Native VRM expressions are driven by Julia; retained for compatibility.',
    'sovits_ping_config.media_type': 'Client requires raw PCM streaming; keep raw.',
    'speech.max_words': 'Soft word limit. Short replies stay whole; longer replies split at preferred punctuation. No forced mid-sentence cut.',
    'speech.split_window_words': 'Search this many words before the limit. If no boundary exists, wait for the next punctuation or generation end. Must not exceed the word limit.',
    'speech.split_priority': 'JSON list of punctuation groups, highest priority first. Default: sentence endings, semicolon/colon, comma, newline. Each character must appear only once.',
    'wake_feedback.rules': 'JSON list of emotion/state rules with local audio (.wav/.flac/.ogg) and animation (.vrma) paths. Exact state+emotion wins, then state, emotion and wildcard. Assets must be in approved media roots. No assets means silent.',
    'wake_feedback.max_clip_seconds': 'Maximum local acknowledgement length. Longer clips are rejected; microphone capture continues. Use headphones to avoid cue echo.',
    'wake_feedback.volume': 'Default cue volume, 0–1. Respects the global audio toggle; avatar animation is independent.',
    'animation.julia_selection': 'Reuse the enabled Julia emotion model to select eligible motion intents. Rules provide immediate fallback; no extra model copy is loaded.',
    'animation.walk_speed': 'Desktop movement speed in pixels/second. Walking is explicitly requested; dragging always cancels it.',
}
LABELS = {
    'runtime.provider':'Backend', 'runtime.model_path':'Local GGUF file', 'runtime.server_path':'llama-server executable',
    'runtime.hf_repo_id':'Hugging Face model', 'runtime.hf_filename':'GGUF file', 'runtime.hf_revision':'Model revision',
    'runtime.hf_local_files_only':'Offline mode', 'runtime.n_ctx':'Live context length (tokens)',
    'runtime.kv_unified':'Shared KV pool',
    'runtime.n_batch':'Prompt batch size', 'runtime.n_ubatch':'Physical batch size',
    'runtime.n_threads':'CPU generation threads', 'runtime.n_threads_batch':'CPU prompt threads',
    'runtime.type_k':'Key cache precision', 'runtime.type_v':'Value cache precision', 'runtime.flash_attn':'Flash attention',
    'runtime.offload_kqv':'GPU KV offload', 'runtime.use_mmap':'Memory-mapped weights', 'runtime.use_mlock':'Lock weights in RAM',
    'runtime.main_gpu':'Primary GPU', 'runtime.split_mode':'GPU split strategy', 'runtime.tensor_split':'GPU allocation weights',
    'runtime.cache_size_mb':'RAM prompt cache (MiB)', 'runtime.chat_format':'Chat template override',
    'runtime.max_output_tokens':'Response token limit', 'runtime.n_gpu_layers':'GPU layer offload',
    'runtime.parallel_slots':'Parallel inference slots', 'runtime.startup_timeout_seconds':'Startup timeout (seconds)',
    'runtime.pause_background_on_live':'Pause background inference during foreground turns',
    'runtime.request_timeout_seconds':'Request inactivity timeout (seconds)', 'runtime.api_key':'API key',
    'voice.asr_model':'Speech recognition model', 'voice.asr_device':'Speech recognition device', 'voice.asr_compute_type':'Speech recognition precision',
    'voice.input_device':'Microphone', 'sovits_ping_config.url':'GPT-SoVITS endpoint',
    'sovits_ping_config.ref_audio_path':'Reference audio file', 'sovits_ping_config.prompt_text':'Reference transcript',
    'speech.max_words':'Speech chunk limit (words)', 'speech.split_window_words':'Boundary search window (words)',
    'speech.split_priority':'Punctuation priority',
}
ADVANCED_MODEL = {'runtime.seed', 'runtime.n_threads', 'runtime.n_threads_batch', 'runtime.tokenizer_model',
    'runtime.main_gpu', 'runtime.split_mode', 'runtime.tensor_split', 'runtime.use_mlock', 'runtime.chat_format',
    'runtime.cache_size_mb', 'runtime.verbose'}


def model_section(path):
    key = path.rsplit('.', 1)[-1]
    if path.startswith('presets.'): return 'Preset fallbacks'
    if key in {'provider','server_path','model_path','hf_repo_id','hf_filename','hf_revision','hf_local_files_only','tokenizer_model','model','base_url','api_key','api_mode','reuse_response_ids'}: return 'Model source'
    if key in {'n_ctx','max_output_tokens'}: return 'Token budgets'
    if key in {'temperature','seed'}: return 'Generation'
    if key in {'parallel_slots','warmup','startup_timeout_seconds','request_timeout_seconds','pause_background_on_live'}: return 'Scheduling & startup'
    return 'Compute & cache'


class SettingsConflict(ValueError): pass


def revision(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()


def flatten(value, prefix=''):
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(item, dict) and item:
            yield from flatten(item, path)
        else: yield path, item


def field(path, value):
    kind = 'boolean' if type(value) is bool else 'number' if type(value) in (int, float) or path in INTEGER_NULLS else 'json' if isinstance(value, (list, dict)) or path in JSON_NULLS else 'text'
    nullable = value is None or path in INTEGER_NULLS | JSON_NULLS | OPTIONAL_TEXT
    group = {'runtime':'models', 'voice':'voice', 'speech':'speech', 'sovits_ping_config':'speech', 'memory':'memory',
        'wake_feedback':'voice',
        'emotion':'memory', 'initiative':'initiative', 'tasks':'tools', 'tools':'tools',
        'avatar':'appearance', 'desktop':'appearance', 'animation':'appearance'}.get(path.split('.')[0], 'character')
    if path.startswith('presets.default.model_params.'): group = 'models'
    resource_sections = {
        'voice.asr_model': 'Speech recognition model', 'voice.asr_device': 'Speech recognition model', 'voice.asr_compute_type': 'Speech recognition model',
        'memory.context_window_tokens': 'Token budgets', 'memory.token_budget': 'Token budgets', 'memory.max_results': 'Token budgets',
        'initiative.context_window_tokens': 'Token budgets', 'initiative.max_output_tokens': 'Token budgets',
        'memory.reflection_context_window_tokens': 'Token budgets', 'memory.reflection_max_output_tokens': 'Token budgets',
        'memory.reflection_enabled': 'Background models', 'memory.system1_enabled': 'Background models',
        'memory.system1_model_id': 'Background models', 'memory.system1_cache_dir': 'Background models', 'memory.system1_max_length': 'Token budgets',
        'memory.embeddings_enabled': 'Embedding model', 'memory.embedding_model': 'Embedding model', 'memory.embedding_dimension': 'Embedding model',
    }
    if path.startswith('emotion.'): resource_sections[path] = 'Emotion model'
    if path in resource_sections: group = 'models'
    result = dict(path=path, label=LABELS.get(path, path.rsplit('.', 1)[-1].replace('_', ' ').capitalize()), group=group,
        kind=kind, nullable=nullable, integer=type(value) is int or path in INTEGER_NULLS,
        help=HELP.get(path, ''), restart=True)
    result['section'] = model_section(path) if group == 'models' else 'Emotion' if path.startswith('emotion.') else 'Settings'
    if path.startswith('speech.'): result['section'] = 'Response chunking'
    if path.startswith('wake_feedback.'): result['section'] = 'Wake acknowledgement'
    if path.startswith('animation.'): result['section'] = 'Animation engine'
    if path.startswith('avatar.'): result['section'] = 'Avatar model'; result['restart'] = False
    if path == 'runtime.pause_background_on_live': result['restart'] = False
    if path.startswith('memory.reflection_'): result['section'] = 'Reflection'
    if path in resource_sections: result['section'] = resource_sections[path]
    result['advanced'] = path in ADVANCED_MODEL or path.startswith('presets.default.model_params.')
    result['readonly'] = path in {'avatar.camera.distance', 'desktop.shortcuts.effects', 'avatar.expression_engine'}
    if path in ENUMS: result['options'] = ENUMS[path]
    if path in RANGES: result['min'], result['max'] = RANGES[path]
    if kind == 'number' and 'min' not in result and path.endswith(('_seconds', '_tokens', '_length', '_size', '_dimension', '_results')): result['min'] = 0
    if 'importance' in path or 'temperature' in path:
        result.setdefault('min', 0); result.setdefault('max', 2 if 'temperature' in path else 1)
    result['secret'] = any(s in path.lower() for s in ('api_key', 'token', 'password')) and kind == 'text' and 'tokenizer' not in path
    result['file'] = kind == 'text' and (path.endswith(('_file', '_path', '_directory', '_dir')) or path in {'runtime.server_path', 'avatar.model'})
    result['multiline'] = kind == 'json' or 'prompt' in path or 'rules' in path
    return result


class SettingsStore:
    def __init__(self, path): self.path = Path(path)

    def _read(self):
        from ruamel.yaml import YAML
        text = self.path.read_text(encoding='utf-8')
        raw = YAML().load(text)
        if not isinstance(raw, dict): raise ValueError('Configuration must be a YAML mapping')
        return text, raw

    def snapshot(self):
        text, raw = self._read()
        values = self._values(raw)
        return {'revision': revision(text), 'values': values, 'fields': [field(k, v) for k, v in values.items()
            if k not in OBSOLETE and not k.startswith('presets.default.model_params.')
            and not (values.get('runtime.provider') == 'llama_cpp' and k in {'runtime.model','runtime.base_url','runtime.api_key','runtime.api_mode','runtime.reuse_response_ids','runtime.tokenizer_model'})],
            'path': str(self.path), 'restart_required': True}

    def _values(self, raw):
        # Include effective defaults, not just settings already written in YAML.
        candidate = load_config(self.path)
        values = {}
        for group in ('runtime', 'memory', 'emotion', 'tools'):
            for key, value in asdict(getattr(candidate, group)).items():
                if group == 'memory' and key in {'default_memories', 'history_file'}: continue
                values[f'{group}.{key}'] = value
        values.update(dict(flatten(raw)))
        for key, value in {'model':'character_files/Mita.vrm', 'format':'auto', 'enabled':True}.items():
            values.setdefault('avatar.' + key, value)
        from ..audio.speech_chunks import DEFAULTS
        for key, value in DEFAULTS.items():
            values.setdefault(f'speech.{key}', deepcopy(value))
        from ..audio.wake_feedback import DEFAULTS as WAKE_DEFAULTS
        for key, value in WAKE_DEFAULTS.items():
            values.setdefault(f'wake_feedback.{key}', deepcopy(value))
        from ..animation.library import DEFAULTS as ANIMATION_DEFAULTS
        for key, value in ANIMATION_DEFAULTS.items():
            values.setdefault(f'animation.{key}', deepcopy(value))
        from ..runtime.initiative import DEFAULTS as INITIATIVE_DEFAULTS
        for key, value in INITIATIVE_DEFAULTS.items():
            values.setdefault(f'initiative.{key}', deepcopy(value))
        values['initiative.context_window_tokens'] = candidate.runtime.initiative_n_ctx
        values['initiative.max_output_tokens'] = candidate.runtime.initiative_max_output_tokens
        if candidate.runtime.kv_pool_auto: values['runtime.kv_pool_tokens'] = candidate.runtime.kv_pool_tokens
        for key, value in {'asr_model':'distil-small.en','asr_device':'cuda','asr_compute_type':'int8_float16','live_transcript_interval_seconds':2.0}.items():
            values.setdefault('voice.' + key, value)
        return json.loads(json.dumps(values, default=str))

    def prepare(self, changes):
        text, raw = self._read()
        values = self._values(raw); errors = {}
        if not isinstance(changes, dict) or len(changes) > 300: raise ValueError('Invalid settings patch')
        for path, value in changes.items():
            if path not in values: errors[path] = 'Unknown setting'; continue
            spec = field(path, values[path])
            if spec['readonly']: errors[path] = 'Legacy setting is not used by the current runtime'; continue
            if value is None and spec['nullable']: pass
            elif spec['kind'] == 'boolean' and type(value) is not bool: errors[path] = 'Choose on or off'
            elif spec['kind'] == 'number':
                if type(value) not in (int, float) or not math.isfinite(value): errors[path] = 'Enter a finite number'
                elif spec['integer'] and type(value) is not int: errors[path] = 'Enter a whole number'
                elif 'min' in spec and value < spec['min'] or 'max' in spec and value > spec['max']: errors[path] = f"Allowed range: {spec.get('min', '−∞')} to {spec.get('max', '∞')}"
            elif spec['kind'] == 'text' and not isinstance(value, str): errors[path] = 'Enter text'
            elif spec['kind'] == 'json' and not isinstance(value, type(values[path])) and values[path] is not None: errors[path] = 'Keep the same JSON structure type'
            if spec.get('options') and value not in spec['options']: errors[path] = 'Choose a supported option'
            if isinstance(value, str) and len(value) > 100000: errors[path] = 'Value is too long'
            if path not in errors:
                node = raw
                keys = path.split('.')
                for key in keys[:-1]:
                    if key not in node: node[key] = {}
                    node = node[key]
                node[keys[-1]] = deepcopy(value)
        if errors: return text, None, errors
        if any(key in changes for key in ('avatar.model', 'avatar.format')):
            from ..desktop.avatar_models import AvatarModels
            try: AvatarModels(self.path.parent).validate(raw['avatar'].get('model', values['avatar.model']), raw['avatar'].get('format', values['avatar.format']))
            except (ValueError, OSError) as exc:
                errors['avatar.model'] = str(exc)
                return text, None, errors
        from ruamel.yaml import YAML
        yaml = YAML(); yaml.preserve_quotes = True
        stream = io.StringIO(); yaml.dump(raw, stream)
        output = stream.getvalue()
        fd, temporary = tempfile.mkstemp(suffix='.yaml', prefix='.settings-validation-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as file: file.write(output)
            candidate = load_config(temporary)
            voice = candidate.raw.get('voice', {})
            if voice.get('mode', 'wake_word') not in {'wake_word', 'manual', 'continuous'}: errors['voice.mode'] = 'Invalid activation mode'
            if len(str(voice.get('wake_word', 'Riko')).split()) != 1: errors['voice.wake_word'] = 'Use one short wake name'
            for path, value in dict(flatten(raw)).items():
                if path.endswith(('_url', '.url')) and isinstance(value, str) and not value.startswith(('http://', 'https://')):
                    errors[path] = 'Use an http:// or https:// URL'
            defaults = candidate.raw.get('presets', {}).get('default', {}).get('memories', [])
            if not isinstance(defaults, list) or any(not isinstance(m, dict) or not isinstance(m.get('text'), str) for m in defaults):
                errors['presets.default.memories'] = 'Default memories must be a list of objects with text'
            rt = candidate.runtime
            if 'initiative.context_window_tokens' in changes: rt.initiative_n_ctx = changes['initiative.context_window_tokens']
            from ..inference.background_budget import validate_budget
            validate_budget(rt.initiative_n_ctx, changes.get('initiative.max_output_tokens', rt.initiative_max_output_tokens), 'initiative')
            from ..inference.kv_budget import pool_capacity
            if rt.kv_pool_auto:
                raw.setdefault('runtime', {})['kv_pool_auto'] = True
                raw['runtime']['kv_pool_tokens'] = pool_capacity(rt)
                stream = io.StringIO(); yaml.dump(raw, stream); output = stream.getvalue()
            if rt.provider == 'llama_cpp':
                pool_capacity(rt)
                if rt.n_ctx <= 0: errors['runtime.n_ctx'] = 'Managed server requires a positive context'
                budget = candidate.memory.context_window_tokens + rt.max_output_tokens
                if budget > rt.n_ctx: errors['runtime.n_ctx'] = f'Context must fit conversation budget + response ({budget} tokens)'
        except (ValueError, TypeError, KeyError) as exc: errors['__all__'] = str(exc)
        finally: Path(temporary).unlink(missing_ok=True)
        return text, output, errors

    def validate(self, changes):
        with LOCK:
            _, _, errors = self.prepare(changes)
            return {'valid': not errors, 'errors': errors}

    def save(self, changes, expected_revision):
        with LOCK:
            text, output, errors = self.prepare(changes)
            if revision(text) != expected_revision or revision(self.path.read_text(encoding='utf-8')) != expected_revision:
                raise SettingsConflict('Configuration changed outside this editor. Reload before saving.')
            if errors: return {'saved': False, 'valid': False, 'errors': errors}
            budget_changes = {key.split('.')[-1]: value for key, value in changes.items()
                if key in {'initiative.context_window_tokens', 'initiative.max_output_tokens'}}
            preferences = self.path.parent / 'persistent_memories' / 'initiative_settings.json'
            saved_preferences = None
            if budget_changes and preferences.exists():
                saved_preferences = json.loads(preferences.read_text(encoding='utf-8'))
                if not isinstance(saved_preferences, dict): raise ValueError('Live initiative preferences must be an object')
                saved_preferences.update(budget_changes)
            # Keep a previous config, including its comments and unknown settings.
            backup = self.path.with_suffix(self.path.suffix + '.previous')
            backup.write_text(text, encoding='utf-8')
            fd, temporary = tempfile.mkstemp(prefix='.settings-save-', dir=self.path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    stream.write(output); stream.flush(); os.fsync(stream.fileno())
                if revision(self.path.read_text(encoding='utf-8')) != expected_revision:
                    raise SettingsConflict('Configuration changed while saving. Reload before saving.')
                os.replace(temporary, self.path)
                # Live initiative preferences share these budgets. Preserve all
                # other preferences while saving model/runtime budgets together.
                if saved_preferences is not None:
                    pending = preferences.with_suffix('.settings.tmp')
                    pending.write_text(json.dumps(saved_preferences, indent=2), encoding='utf-8')
                    pending.replace(preferences)
            finally: Path(temporary).unlink(missing_ok=True)
            return {**self.snapshot(), 'saved': True, 'restart_required': any(not key.startswith('avatar.') and key != 'runtime.pause_background_on_live' for key in changes)}

    def check_path(self, value):
        if not isinstance(value, str) or len(value) > 4096: raise ValueError('Invalid path')
        path = Path(value).expanduser()
        if not path.is_absolute(): path = self.path.parent / path
        executable = shutil.which(value)
        return {'resolved': executable or str(path.resolve()), 'exists': path.exists(), 'directory': path.is_dir(), 'executable': executable}
