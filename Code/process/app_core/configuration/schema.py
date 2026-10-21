"""Settings from the schema registry (kernel/schema.py): load_config's section checks and unknown keys, and Settings'
values, fields, rules, restart scopes and live-apply hooks. Configuration declares its own sections here (the top level,
memory, emotion, tools, tasks, logging, desktop, avatar and presets) and imports nothing from the feature packages, which
declare theirs; app_core/__init__.py registers them all before the registry is first read."""
from copy import deepcopy
from dataclasses import asdict
from functools import cache
import json
import math
import re

from ..kernel import schema
from ..kernel.background_budget import validate_budget
from ..kernel.model_pins import JULIA_REVISION, commit
from ..kernel.schema import MISSING, Section, Setting, dataclass_defaults, register
from ..kernel.validation import section, unknown

UNKNOWN = 'This version does not read this setting, so it has no effect. Check its spelling, or remove it from character_config.yaml.'
# Configuration's typed sections, whose effective values Settings lists first, less what it edits elsewhere: the
# background budgets (initiative.*, memory.reflection_*), the memory seed list and history file, and emotion.probe.
TYPED = (('runtime', {'initiative_n_ctx', 'initiative_max_output_tokens', 'reflection_n_ctx'}),
    ('memory', {'default_memories', 'history_file'}), ('emotion', {'probe'}), ('tools', set()))
DEVICES = ('cpu', 'auto', 'cuda', 'cuda:0')  # kernel/torch_device.py; Julia maps mps to the CPU, so emotion leaves it out
BLANK = Setting('')


def check_memory(memory):
    memory = section('memory', memory)
    validate_budget(memory.get('reflection_context_window_tokens', 4096), memory.get('reflection_max_output_tokens', 1024), 'reflection')
    commit('memory.system1_revision', memory.get('system1_revision', JULIA_REVISION))
    return memory


def check_emotion(emotion):
    emotion = section('emotion', emotion)
    commit('emotion.revision', emotion.get('revision', JULIA_REVISION))
    return emotion


def check_tools(tools):
    tools = section('tools', tools)
    if type(tools.get('best_fit_inputs', True)) is not bool: raise ValueError('tools.best_fit_inputs must be boolean')
    for key, default in (('best_fit_timeout_seconds', .4), ('best_fit_min_confidence', .85)):
        value, (low, high) = tools.get(key, default), schema.REGISTRY.setting(f'tools.{key}').range
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high: raise ValueError(f'tools.{key} must be between {low} and {high}')
    disabled = tools.get('disabled') or []  # tool names as the model calls them (tools/tool.py NAME)
    if not isinstance(disabled, list) or not all(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) for name in disabled):
        raise ValueError('tools.disabled must be a list of tool names (letters, digits, _ or -)')
    return tools


def review_memories(candidate, draft, changes):
    defaults = candidate.raw.get('presets', {}).get('default', {}).get('memories', [])
    if not isinstance(defaults, list) or any(not isinstance(m, dict) or not isinstance(m.get('text'), str) for m in defaults):
        return {'presets.default.memories': 'Default memories must be a list of objects with text'}
    return {}


def without_probe(config, values): values.pop('emotion.probe', None)  # emotion.probe.* are listed one by one (emotion/probe.py)


MODELS = dict(group='models')
TOKENS, BACKGROUND, EMBEDDING = (dict(MODELS, section=name) for name in ('Token budgets', 'Background models', 'Embedding model'))
RESTART_PYTHON = ' Restart Python to apply.'
JULIA_REVISION_HELP = ('The Hugging Face commit of {model} to download and run. Julia-1 ships Python code that runs inside Riko, so this is '
    'a full 40-character commit hash, never a branch: a new upstream commit changes nothing until you change this. The default is '
    'the commit this version checks file by file; another commit\'s code runs unverified.{extra} Restart Python.')
register(
    # The top level: what no section declares is character text, edited until Python restarts.
    Section('', group='character', title='Settings', restart='python', advanced=False, settings=(
        Setting('your_name', obsolete=True),
        *(Setting(key, hidden=True) for key in ('model', 'base_url', 'api_key', 'tokenizer_model')))),  # read only when runtime.* lacks them
    Section('memory', group='memory', title='Settings', check=check_memory, settings=(
        Setting('context_window_tokens', range=(1, 1048576), **TOKENS), Setting('token_budget', **TOKENS), Setting('max_results', **TOKENS),
        Setting('reflection_enabled', **BACKGROUND), Setting('reflection_min_importance', range=(0, 1), section='Reflection'),
        Setting('reflection_max_output_tokens', range=(1, 1048575), **TOKENS,
            help='Reflection output budget including reasoning. Invalid/empty/truncated output errors that job without a format-repair retry.'),
        Setting('reflection_context_window_tokens', range=(256, 1048576), **TOKENS,
            help='Total reflection context including evidence/instructions/output. Optional related evidence and formation history are token-trimmed; required focal content stays.'),
        Setting('system1_enabled', **BACKGROUND), Setting('system1_model_id', **BACKGROUND), Setting('system1_cache_dir', nullable=True, **BACKGROUND),
        Setting('system1_revision', kind='text', quoted=True, label='Julia-1 revision', **BACKGROUND,
            help=JULIA_REVISION_HELP.format(model='memory.system1_model_id', extra='')),
        Setting('system1_max_length', **TOKENS),
        Setting('device', options=(*DEVICES, 'mps'), label='Memory model device', **BACKGROUND,
            help='Device for the memory classifier (Julia-1) and the embedding model. CPU by default. auto picks CUDA, then Apple MPS, then CPU. The classifier stays on CPU under MPS; embeddings can use it. GPU use competes with the main model for memory, and a GPU that fails to load falls back to CPU. Restart Python.'),
        Setting('embeddings_enabled', **EMBEDDING), Setting('embedding_model', **EMBEDDING), Setting('embedding_dimension', **EMBEDDING),
        Setting('minimum_importance', range=(0, 1)), Setting('index_file', obsolete=True))),  # FAISS is gone; the index is rebuilt in memory
    Section('emotion', group='models', title='Emotion model', check=check_emotion, effective=without_probe, settings=(
        Setting('device', options=DEVICES, label='Julia-1 compute device',
            help='Julia-1 teacher device: CPU by default, keeping GPU memory for the main model. CUDA is opt-in and needs a GPU build of torch (ROCm builds also report cuda); auto uses it when available. Julia does not support Apple MPS yet, so Macs use CPU. A GPU that fails to load falls back to CPU. The latent probe always runs on CPU. Restart Python.'),
        Setting('model_path', nullable=True), Setting('cache_dir', nullable=True),
        Setting('revision', kind='text', quoted=True, label='Julia-1 revision',
            help=JULIA_REVISION_HELP.format(model='emotion.model_id', extra=' Ignored when emotion.model_path exists.')),
        Setting('temperature', hidden=True), Setting('pause_during_inference', obsolete=True))),
    Section('tools', group='tools', title='Settings', check=check_tools, settings=(
        Setting('require_approval', help='Default for new/unconfigured tools. Per-tool live permissions override this default; approval bubbles expire after two minutes without executing.'),
        Setting('best_fit_inputs', help='Correct finite-choice desktop inputs before approval. Reuses the enabled, already-loaded CPU Julia model; safe spelling normalization is the fallback. Never repairs file permissions, numeric values or arbitrary paths.'),
        Setting('timeout_seconds', range=(.1, 3600)), Setting('max_iterations', range=(1, 100)),
        Setting('best_fit_timeout_seconds', range=(.05, 2)), Setting('best_fit_min_confidence', range=(.5, 1)),
        Setting('disabled', [], label='Disabled tools',
            help='JSON list of tool names Riko never sees or calls, e.g. ["pdf_extractor", "todo_list"]. Each tool\'s description is part of every prompt, so fewer tools mean faster replies. Restart Python.'))),
    Section('tasks', group='tools', title='Settings'),
    Section('logging', group='performance', title='Debug logging', settings=(
        Setting('file_enabled', True, label='Write debug log file',
            help='Write rotating diagnostics to logs/debug.log. Restart Python to apply. Prompts and messages are not collected by inference timing; review logs before sharing.'),
        Setting('level', 'INFO', options=('DEBUG', 'INFO', 'WARNING', 'ERROR'), label='Log detail level',
            help='DEBUG includes detailed application diagnostics; INFO includes inference timings; WARNING and ERROR restrict output.' + RESTART_PYTHON),
        Setting('inference_timings', True, label='Log inference performance timings',
            help='Record provider duration, first-token latency and output rate without recording prompts or responses.' + RESTART_PYTHON),
        Setting('max_mb', 5, range=(1, 100), label='Log file size (MiB)',
            help='Start a new debug.log when it reaches this size in MiB. The previous file becomes a numbered backup. Save and restart Python.'),
        Setting('backups', 3, range=(1, 10), label='Rotated log backups',
            help='Keep this many rotated files plus debug.log. Oldest backups are replaced; approximate maximum disk use is (backups + 1) times the file size. Save and restart Python.'))),
    # Electron reads desktop.debug, desktop.shortcuts and presets.default.name only at launch (electron/main.cjs).
    Section('desktop', group='appearance', title='Settings', settings=(
        Setting('debug', restart='electron'),
        Setting('shortcuts', restart='electron', help='Shortcut configuration is saved; some legacy shortcuts are not implemented.'),
        Setting('shortcuts.effects', obsolete=True, readonly=True, help='Legacy effect editor shortcut; currently not registered.'))),
    Section('desktop.shortcuts', restart='electron'),
    Section('avatar', group='appearance', title='Avatar model', order=-1, restart='none', live='avatar', settings=(  # above Animation engine
        Setting('model', 'character_files/Mita.vrm', file=True, help='Choose or import a VRM in Desktop & avatar. Saved avatar changes apply immediately.'),
        Setting('format', 'auto', options=('auto', 'vrm0', 'vrm1'), help='Auto detects VRM 0.x/1.0. Explicit formats must match the model; this does not convert it.'),
        Setting('enabled', True), Setting('camera.fov', range=(10, 80), help='Perspective field of view; automatic fitting still keeps the whole model visible.'),
        Setting('camera.distance', obsolete=True, readonly=True, help='Legacy fixed distance; current renderer frames the whole model automatically.'),
        Setting('view', obsolete=True, options=('full_body',), help='Current renderer supports full-body framing only.'),
        Setting('expression_engine', obsolete=True, readonly=True, help='Native VRM expressions are driven by Julia; retained for compatibility.'))),
    Section('presets', review=review_memories, settings=(Setting('default.name', restart='electron'),)),
    Section('presets.default.model_params', group='models', title='Preset fallbacks', advanced=True, hidden=True, settings=(  # read when runtime.* lacks them
        Setting('context_window_token_limit', range=(1, 1048576)), Setting('max_output_tokens', range=(1, 1048576)))))


def sections(): return schema.REGISTRY.sections()


def setting(path): return schema.REGISTRY.setting(path) or BLANK


def resolve(path, name): return schema.REGISTRY.resolve(path, name, title=name == 'section')


def at(raw, path):
    """The YAML value at a section's path ({} while it is absent)."""
    value = raw
    for key in path.split('.') if path else ():
        if not isinstance(value, dict) or key not in value: return {}
        value = value[key]
    return value


def checked(raw, errors=None):
    """{path: value} from every registered section's check; voice, speech and sovits_ping_config give AppConfig's. With
    errors (a list), a section whose check fails takes its defaults and its message is appended, instead of raising."""
    result = {}
    for item in sections():
        if not item.check: continue
        try: result[item.path] = item.check(at(raw, item.path))
        except ValueError as exc:
            if errors is None: raise
            errors.append(str(exc))
            result[item.path] = item.check({})
    return result


def configure(config, raw):
    """The checks across sections that need the assembled AppConfig: llama.cpp's runtime and pool, the probe's needs."""
    for item in sections():
        if item.configure: item.configure(config, raw)


def unknown_settings(raw):
    """section.key for each key a strict section does not declare, in the order the YAML lists them, each logged once: a
    typo or an obsolete key keeps its default and is reported here and in Settings, never a startup failure."""
    strict, known = {item.path for item in sections() if item.strict}, {}
    for item in sections():
        known.setdefault(item.path, set()).update(item.known, (spec.key.split('.')[0] for spec in item.settings if not spec.obsolete))
    found = []
    def walk(mapping, prefix):
        if prefix in strict: found.extend(unknown(prefix, mapping, known[prefix]))
        for key, value in mapping.items():
            if isinstance(value, dict): walk(value, f'{prefix}.{key}' if prefix else str(key))
    if isinstance(raw, dict): walk(raw, '')
    return tuple(found)


def flatten(value, prefix=''):
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(item, dict) and item: yield from flatten(item, path)
        else: yield path, item


def values(config, raw):
    """Settings' values, as JSON: configuration's typed sections as loaded, the YAML's own, each registered setting's
    default the YAML lacks, then the sections' adjustments (the budgets and pool in force)."""
    found = {f'{name}.{key}': value for name, skip in TYPED for key, value in asdict(getattr(config, name)).items() if key not in skip}
    found.update(flatten(raw))
    for item in sections():
        for spec in item.settings:
            if spec.listed and spec.default is not MISSING:
                found.setdefault(item.path_of(spec.key), spec.default(config) if callable(spec.default) else deepcopy(spec.default))
    for item in sections():
        if item.effective: item.effective(config, found)
    return json.loads(json.dumps(found, default=str))


@cache
def typed_defaults():
    from .config import EmotionConfig, MemoryConfig, RuntimeConfig, ToolConfig
    return {f'{name}.{key}': value for (name, _), cls in zip(TYPED, (RuntimeConfig, MemoryConfig, EmotionConfig, ToolConfig))
        for key, value in dataclass_defaults(cls).items()}


def declared_default(path):
    spec = setting(path)
    return spec.default if spec.default is not MISSING and not callable(spec.default) else typed_defaults().get(path, MISSING)


def restart_scope(path): return resolve(path, 'restart') or 'python'


def quoted(path): return setting(path).quoted


SECRET = re.compile(r'(?i)(^|_)(api_key|token|password|passphrase|secret)$')


def secret(path):
    """Whether a setting holds a credential, by its last name (api_key, *_token, password, ...; not *_tokens budgets or
    tokenizer_model): Settings masks its value and never sends it back (configuration/settings_store.py)."""
    return bool(SECRET.search(str(path).rsplit('.', 1)[-1]))


def field(path, value):
    """The form for one value: what its Setting declares, else inferred from the value and the sections around it."""
    spec, default = setting(path), declared_default(path)
    # A credential is text even when PyYAML reads it as a number (api_key: 123456789), so Settings masks it (secret()).
    kind = 'text' if spec.quoted or not spec.kind and secret(path) and type(value) in (int, float) else spec.kind or (
        'boolean' if type(value) is bool else 'number' if type(value) in (int, float)
        else 'json' if isinstance(value, (list, dict)) else 'text')
    # A numeric default says whether it is a whole number: YAML's 1 for a temperature no longer makes 0.7 an error.
    # A whole-number default makes a field integer unless the YAML holds a fraction the backend accepts (initiative seconds).
    integer = spec.integer if spec.integer is not None else type(default) is int and type(value) is not float if type(default) in (int, float) else type(value) is int
    scope = restart_scope(path)
    result = dict(path=path, label=spec.label or path.rsplit('.', 1)[-1].replace('_', ' ').capitalize(), group=resolve(path, 'group') or 'character',
        kind=kind, nullable=value is None or spec.nullable, integer=integer, help=spec.help, restart=scope != 'none')
    result['section'] = resolve(path, 'section') or 'Settings'
    result['advanced'] = bool(resolve(path, 'advanced'))
    result['readonly'] = spec.readonly
    if spec.options: result['options'] = list(spec.options)
    if spec.range: result['min'], result['max'] = spec.range
    if kind == 'number' and 'min' not in result and path.endswith(('_seconds', '_tokens', '_length', '_size', '_dimension', '_results')): result['min'] = 0
    if 'importance' in path or 'temperature' in path:
        result.setdefault('min', 0); result.setdefault('max', 2 if 'temperature' in path else 1)
    result['secret'] = secret(path) and kind == 'text'
    result['file'] = kind == 'text' and (bool(spec.file) or path.endswith(('_file', '_path', '_directory', '_dir')))
    result['multiline'] = kind == 'json' or 'prompt' in path or 'rules' in path
    result['restart_scope'] = scope  # RESTART in kernel/schema.py
    if spec.visible_when: result['visible_when'] = {name: list(allowed) for name, allowed in spec.visible_when}
    return result


def visible(spec, values): return all(values.get(name) in allowed for name, allowed in spec.visible_when)


def offered(path):
    """Whether Settings offers a value as a field: not hidden or obsolete. A key of another provider is offered too, so the
    renderer can show it as soon as the draft switches provider (its visible_when)."""
    spec = setting(path)
    return not (spec.obsolete or resolve(path, 'hidden'))


def rules(): return [rule.payload() for item in sections() for rule in item.rules]


def live_hooks(changes):
    """{hook: [paths]} for the changed settings that apply on save (Setting.live, or their section's)."""
    hooks = {}
    for path in changes:
        name = resolve(path, 'live')
        if name: hooks.setdefault(name, []).append(path)
    return hooks


def review(candidate, draft, changes):
    """Settings' checks beyond load_config, from every section; one that raises is reported, and the others still run."""
    errors = {}
    for item in sections():
        if not item.review: continue
        try: errors.update(item.review(candidate, draft, changes))
        except (ValueError, TypeError, KeyError) as exc: errors.setdefault('__all__', str(exc))
    return errors


def derive(candidate):
    """{path: value} that Settings writes with every save (the automatic KV pool); raises when one cannot be derived."""
    writes = {}
    for item in sections():
        if item.derive: writes.update(item.derive(candidate))
    return writes
