"""The packaged first run's configuration, and the command line that checks a configuration file. Electron's setup page
(electron/src/first_setup.jsx) collects the choices; electron/release.cjs saveSetup runs `riko-backend --setup-config <data
folder>` with them as JSON on stdin and writes the YAML this answers. So the backend, which owns the settings schema, also
authors the file it will load: the choices over what a packaged build starts with, written by the YAML library that reads
it and checked as Settings checks an edit. Electron keeps the data folder's rules and never overwrites an existing file.
`--validate-config [path]` runs the same checks on any configuration file."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from ..kernel.code_paths import CodePaths
from .native_backends import backends_for, bundled_library, manifest, release_backends
from .paths import config_file
from .settings_store import SettingsStore

PROMPT = 'You are a helpful local companion.'
TTS = 'http://127.0.0.1:9880/tts'


def whole(value):
    """A whole number the form holds as a number, or as the text typed into it; None otherwise."""
    if isinstance(value, str):
        try: value = float(value)
        except ValueError: return None
    if type(value) in (int, float) and math.isfinite(value) and float(value).is_integer(): return int(value)
    return None


def setup_config(choices, platform=None, bundle=None):
    """The configuration for the setup page's choices, as a mapping; ValueError names the first choice it cannot use. bundle
    is the installed app's resources (default RIKO_BUNDLE_ROOT), where bundled:<backend> must exist for this platform."""
    if not isinstance(choices, dict): raise ValueError('Setup choices must be an object')
    platform = platform or sys.platform
    def text(key): return choices[key] if isinstance(choices.get(key), str) else ''
    executable = text('sovitsExecutable')
    if choices.get('sovitsAuto') and not (os.path.isabs(executable) and os.path.exists(executable)): raise ValueError('Choose an existing absolute GPT-SoVITS executable')
    shipped, backend = backends_for(platform), choices.get('backend')
    if backend not in shipped:
        labels = {item['id']: item['label'] for item in manifest()['backends']}
        # The default bundles: an optional one (hip) is offered only where installed, and its missing bundle is reported below.
        raise ValueError('Choose ' + ' or '.join(labels[name] for name in release_backends(platform)) if shipped else f'No native backend ships for {platform}')
    library = bundled_library(backend, bundle if bundle is not None else CodePaths.current().bundle, platform)
    if not library.exists(): raise ValueError(f'Packaged native backend is missing: {library}')
    context, output, threads = (whole(choices.get(key)) for key in ('context', 'output', 'threads'))
    if context is None or not 2048 <= context <= 131072 or output is None or not 64 <= output < context: raise ValueError('Invalid context/output budget')
    if threads is None or not 1 <= threads <= 1024: raise ValueError('Invalid CPU thread count')
    model, repo, filename = text('modelPath'), text('repo'), text('filename')
    if not model and (not repo or not filename or not filename.endswith('.gguf') or '..' in filename or filename.startswith('/')):
        raise ValueError('Choose a local GGUF or exact Hugging Face repository/file')
    if model and (not os.path.isabs(model) or not model.lower().endswith('.gguf') or not os.path.exists(model)): raise ValueError('Local GGUF does not exist')
    # The name may be several words, but the wake detector enrolls one: the wake name is the name's first word.
    name, julia = text('name').strip() or 'Riko', bool(choices.get('julia'))
    memories = [{'text': line, 'memory_type': 'factual', 'importance': .8} for line in text('memories').split('\n') if line.strip()]
    # What a packaged build starts with beside the choices: the bundled library, two slots, f16 caches, the conversation
    # budget leaving room for the reply, speech recognition on the CPU, the probe off, tools that ask first, no
    # initiative, and Settings left open should a section fail to load. The schema's defaults fill in everything else.
    return {'runtime': {'provider': 'llama_cpp', 'native_library': f'bundled:{backend}', 'model_path': model or None, 'hf_repo_id': repo or None,
            'hf_filename': filename or None, 'hf_revision': text('revision') or 'main', 'n_ctx': context, 'max_output_tokens': output, 'n_threads': threads,
            'n_gpu_layers': 0 if choices.get('cpuOnly') else -1, 'parallel_slots': 2, 'flash_attn': 'auto', 'type_k': 'f16', 'type_v': 'f16', 'warmup': False},
        'presets': {'default': {'name': name, 'system_prompt': text('prompt') or PROMPT}},
        'memory': {'context_window_tokens': context - output, 'default_memories': memories, 'embeddings_enabled': bool(choices.get('embeddings')),
            'system1_enabled': julia, 'reflection_enabled': bool(choices.get('reflection'))},
        'emotion': {'enabled': julia, 'device': 'cpu', 'probe': {'enabled': False}},
        'voice': {'wake_word': name.split()[0], 'asr_device': 'cpu', 'asr_compute_type': 'int8'},
        'tools': {'require_approval': True}, 'initiative': {'enabled': False}, 'desktop': {'setup_on_startup_error': True},
        'sovits_ping_config': {'auto_start': bool(choices.get('sovitsAuto')), 'executable': executable or None, 'arguments': [], 'url': text('sovitsUrl') or TTS,
            'ref_audio_path': text('referenceAudio'), 'prompt_text': text('referenceText'), 'text_lang': 'en', 'prompt_lang': 'en', 'sample_rate': 32000}}


def render(config):
    """YAML as the backend reads it: PyYAML quotes what it would not read back as text (a companion named Yes, a memory 42)."""
    import yaml
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=4096)


def setup_text(choices, directory):
    """The first run's YAML for this OS and bundle (RIKO_BUNDLE_ROOT, as load_config resolves bundled:<backend>), checked
    where it will live (relative paths resolve there) as if every value were edited in Settings; ValueError lists what
    fails. Nothing is left in directory."""
    text = render(setup_config(choices))
    descriptor, temporary = tempfile.mkstemp(prefix='.setup-', suffix='.yaml', dir=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream: stream.write(text)
        result = SettingsStore(temporary).validate_file()
    finally: Path(temporary).unlink(missing_ok=True)
    if not result['valid']: raise ValueError('; '.join(message if path == '__all__' else f'{path}: {message}' for path, message in result['errors'].items()))
    return text


def command(argv, stdin=None, stdout=None):
    """`--setup-config <data folder>` (the choices as JSON on stdin) answers {"config": yaml text}; `--validate-config [path]`
    (default: RIKO_CONFIG, else ./character_config.yaml) answers SettingsStore.validate_file's {valid, errors, warnings?}.
    Either answers {"error": message} when it cannot. One line of JSON, ASCII whatever the console's code page, with exit
    status 0 only for a configuration or a valid file. It needs no data root, port or model, and leaves no file behind."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    flag = '--setup-config' if '--setup-config' in argv else '--validate-config'
    arguments = argv[argv.index(flag) + 1:]
    def answer(value, ok): stdout.write(json.dumps(value) + '\n'); stdout.flush(); return 0 if ok else 1
    try:
        if flag == '--setup-config':
            if not arguments: raise ValueError('--setup-config needs the data folder')
            source = stdin.buffer.read().decode('utf-8') if hasattr(stdin, 'buffer') else stdin.read()
            return answer({'config': setup_text(json.loads(source), arguments[0])}, True)
        result = SettingsStore(Path(arguments[0]) if arguments else config_file()).validate_file()
        return answer(result, result['valid'])
    except Exception as exc: return answer({'error': str(exc) or type(exc).__name__}, False)  # a YAML or JSON syntax error too
