"""Round-trip YAML settings with typed forms, validation and conflict-safe saves. What each setting is (its kind, range,
help, group, restart scope and live hook) comes from the settings schema (configuration/schema.py over kernel/schema.py),
where each package registers its own sections."""
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading

from . import schema
from .config import AppConfig, load_config
from .paths import DataPaths
from .schema import UNKNOWN, field, flatten, secret  # noqa: F401  (callers and tests import field and UNKNOWN from here)
from ..kernel.safe_paths import local_path
from ..persistence.atomic import atomic_write

LOCK = threading.RLock()
# What Settings sends for a saved secret (schema.secret: API keys, tokens, passwords) instead of its value; a change equal to
# it means "keep the saved value" (kept()). The renderer never receives a secret, so a page that displays text cannot leak it.
SECRET_MASK = '\u2022' * 8
EXTERNAL = 'Set outside character_config.yaml (an environment variable or a top-level key); a value saved here overrides it.'


class SettingsConflict(ValueError): pass


def unknown_warnings(raw):
    """{path: UNKNOWN} for keys the strict sections do not declare (configuration/schema.py unknown_settings): reported,
    never an error, so a typo or an obsolete key never blocks saving."""
    return {path: UNKNOWN for path in schema.unknown_settings(raw)}


def revision(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()


def kept(changes):
    """changes without the secrets left as SECRET_MASK: an untouched masked field keeps its saved value."""
    if not isinstance(changes, dict): return changes
    return {path: value for path, value in changes.items() if not (value == SECRET_MASK and secret(path))}


def stored(raw, path):
    """The YAML's own value at a dotted path (None when absent), not the effective one."""
    node = raw
    for key in path.split('.'):
        if not isinstance(node, dict) or key not in node: return None
        node = node[key]
    return node


def mask(values, fields, raw):
    """Replace each secret in values (and say so on its field): SECRET_MASK for one the YAML holds; '' for one that comes
    from elsewhere (RIKO_API_KEY, a legacy top-level key), so an environment variable is never sent; the default as is. A
    key PyYAML reads as a number (api_key: 123456789) is a secret too."""
    specs = {item['path']: item for item in fields}
    for path, value in values.items():
        if not secret(path) or isinstance(value, bool) or not isinstance(value, (str, int, float)) or value == '': continue
        spec, saved = specs.get(path), stored(raw, path)
        if saved is not None and saved != '': values[path], state = SECRET_MASK, 'saved'
        elif value != schema.declared_default(path): values[path], state = '', 'external'
        else: continue
        if spec is not None:
            spec['secret_state'] = state
            if state == 'external': spec['help'] = ' '.join(filter(None, (spec.get('help'), EXTERNAL)))


def value_error(spec, value, current):
    """Why Settings refuses value for the field spec (field()) whose stored value is current; '' when it accepts it."""
    if spec['readonly']: return 'Legacy setting is not used by the current runtime'
    error = ''
    if value is None and spec['nullable']: pass
    elif spec['kind'] == 'boolean' and type(value) is not bool: error = 'Choose on or off'
    elif spec['kind'] == 'number':
        if type(value) not in (int, float) or not math.isfinite(value): error = 'Enter a finite number'
        elif spec['integer'] and type(value) is not int: error = 'Enter a whole number'
        elif 'min' in spec and value < spec['min'] or 'max' in spec and value > spec['max']: error = f"Allowed range: {spec.get('min', '−∞')} to {spec.get('max', '∞')}"
    elif spec['kind'] == 'text' and not isinstance(value, str): error = 'Enter text'
    elif spec['kind'] == 'json' and not isinstance(value, type(current)) and current is not None: error = 'Keep the same JSON structure type'
    if spec.get('options') and value not in spec['options']: error = 'Choose a supported option'
    if isinstance(value, str) and len(value) > 100000: error = 'Value is too long'
    return error


def beyond_values(candidate, written, draft, changes):
    """The checks after load_config accepted the candidate: each section's review (configuration/schema.py) and the URLs
    the YAML holds (written: {path: value})."""
    # Each section's checks beyond load_config: the wake word, an edited speech recognition pair or avatar, default
    # memories, the llama.cpp budget and split, the initiative budget.
    errors = schema.review(candidate, draft, changes)
    for path, value in written.items():
        if path.endswith(('_url', '.url')) and isinstance(value, str) and not value.startswith(('http://', 'https://')):
            errors[path] = 'Use an http:// or https:// URL'
    return errors


class SettingsStore:
    def __init__(self, path, running=None):
        """running: the AppConfig the backend started with, so snapshot() can say which saved settings await a restart."""
        self.path, self.running = Path(path), running

    def _read(self):
        from ruamel.yaml import YAML
        text = self.path.read_text(encoding='utf-8')
        # Keep quotes: ruamel writes YAML 1.2, and the backend's PyYAML reads an unquoted yes or 42 as a boolean or number.
        yaml = YAML(); yaml.preserve_quotes = True
        raw = yaml.load(text)
        if not isinstance(raw, dict): raise ValueError('Configuration must be a YAML mapping')
        return text, raw

    def snapshot(self):
        text, raw = self._read()
        candidate = load_config(self.path, recover=True)  # open even while a section is broken, so Settings can repair it
        values = schema.values(candidate, raw)
        fields = [field(k, v) for k, v in values.items() if schema.offered(k)]
        warnings = unknown_warnings(raw)
        for item in fields:
            if any(item['path'] == path or item['path'].startswith(path + '.') for path in warnings): item.update(unknown=True, help=UNKNOWN)
        mask(values, fields, raw)  # after the fields, built from the real values
        pending = self.pending(candidate)
        # restart_pending ({path: restart scope}) and rules (checks across settings, kernel/schema.py Rule) are additions;
        # restart_required used to be always true. The config's absolute path is no longer sent.
        result = {'revision': revision(text), 'values': values, 'fields': fields, 'restart_required': bool(pending),
            'restart_pending': pending, 'rules': schema.rules()}
        return {**result, 'warnings': warnings} if warnings else result

    def _values(self, raw): return schema.values(load_config(self.path, recover=True), raw)  # effective defaults, not just what the YAML holds

    def stored_values(self):
        """The effective values, secrets unmasked: for comparisons in the backend only, never for a response."""
        return self._values(self._read()[1])

    def pending(self, candidate):
        """{path: restart scope} for saved settings that differ from what the running backend loaded, both read as it
        reads them (PyYAML); empty without a running backend."""
        if not isinstance(self.running, AppConfig): return {}
        before, after = schema.values(self.running, self.running.raw), schema.values(candidate, candidate.raw)
        changed = (path for path in sorted({*before, *after}) if before.get(path) != after.get(path))
        return {path: scope for path in changed if (scope := schema.restart_scope(path)) != 'none'}

    def prepare(self, changes):
        text, raw = self._read()
        values = self._values(raw); errors = {}
        if not isinstance(changes, dict) or len(changes) > 300: raise ValueError('Invalid settings patch')
        changes = kept(changes)
        for path, value in changes.items():
            if path not in values: errors[path] = 'Unknown setting'; continue
            if secret(path) and isinstance(value, str) and SECRET_MASK in value: errors[path] = 'Type the whole new value, or clear the field'; continue
            error = value_error(field(path, values[path]), value, values[path])
            if error: errors[path] = error
            else:
                node = raw
                keys = path.split('.')
                for key in keys[:-1]:
                    if node.get(key) is None: node[key] = {}  # absent, or a section whose keys are all commented out
                    node = node[key]
                if schema.quoted(path) and isinstance(value, str):
                    from ruamel.yaml.scalarstring import DoubleQuotedScalarString
                    value = DoubleQuotedScalarString(value)
                node[keys[-1]] = deepcopy(value)
        if errors: return text, None, errors
        from ruamel.yaml import YAML
        yaml = YAML(); yaml.preserve_quotes = True
        stream = io.StringIO(); yaml.dump(raw, stream)
        output = stream.getvalue()
        fd, temporary = tempfile.mkstemp(suffix='.yaml', prefix='.settings-validation-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as file: file.write(output)
            candidate = load_config(temporary)  # every registered section's check, as the backend will run it
            errors.update(beyond_values(candidate, dict(flatten(raw)), {**values, **changes}, changes))
            writes = schema.derive(candidate)  # e.g. the automatic KV pool, saved with every change
            if writes:
                for path, value in writes.items():
                    node, (*parents, key) = raw, path.split('.')
                    for name in parents:
                        if node.get(name) is None: node[name] = {}
                        node = node[name]
                    node[key] = value
                stream = io.StringIO(); yaml.dump(raw, stream); output = stream.getvalue()
        except (ValueError, TypeError, KeyError) as exc: errors.setdefault('__all__', str(exc))
        finally: Path(temporary).unlink(missing_ok=True)
        return text, output, errors

    def validate(self, changes):
        with LOCK:
            _, _, errors = self.prepare(kept(changes))
            warnings = unknown_warnings(self._read()[1])
            return {'valid': not errors, 'errors': errors, **({'warnings': warnings} if warnings else {})}

    def validate_file(self):
        """validate() for the file as it stands, as if Settings had just set every value it offers: each section's check
        (load_config), each value's type, range and options, and the checks beyond them (a speech recognition pair this
        machine cannot run fails here, where loading would fall back). It judges the file as the backend reads it (PyYAML,
        YAML 1.1) and its effective values, not ruamel's round-trip view. The packaged first run checks its configuration
        with it, and so does `--validate-config` (configuration/first_run.py)."""
        import yaml
        with LOCK:
            read = yaml.safe_load(self.path.read_text(encoding='utf-8'))
            if not isinstance(read, dict): raise ValueError('Configuration must be a YAML mapping')
            errors, written = {}, dict(flatten(read))
            try:
                candidate = load_config(self.path)
                values = schema.values(candidate, read)
                specs = {path: field(path, values[path]) for path in written if path in values and schema.offered(path)}
                # A key the file's provider does not use (visible_when), perhaps left by another provider, is not judged.
                specs = {path: spec for path, spec in specs.items() if all(values.get(name) in allowed for name, allowed in spec.get('visible_when', {}).items())}
                changes = {path: values[path] for path, spec in specs.items() if not spec['readonly']}  # a legacy key is load_config's
                errors.update((path, error) for path, value in changes.items() if (error := value_error(specs[path], value, value)))
                errors.update(beyond_values(candidate, written, values, changes))
                schema.derive(candidate)  # a manual KV pool too small for the budgets
            except (ValueError, TypeError, KeyError) as exc: errors.setdefault('__all__', str(exc))
            warnings = unknown_warnings(read)
            return {'valid': not errors, 'errors': errors, **({'warnings': warnings} if warnings else {})}

    def save(self, changes, expected_revision):
        with LOCK:
            changes = kept(changes)
            text, output, errors = self.prepare(changes)
            if revision(text) != expected_revision or revision(self.path.read_text(encoding='utf-8')) != expected_revision:
                raise SettingsConflict('Configuration changed outside this editor. Reload before saving.')
            if errors: return {'saved': False, 'valid': False, 'errors': errors}
            budget_changes = {key.split('.')[-1]: value for key, value in changes.items()
                if key in {'initiative.context_window_tokens', 'initiative.max_output_tokens'}}
            preferences = DataPaths.build(self.path).initiative_settings  # beside the YAML being edited
            saved_preferences = None
            if budget_changes and preferences.exists():
                saved_preferences = json.loads(preferences.read_text(encoding='utf-8'))
                if not isinstance(saved_preferences, dict): raise ValueError('Live initiative preferences must be an object')
                saved_preferences.update(budget_changes)
            # Keep a previous config, including its comments and unknown settings: owner-only, like the config (it may hold keys).
            atomic_write(self.path.with_suffix(self.path.suffix + '.previous'), text, mode=0o600)
            fd, temporary = tempfile.mkstemp(prefix='.settings-save-', dir=self.path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    stream.write(output); stream.flush(); os.fsync(stream.fileno())
                if revision(self.path.read_text(encoding='utf-8')) != expected_revision:
                    raise SettingsConflict('Configuration changed while saving. Reload before saving.')
                os.replace(temporary, self.path)
                # Live initiative preferences share these budgets. Preserve all
                # other preferences while saving model/runtime budgets together.
                if saved_preferences is not None: atomic_write(preferences, json.dumps(saved_preferences, indent=2))
            finally: Path(temporary).unlink(missing_ok=True)
            # Whether this save needs a restart; snapshot()'s restart_pending also names earlier saves still waiting for one.
            return {**self.snapshot(), 'saved': True, 'restart_required': any(schema.restart_scope(key) != 'none' for key in changes)}

    def check_path(self, key, value):
        """Settings' path check for the file or folder setting key: whether value exists, as an absolute path (never
        symlink-resolved), or the PATH match of a bare command for an executable. Any other setting, and a network or
        device path (kernel/safe_paths.py: checked as text, so it is never contacted), is refused."""
        if not isinstance(value, str) or not value or len(value) > 4096: raise ValueError('Invalid path')
        values = self.stored_values()
        if not isinstance(key, str) or key not in values or not schema.offered(key) or not field(key, values[key])['file']:
            raise ValueError('Path checks are only for file and folder settings')
        if key == 'runtime.native_library' and value.startswith('bundled:'):  # a library inside the installed app, not a path
            from ..kernel.code_paths import CodePaths
            from .native_backends import bundled_library
            path = bundled_library(value.split(':', 1)[1], CodePaths.current().bundle)
            return {'resolved': str(path), 'exists': path.is_file(), 'directory': False, 'executable': False}
        if key.endswith('executable') and not any(separator in value for separator in '/\\'):
            command = shutil.which(value)
            if command: return {'resolved': command, 'exists': True, 'directory': False, 'executable': True}
        path = local_path(value, self.path.parent)
        return {'resolved': str(path), 'exists': path.exists(), 'directory': path.is_dir(), 'executable': False}
