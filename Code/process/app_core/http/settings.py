"""/api/settings: the YAML through SettingsStore, the confirmation that security-sensitive changes need, and the hooks that
apply a saved setting without a restart."""
import hashlib
import re
from copy import deepcopy

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from ..configuration.config import load_config
from .backend import Services

router = APIRouter()

# Settings that load code, start programs, choose models or network endpoints, move user data, widen
# file access, or decide what runs unattended. Classified by name, so settings added later with these
# names are covered; any other value that is a URL or an absolute path is treated the same way.
SENSITIVE_SETTING_NAMES = {'native_library', 'mcp_config', 'executable', 'arguments', 'auto_start', 'api_key', 'url',
    'provider', 'model_id', 'embedding_model', 'asr_model', 'require_approval', 'system1_enabled',
    'revision', 'system1_revision'}  # Julia-1's commit: another one runs other code (emotion/teacher.py)
LOCATION = re.compile(r'\s*([a-z][a-z0-9+.-]*://|[/\\~]|[a-z]:[/\\])', re.IGNORECASE)
def security_sensitive(key, value=None):
    # An empty mapping in the YAML is one editable value, so a whole section can arrive at once: check each leaf.
    if isinstance(value, dict): return security_sensitive(key) or any(security_sensitive(f'{key}.{name}', item) for name, item in value.items())
    if isinstance(value, (list, tuple)): return security_sensitive(key) or any(security_sensitive(key, item) for item in value)
    section, _, name = key.rpartition('.')
    if (name in SENSITIVE_SETTING_NAMES or name.startswith('hf_') or key == 'emotion.enabled'
            or name.endswith(('_path', '_dir', '_directory', '_folder', '_root', '_file', '_url', '_model_id', '_library', '_config', '_executable'))):
        return True
    return isinstance(value, str) and key != 'avatar.model' and bool(LOCATION.match(value))  # the avatar endpoint serves only .vrm files


def shown(key, value):
    """A change as the native confirmation dialog shows it, and the challenge signs it: a secret's new value (schema.secret)
    only as a fingerprint, so a typed API key never appears on screen. The fingerprint still binds the approval to that value."""
    from ..configuration.schema import secret
    if isinstance(value, dict): return {name: shown(f'{key}.{name}', item) for name, item in value.items()}
    if not secret(key) or value is None or value == '' or isinstance(value, (bool, list, tuple)): return value
    return 'new secret, sha256 ' + hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:16]


@router.get('/api/settings')
def get_settings(backend: Services): return backend.settings_store().snapshot()

class SettingsPatch(BaseModel):
    changes: dict
    revision: str = ''

@router.post('/api/settings/validate')
def validate_settings(request: SettingsPatch, backend: Services):
    try: return backend.settings_store().validate(request.changes)
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))

@router.put('/api/settings')
def save_settings(request: SettingsPatch, backend: Services, x_riko_confirmation: str | None = Header(default=None)):
    from ..configuration.settings_store import SettingsConflict, kept
    changes = kept(request.changes)  # a secret left masked keeps its saved value
    # Compared with the saved values, secrets unmasked (the snapshot masks them), so retyping the same key asks nothing.
    try: current = backend.settings_store().stored_values()
    except (OSError, ValueError, TypeError): current = {}
    backend.require_confirmation('settings', {key: shown(key, value) for key, value in changes.items()
        if security_sensitive(key, value) and current.get(key) != value}, x_riko_confirmation)
    try:
        result = backend.settings_store().save(changes, request.revision)
        if result.get('saved'):
            from ..configuration.schema import live_hooks
            # result['values'] has its secrets masked: a live hook that ever needs a secret must read the config instead.
            for name, paths in live_hooks(changes).items(): LIVE[name](backend, result['values'], paths)
        return result
    except SettingsConflict as exc: raise HTTPException(409, str(exc))
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))
    except OSError as exc: raise HTTPException(500, 'Could not write YAML. Check file permissions or close another editor holding the file.') from exc

# Saved settings that apply without a restart, by the hook their schema entry names (Setting.live, restart scope 'none'
# unless noted): each takes the backend, the saved values and the changed paths.
def live_probe_interval(backend, values, paths):
    interval = values['emotion.probe.interval_tokens']
    backend.config.emotion.probe['interval_tokens'] = interval
    host = backend.probe_host()
    if host: host.set_probe_interval(interval)

def live_background_pause(backend, values, paths):
    enabled, chat, session = values['runtime.pause_background_on_live'], backend.chat, backend.session
    backend.config.runtime.pause_background_on_live = enabled
    if chat is not None: chat.provider.set_pause_background(enabled)
    memory = getattr(chat, 'memory_store', None)
    if memory: memory.set_foreground(bool(enabled and session is not None and session.status().generating))

def live_avatar(backend, values, paths):
    from ..configuration.settings_store import LOCK
    config = backend.config
    with LOCK:
        config.avatar = deepcopy(load_config(backend.settings_store().path).avatar)
        config.raw['avatar'] = deepcopy(config.avatar)
    backend.bus.publish('state.snapshot', **backend.snapshot())

def live_initiative_budgets(backend, values, paths):  # the running initiative takes them; resizing the KV pool still needs a restart
    initiative = getattr(backend.session, 'initiative', None)
    if initiative: initiative.update({key.split('.')[-1]: values[key] for key in ('initiative.context_window_tokens', 'initiative.max_output_tokens')})

LIVE = {'probe_interval': live_probe_interval, 'background_pause': live_background_pause, 'avatar': live_avatar,
    'initiative_budgets': live_initiative_budgets}

class PathCheck(BaseModel):  # a file to import (/api/avatar/models/import)
    value: str

class SettingsPathCheck(BaseModel):  # the file or folder setting key, and the value its form holds
    key: str
    value: str

@router.post('/api/settings/path')
def check_settings_path(request: SettingsPathCheck, backend: Services):
    try: return backend.settings_store().check_path(request.key, request.value)
    except (ValueError, OSError) as exc: raise HTTPException(400, str(exc))

@router.get('/api/settings/huggingface/search')
def search_huggingface(query: str = '', gguf: bool = True):
    if len(query) > 200: raise HTTPException(400, 'Search is too long')
    import httpx
    params = {'search': query, 'limit': 12, 'sort': 'downloads', 'direction': -1}
    if gguf: params['filter'] = 'gguf'
    try:
        response = httpx.get('https://huggingface.co/api/models', params=params, timeout=8)
        response.raise_for_status()
        return {'models': [{'id': m['id'], 'downloads': m.get('downloads', 0)} for m in response.json()]}
    except (httpx.HTTPError, ValueError) as exc: raise HTTPException(502, 'Hugging Face search unavailable; enter an ID manually') from exc

@router.get('/api/settings/huggingface/files')
def huggingface_files(repo: str, revision: str = 'main'):
    import httpx
    from urllib.parse import quote
    if len(repo)>200 or '..' in repo or not re.fullmatch(r'[\w-]+/[\w.-]+', repo) or not revision or len(revision) > 200: raise HTTPException(400, 'Use owner/repository and a valid revision')
    try:
        response = httpx.get(f'https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe="")}', timeout=8)
        response.raise_for_status()
        data = response.json()
        return {'revision': data.get('sha'), 'files': [x['rfilename'] for x in data.get('siblings', []) if x['rfilename'].lower().endswith('.gguf') and 'mmproj' not in x['rfilename'].lower()],
            'context_length': data.get('gguf', {}).get('context_length')}
    except (httpx.HTTPError, ValueError) as exc: raise HTTPException(502, 'Cannot list that repository/revision; manual filenames remain available') from exc
