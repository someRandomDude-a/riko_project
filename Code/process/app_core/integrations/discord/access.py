"""Local, fail-closed Discord access editor. Bot credentials stay in .env."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import threading

from .config import BotSettings

KEYS = {'admins', 'users', 'channels', 'allow_dms', 'admin_actions'}


def validate_access(raw):
    if not isinstance(raw, dict) or set(raw) != KEYS: raise ValueError('Supply admins, users, channels, allow_dms and admin_actions')
    result = {}
    for key in ('admins', 'users', 'channels'):
        values = raw[key]
        if not isinstance(values, list) or len(values) > 256: raise ValueError(f'{key} must be a list of at most 256 Discord IDs')
        if any(not isinstance(value, str) or not value.isascii() or not value.isdecimal() or not 0 < int(value) < 2**64 for value in values):
            raise ValueError('Discord IDs must be positive decimal strings (enable Developer Mode in Discord to copy IDs)')
        result[key] = sorted({str(int(value)) for value in values}, key=int)
    for key in ('allow_dms', 'admin_actions'):
        if type(raw[key]) is not bool: raise ValueError(f'{key} must be a boolean')
        result[key] = raw[key]
    return result


def apply_access(settings, raw):
    values = validate_access(raw)
    return replace(settings, **{key: frozenset(map(int, values[key])) for key in ('admins', 'users', 'channels')},
        allow_dms=values['allow_dms'], admin_actions=values['admin_actions'])


class DiscordAccess:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'persistent_memories' / 'discord_access.json'
        self.lock = threading.RLock()
        self._credentials_key = None
        self._credentials = None
        self._access_key = None
        self._values = None

    @staticmethod
    def signature(path):
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size
        except FileNotFoundError: return None

    def credentials(self):
        from dotenv import dotenv_values
        with self.lock:
            path = self.root / '.env'
            key = (self.signature(path), tuple(sorted((name,value) for name,value in os.environ.items() if name.startswith('Discord_'))))
            if self._credentials is not None and key == self._credentials_key: return self._credentials
            env = {name: value for name, value in dotenv_values(path).items() if value is not None}
            settings = BotSettings.from_env(self.root, {**env, **os.environ})
            self._credentials_key, self._credentials = key, settings
            return settings

    def read(self):
        with self.lock:
            key = self.signature(self.path)
            if key is not None and key == self._access_key and self._values is not None:
                values = self._values
            elif key is not None:
                try: values = validate_access(json.loads(self.path.read_text(encoding='utf-8')))
                except (ValueError, OSError, TypeError) as exc: raise ValueError('Invalid Discord access file; original preserved. Repair it before starting Discord.') from exc
            else:
                settings = self.credentials()
                values = {key: sorted(map(str, getattr(settings, key)), key=int) for key in ('admins', 'users', 'channels')}
                values.update(allow_dms=settings.allow_dms, admin_actions=settings.admin_actions)
            revision = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
            self._access_key, self._values = key, values
            return {'values': {name: list(value) if isinstance(value, list) else value for name,value in values.items()}, 'revision': revision}

    def settings(self):
        with self.lock: return apply_access(self.credentials(), self.read()['values'])

    def save(self, values, revision):
        with self.lock:
            current = self.read()
            if current['revision'] != revision: raise RuntimeError('Discord settings changed elsewhere. Reload before saving.')
            values = validate_access(values)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(values, indent=2), encoding='utf-8')
            temporary.replace(self.path)
            self._access_key = None
            return self.read()
