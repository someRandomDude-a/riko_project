"""Fail-closed Discord access policy; credentials never come from chat/settings."""
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def ids(value):
    result = frozenset(int(item.strip()) for item in value.split(',') if item.strip())
    if any(item <= 0 for item in result): raise ValueError('Discord IDs must be positive integers')
    return result


@dataclass(frozen=True)
class BotSettings:
    root: Path
    token: str = field(default='', repr=False)
    admins: frozenset = frozenset()
    users: frozenset = frozenset()
    channels: frozenset = frozenset()
    backend_url: str = 'http://127.0.0.1:8765'  # from_env: Discord_backend_url, else the backend's RIKO_PORT (api_guard.backend_url)
    ffmpeg: str = 'ffmpeg'
    camera_url: str = ''
    sync_guild: int | None = None
    allow_dms: bool = True
    admin_actions: bool = True

    # Every key from_env reads; environment overrides are matched to them without case (Windows upper-cases names).
    KEYS = ('Discord_backend_url', 'RIKO_PORT', 'Discord_camera_url', 'Discord_bot_token', 'Discord_admins', 'Discord_allowed_users',
        'Discord_Channel_whitelist', 'Discord_ffmpeg', 'Discord_sync_guild')

    @classmethod
    def overrides(cls, environ):
        """The process environment's values for KEYS, under their own spelling whatever case the OS reports."""
        canonical = {key.casefold(): key for key in cls.KEYS}
        return {canonical[name.casefold()]: value for name, value in environ.items() if name.casefold() in canonical}

    @classmethod
    def from_env(cls, root, env):
        from ...desktop.api_guard import backend_url
        # The launcher passes its own backend's address here, so a started worker reaches it whatever the .env says.
        backend = (env.get('Discord_backend_url') or backend_url(env)).rstrip('/')
        parsed = urlsplit(backend)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError('Discord backend must be http://127.0.0.1:<port> without credentials or a path')
        # Always 127.0.0.1: 'localhost' can resolve to ::1, where another account's listener would receive the API token.
        backend = 'http://127.0.0.1' + (f':{parsed.port}' if parsed.port else '')
        camera = env.get('Discord_camera_url', '').strip()
        if camera:
            url = urlsplit(camera)
            if url.scheme != 'https' or not url.hostname or url.username or url.password:
                raise ValueError('Discord_camera_url must be an explicitly configured HTTPS stream/viewer URL')
        return cls(Path(root), env.get('Discord_bot_token', '').strip(), ids(env.get('Discord_admins', '')),
            ids(env.get('Discord_allowed_users', '')), ids(env.get('Discord_Channel_whitelist', '')),
            backend, env.get('Discord_ffmpeg', 'ffmpeg'), camera,
            int(env['Discord_sync_guild']) if env.get('Discord_sync_guild') else None)

    def allows(self, user_id, channel_id, *, guild=False, admin=False):
        if admin and not self.admin_actions or not guild and not self.allow_dms: return False
        trusted = self.admins if admin else self.admins | self.users
        return user_id in trusted and (not guild or channel_id in self.channels)


# A deliberately narrow remote editor. Never expose credentials, paths, model
# downloads, server commands, MCP configuration or the complete YAML snapshot.
BASIC_SETTINGS = frozenset({'runtime.temperature', 'runtime.max_output_tokens',
    'speech.max_words', 'speech.split_window_words', 'voice.vad_threshold',
    'voice.utterance_end_seconds', 'voice.interruption_seconds',
    'memory.token_budget', 'initiative.context_window_tokens', 'initiative.max_output_tokens'})
