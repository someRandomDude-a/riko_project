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
    backend_url: str = 'http://127.0.0.1:8765'
    ffmpeg: str = 'ffmpeg'
    camera_url: str = ''
    sync_guild: int | None = None
    allow_dms: bool = True
    admin_actions: bool = True

    @classmethod
    def from_env(cls, root, env):
        backend = env.get('Discord_backend_url', 'http://127.0.0.1:8765').rstrip('/')
        parsed = urlsplit(backend)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError('Discord backend must be a loopback http:// URL without credentials or a path')
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
