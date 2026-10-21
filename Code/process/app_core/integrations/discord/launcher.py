"""User-started, managed Discord transport. Never loads a second model."""
import os
import subprocess
import threading
from uuid import UUID
import time

from .access import DiscordAccess
from ...events.bus import event_bus
from ...events.outbox import Outbox
from ...kernel.code_paths import CodePaths


class DiscordLauncher:
    def __init__(self, paths, activity=None):
        """paths: the backend's DataPaths (configuration/paths.py): the access file, the .env and the data root the worker shares.
        The worker itself is code (CodePaths), never looked for under the data root. activity: the desktop's ActivityLog
        (desktop/activity.py), which shows the client's status, notifications and allowed messages."""
        self.paths = paths
        self.lock = threading.RLock()
        self.token = lambda: ''  # The backend's API token accessor; http/backend.Backend sets it.
        self.outbox = Outbox(event_bus)  # status/notifications reach the session; never emit under self.lock
        self.process = None
        self.error = ''
        self.access = DiscordAccess(paths)
        self.activity = activity
        self.client_id = None
        self.ready = False
        self.bot_name = ''
        self.inbox = []
        self.received = 0

    def status(self):
        with self.lock:
            managed = self.process is not None and self.process.poll() is None
            result = {'running': managed or self.client_id is not None, 'managed': managed, 'ready': self.ready,
                'status': 'ready' if self.ready else 'starting' if managed or self.client_id else 'error' if self.error else 'stopped',
                'error': self.error, 'bot_name': self.bot_name, 'received': self.received}
            try:
                result['settings'] = self.access.read()
                result['token_configured'] = bool(self.access.credentials().token)
            except ValueError:
                result['configuration_error'] = 'Discord configuration is invalid; check local access settings and .env.'
            return result

    def publish(self):
        value = self.status()
        event_bus.publish('resource.discord', **value)
        if self.activity: self.activity.set_discord({key: value[key] for key in ('running', 'ready', 'status', 'error', 'bot_name', 'received')})

    def notify(self, text, level='info'):
        if self.activity: self.activity.notify('discord', text, level)
        else: event_bus.publish('discord.notification', text=text, level=level, source='discord')

    def attach(self, client_id):
        UUID(client_id)
        with self.lock:
            if self.client_id is not None: raise ValueError('A Discord client is already connected')
            self.client_id, self.ready = client_id, False
        self.publish()

    def detach(self, client_id):
        with self.lock:
            if self.client_id != client_id: return
            self.client_id, self.ready = None, False
        self.notify('Discord client disconnected.')
        self.publish()

    def inbox_snapshot(self):
        with self.lock: return {'messages': [dict(item) for item in self.inbox], 'received': self.received, 'limit': 256}

    def report(self, client_id, value):
        try: self._report(client_id, value)
        finally: self.outbox.flush()

    def _report(self, client_id, value):
        if not isinstance(value, dict): raise ValueError('Invalid Discord report')
        with self.lock:
            if self.client_id != client_id: raise ValueError('Discord connection changed')
            if value.get('kind') == 'ready':
                first = not self.ready
                self.ready, self.error = True, ''
                self.bot_name = str(value.get('bot_name', 'Discord bot'))[:100]
                if first: self.outbox.defer(self.notify, 'Discord connected as ' + self.bot_name + '.')
                self.outbox.defer(self.publish)
                return
            if value.get('kind') == 'disconnected':
                self.ready = False
                self.outbox.defer(self.notify, 'Discord gateway disconnected; reconnecting.', 'warning')
                self.outbox.defer(self.publish)
                return
            if value.get('kind') != 'message': raise ValueError('Unknown Discord report')
            def snowflake(key, optional=False):
                raw = value.get(key)
                if optional and raw is None: return None
                if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal() or not 0 < int(raw) < 2**64: raise ValueError('Invalid Discord ID')
                return raw
            user_id, channel_id, guild_id = snowflake('user_id'), snowflake('channel_id'), snowflake('guild_id', True)
            message_id = str(value.get('message_id', ''))[:100]
            if not message_id: raise ValueError('Message ID is required')
            key = client_id + ':' + message_id
            if any(item['id'] == key for item in self.inbox): return
            ignored = value.get('author_bot') is True or value.get('webhook') is True
            allowed = not ignored and self.access.settings().allows(int(user_id), int(channel_id), guild=guild_id is not None)
            item = {'id': key, 'message_id': message_id, 'user_id': user_id, 'channel_id': channel_id, 'guild_id': guild_id,
                'user_name': str(value.get('user_name', 'Discord user'))[:100], 'channel_name': str(value.get('channel_name', channel_id))[:100],
                'text': str(value.get('text', ''))[:2000], 'timestamp': time.time(), 'source': 'discord',
                'status': 'ignored' if ignored else 'allowed' if allowed else 'blocked',
                'attachments': [str(name)[:200] for name in value.get('attachments', [])[:10]] if isinstance(value.get('attachments', []), list) else []}
            self.received += 1
            self.inbox = [item, *self.inbox[:255]]
        event_bus.publish('discord.message_seen', **item)
        if allowed and self.activity:
            self.activity.observe_input('discord', item['text'], message_id=message_id,
                context={key: item[key] for key in ('user_id', 'channel_id', 'guild_id', 'user_name')})

    def configure(self, values, revision):
        result = self.access.save(values, revision)
        self.notify('Discord access settings saved; they apply to new requests immediately.')
        self.publish()
        return result

    def start(self):
        try: return self._start()
        finally: self.outbox.flush()

    def _start(self):
        with self.lock:
            if self.status()['running']:
                self.outbox.defer(self.notify, 'Discord is already started' + (' and connected.' if self.ready else '; it is connecting.'))
                return {**self.status(), 'already_started': True}
            try:
                try: settings = self.access.settings()
                except ValueError as exc: raise ValueError('Discord configuration is invalid. Check local access IDs and the backend URL.') from exc
                if not settings.token: raise ValueError('Configure Discord_bot_token in the local .env before starting Discord')
                if not settings.admins: raise ValueError('Configure Discord administrators in Settings → Discord or Discord_admins in the local .env')
                code = CodePaths.current()
                if not code.frozen and not code.discord_bot.is_file(): raise ValueError('Discord client entry point is missing')
                # The worker exits when its stdin closes, i.e. when this backend exits however it exits,
                # so a client holding a dead backend's token never outlives it. It finds this backend's data root, .env and
                # token from RIKO_CONFIG (absolute) and RIKO_DATA_DIR, and its address (RIKO_PORT, which run_server bound)
                # from Discord_backend_url, which outranks the .env's, whatever this process's environment held.
                from ...desktop.api_guard import backend_url
                env = {**os.environ, 'RIKO_API_TOKEN': self.token(), 'RIKO_DATA_DIR': str(self.paths.root),
                    'RIKO_CONFIG': str(self.paths.config_file), 'RIKO_EXIT_WITH_BACKEND': '1', 'Discord_backend_url': backend_url()}
                self.process = subprocess.Popen(code.discord_command(), cwd=self.paths.root, env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.error = ''
            except ValueError as exc:
                self.error = str(exc); self.outbox.defer(self.publish); raise
            except OSError as exc:
                self.error = 'Discord client could not be started'; self.outbox.defer(self.publish); raise RuntimeError(self.error) from exc
            process = self.process
            threading.Thread(target=self._watch, args=(process,), name='discord-process', daemon=True).start()
            self.outbox.defer(self.publish)
            self.outbox.defer(self.notify, 'Discord client is starting; waiting for its gateway connection.')
            return self.status()

    def _watch(self, process):
        errors = {'login': 'Discord rejected the bot token. Update Discord_bot_token in the local .env.',
            'intents': 'Enable Message Content Intent in the Discord developer portal, then restart Discord.',
            'backend': 'Discord could not connect to the current Python backend.',
            'dependency': 'Discord dependencies are missing in the backend Python environment.',
            'startup': 'Discord startup failed. Check local bot configuration and permissions.'}
        stream = getattr(process, 'stdout', None)
        if stream:
            for line in iter(lambda: stream.readline(4096), ''):
                if line.strip().startswith('RIKO_DISCORD_ERROR:'):
                    code = line.strip().partition(':')[2]
                    if code in errors:
                        with self.lock:
                            if self.process is process: self.error = errors[code]
        code = process.wait()
        if getattr(process, 'stdin', None):
            try: process.stdin.close()
            except OSError: pass
        with self.lock:
            if self.process is not process: return
            self.process = None
            if code and not self.error: self.error = 'Discord client exited. Check bot credentials, permissions and installed Discord dependencies.'
        self.notify(self.error if code else 'Discord client stopped.', 'error' if code else 'info')
        self.publish()

    def stop(self):
        with self.lock:
            process = self.process
            self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=2)
        self.publish()
