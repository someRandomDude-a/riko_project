"""Local API trust boundary: only Riko's own clients may use the backend.

Every HTTP request and WebSocket handshake needs this backend start's token as
`Authorization: Bearer <token>`, a loopback Host, and for sockets an app Origin. The
token and a separate confirmation key are fresh on every start. Packaged Electron generates
both, passes them in the environment, and sends the token only after the backend reports
that it holds its port. In development the backend mints them only after it holds its
port and writes them under persistent_memories (owner-only) for Electron to read, so a
token sent to a port squatter while the backend was down is useless afterwards. Electron main adds the token at the network layer, so renderer
JavaScript never sees it. Security-sensitive changes also need a single-use confirmation:
Electron main signs the server's challenge with the confirmation key, which never
travels over the network, only after the user approves a native dialog.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time

HOSTS = {'127.0.0.1', 'localhost'}
# Packaged file:// renderer, the Vite dev server, and clients that send no Origin (Python, Node).
WEBSOCKET_ORIGINS = {'null', 'file://', 'http://localhost:5173', 'http://127.0.0.1:5173'}
SIGNATURE = re.compile(r'[0-9a-f]{64}')
SECRET = re.compile(r'[A-Za-z0-9_-]{32,256}')  # what Electron and secrets.token_urlsafe produce
DEFAULT_PORT = 8765  # electron/backend_origin.cjs holds Electron's copy of the default


def backend_port(env=None):
    """The loopback port this backend binds and its clients reach: RIKO_PORT (1024-65535), else 8765. Packaged Electron
    passes the free port it chose; in development set it here and RIKO_BACKEND_URL for Electron. The guard checks the Host
    name, never the port, so it holds on any port."""
    value = (os.environ if env is None else env).get('RIKO_PORT', '').strip()
    if not value: return DEFAULT_PORT
    if not (value.isascii() and value.isdigit() and 1024 <= int(value) <= 65535):
        raise ValueError(f'RIKO_PORT must be a port from 1024 to 65535, not {value!r}')
    return int(value)


def backend_url(env=None):
    """This backend's address for a client: always 127.0.0.1, since 'localhost' may resolve to ::1."""
    return f'http://127.0.0.1:{backend_port(env)}'


def secret_path(root, name):
    """A client's view of DataPaths.api_token and .confirm_key (configuration/paths.py), from the data root alone."""
    return Path(root) / 'persistent_memories' / name


def write_secret(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{secrets.token_hex(4)}.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # owner-only from the start
    try:
        with os.fdopen(descriptor, 'w', encoding='ascii') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def issue_secrets(paths):
    """Fresh (token, confirmation key) for this backend start; call once the port is bound. paths: the backend's DataPaths
    (configuration/paths.py), whose api_token and confirm_key files electron/main.cjs reads in development.

    Values passed by packaged Electron are used as given. Both are removed from os.environ
    so tool workers, MCP servers and other children never inherit them.
    """
    token, key = os.environ.pop('RIKO_API_TOKEN', ''), os.environ.pop('RIKO_CONFIRM_KEY', '')
    if SECRET.fullmatch(token) and SECRET.fullmatch(key) and token != key: return token, key
    token, key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    write_secret(paths.api_token, token)
    write_secret(paths.confirm_key, key)
    return token, key


def client_root():
    """The data root a client process shares with the backend: the directory of RIKO_CONFIG."""
    base = Path(os.environ.get('RIKO_DATA_DIR') or Path.cwd())
    config = Path(os.environ.get('RIKO_CONFIG', 'character_config.yaml')).expanduser()
    return (config if config.is_absolute() else base / config).parent


def client_token():
    """The current token for a client such as the Discord worker; read again on every use,
    because it changes whenever the backend restarts."""
    if os.environ.get('RIKO_API_TOKEN'): return os.environ['RIKO_API_TOKEN']
    try: return secret_path(client_root(), 'api_token').read_text(encoding='ascii').strip()
    except (OSError, ValueError): return ''


def signature(key, challenge):
    return hmac.new(key.encode(), challenge.encode(), hashlib.sha256).hexdigest()


class Confirmations:
    """Single-use, short-lived challenges for security-sensitive changes."""
    def __init__(self, ttl=120, limit=64):
        self.ttl, self.limit = ttl, limit
        self.pending = {}  # nonce -> (challenge text, action, changes, expires)
        self.lock = threading.Lock()

    def challenge(self, action, changes):
        nonce, expires = secrets.token_urlsafe(16), int(time.time() + self.ttl)
        text = json.dumps({'action': action, 'changes': changes, 'nonce': nonce, 'expires': expires},
            sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        with self.lock:
            self._prune()
            while len(self.pending) >= self.limit: self.pending.pop(next(iter(self.pending)))
            self.pending[nonce] = (text, action, json.loads(json.dumps(changes)), expires)
        return text

    def consume(self, key, action, changes, provided):
        """True once for a valid signature over a pending challenge for exactly these changes."""
        if not isinstance(provided, str) or not SIGNATURE.fullmatch(provided): return False
        changes = json.loads(json.dumps(changes))
        with self.lock:
            self._prune()
            for nonce, (text, pending_action, pending_changes, _) in list(self.pending.items()):
                if pending_action == action and pending_changes == changes and hmac.compare_digest(signature(key, text), provided):
                    del self.pending[nonce]
                    return True
        return False

    def _prune(self):
        now = time.time()
        for nonce in [n for n, item in self.pending.items() if item[3] < now]: del self.pending[nonce]


class LocalAPIGuard:
    """Pure ASGI middleware, so it covers HTTP and WebSocket scopes alike."""
    def __init__(self, app, token):
        self.app, self.token = app, token  # token: zero-argument callable returning the current token

    async def __call__(self, scope, receive, send):
        if scope['type'] not in ('http', 'websocket'): return await self.app(scope, receive, send)
        headers = {key.decode('latin-1').lower(): value.decode('latin-1') for key, value in scope.get('headers', [])}
        problem = self.problem(scope['type'], headers)
        if problem is None: return await self.app(scope, receive, send)
        status, detail = problem
        if scope['type'] == 'websocket':
            await receive()  # websocket.connect; closing before accept rejects the handshake (HTTP 403)
            await send({'type': 'websocket.close', 'code': 1008, 'reason': detail})
            return
        body = json.dumps({'detail': detail}).encode()
        await send({'type': 'http.response.start', 'status': status,
            'headers': [(b'content-type', b'application/json'), (b'content-length', str(len(body)).encode())]})
        await send({'type': 'http.response.body', 'body': body})

    def problem(self, kind, headers):
        host = headers.get('host', '')
        if (host.split(':', 1)[0] if not host.startswith('[') else host).lower() not in HOSTS:
            return 403, 'Requests must be addressed to 127.0.0.1 or localhost'
        if kind == 'websocket' and headers.get('origin') not in (None, *WEBSOCKET_ORIGINS):
            return 403, 'This origin may not open Riko sockets'
        expected = 'Bearer ' + self.token()
        if not hmac.compare_digest(headers.get('authorization', '').encode('latin-1'), expected.encode('latin-1')):
            return 401, 'Missing or invalid API token'
        return None
