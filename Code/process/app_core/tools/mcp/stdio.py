"""MCP over stdio: one persistent server process per configured server, restarted by the next call after it exits."""
from __future__ import annotations

from concurrent.futures import TimeoutError
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time

from ..tool import ToolCancelled

logger = logging.getLogger(__name__)
# An MCP server is third-party code, and the backend's environment holds API keys (RIKO_API_KEY, OPENAI_API_KEY, HF_TOKEN),
# the Discord token and whatever else the user exported. A server starts with only what a process needs to run (the
# official MCP SDK's default set), Riko's data location (so a configured Code/task_mcp_server.py finds the store), the
# names its mcp.json entry lists in inherit_env, and its own env.
SAFE_ENVIRONMENT = {
    'nt': ('APPDATA', 'HOMEDRIVE', 'HOMEPATH', 'LOCALAPPDATA', 'PATH', 'PATHEXT', 'PROCESSOR_ARCHITECTURE', 'SYSTEMDRIVE',
        'SYSTEMROOT', 'TEMP', 'TMP', 'USERNAME', 'USERPROFILE', 'WINDIR', 'COMSPEC', 'PROGRAMDATA', 'PROGRAMFILES', 'PROGRAMFILES(X86)',
        'PROGRAMW6432', 'COMMONPROGRAMFILES', 'NUMBER_OF_PROCESSORS', 'OS'),
    'posix': ('HOME', 'LOGNAME', 'USER', 'PATH', 'SHELL', 'TERM', 'LANG', 'TZ', 'TMPDIR', 'XDG_RUNTIME_DIR', 'XDG_CONFIG_HOME',
        'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'SSL_CERT_FILE', 'SSL_CERT_DIR')}
RIKO_ENVIRONMENT = ('RIKO_DATA_DIR', 'RIKO_CONFIG')
ENVIRONMENT_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_()]*')


def server_environment(declared=None, inherit=(), environ=None, *, windows=None):
    """An MCP server's environment: the safe set and RIKO_DATA_DIR/RIKO_CONFIG from environ (os.environ), the names in
    inherit, then declared (its mcp.json env), which wins. Names match case-insensitively on Windows; a value starting with
    '()' (an exported shell function) is dropped. MCP stdio is UTF-8: Python servers on Windows write the ANSI code page
    unless told otherwise, and one byte that does not decode would stop the stderr drain, so PYTHONIOENCODING is utf-8
    unless declared says otherwise (it changes only their stdio, not their file encodings)."""
    environ = os.environ if environ is None else environ
    windows = os.name == 'nt' if windows is None else windows
    fold = str.upper if windows else str
    wanted = {fold(name) for name in (*SAFE_ENVIRONMENT['nt' if windows else 'posix'], *RIKO_ENVIRONMENT, *inherit)}
    kept = {name: value for name, value in environ.items()
        if (fold(name) in wanted or not windows and name.startswith('LC_')) and not str(value).startswith('()')}
    return {'PYTHONIOENCODING': 'utf-8', **kept, **(declared or {})}


def inherited_names(value):
    """mcp.json's inherit_env for one server: a list of environment variable names, else ValueError (the server is not
    loaded and the Tools notice says why)."""
    if value is None: return ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(name, str) and ENVIRONMENT_NAME.fullmatch(name) for name in value):
        raise ValueError('inherit_env must be a list of environment variable names')
    return tuple(value)


def resolve_command(command, env):
    """Find an MCP server command on the PATH its process will get. Windows CreateProcess applies neither PATHEXT nor
    the child's PATH, so npx/uvx (.cmd shims) fail by bare name; a resolved .cmd/.bat path runs through cmd.exe."""
    if os.path.dirname(command): return command  # an explicit path runs as written (CreateProcess still adds .exe)
    search = os.pathsep.join(os.get_exec_path(env))
    found = shutil.which(command, path=search)
    if not found: raise FileNotFoundError(f'MCP server command {command!r} was not found on PATH: {search}')
    return found


class StdioMCPClient:
    """Minimal MCP JSON-RPC stdio client with a persistent server process."""
    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None, inherit=()):
        self.command, self.args, self.env, self.inherit = command, args, env, inherited_names(inherit)
        self._restart_lock = threading.Lock()
        withheld = len(set(os.environ) - set(server_environment(self.env, self.inherit)))
        if withheld: logger.info('MCP server %s starts without %d of the backend\'s environment variables; list names in its inherit_env to pass them',
            os.path.basename(str(command)), withheld)
        self._start()

    def _start(self):
        environment = server_environment(self.env, self.inherit)  # never the backend's whole environment (its keys and tokens)
        process = subprocess.Popen([resolve_command(self.command, environment), *(self.args or [])], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1, env=environment)
        responses = queue.Queue(maxsize=256)
        # A restart replaces the lock too: a request still holding the old one finishes against the old, exited process.
        self.process, self._responses, self._counter, self._lock = process, responses, 0, threading.Lock()
        def read_stdout():
            try:
                for line in process.stdout:
                    value = json.loads(line)
                    while True:
                        try: responses.put(value, timeout=.1); break
                        except queue.Full:
                            if process.poll() is not None: return
            except Exception as exc:
                try: responses.put_nowait(exc)
                except queue.Full: pass
            finally:
                try: responses.put_nowait(RuntimeError('MCP server closed stdout'))
                except queue.Full: pass
        threading.Thread(target=read_stdout, daemon=True, name='mcp-stdout').start()
        def drain_stderr():
            try:
                for _ in process.stderr: pass
            except (OSError, ValueError): pass
        threading.Thread(target=drain_stderr, daemon=True, name='mcp-stderr').start()
        try: self._initialize()
        except BaseException:
            self.close()
            raise

    def _read(self, timeout=30):
        try: response = self._responses.get(timeout=timeout)
        except queue.Empty: raise TimeoutError('MCP response deadline exceeded')
        if isinstance(response, Exception): raise response
        return response

    def _request(self, method, params=None, cancelled=lambda: False):
        with self._lock:
            if cancelled(): raise ToolCancelled('MCP request cancelled')  # stopped while another request held the server
            self._counter += 1
            request = {"jsonrpc": "2.0", "id": self._counter, "method": method, "params": params or {}}
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + 30
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError('MCP request deadline exceeded')
                if cancelled():
                    # MCP cancellation: the server should stop the request. A late reply carries this id, which later requests skip.
                    try:
                        self.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/cancelled',
                            'params': {'requestId': request['id'], 'reason': 'The user stopped the reply'}}) + '\n')
                        self.process.stdin.flush()
                    except (OSError, ValueError): pass  # the server already exited; the next call restarts it
                    raise ToolCancelled('MCP request cancelled')
                try: response = self._read(min(remaining, .05))
                except TimeoutError: continue
                if response.get("id") == request["id"]:
                    if "error" in response: raise RuntimeError(response["error"])
                    return response.get("result", {})

    def _initialize(self):
        self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "riko", "version": "0.1"}})
        with self._lock:
            self.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) + '\n')
            self.process.stdin.flush()

    def list_tools(self): return self._request("tools/list").get("tools", [])
    def call(self, name, arguments, cancelled=lambda: False):
        if self.process.poll() is not None:
            with self._restart_lock:
                if self.process.poll() is not None: self._start()
        return self._request("tools/call", {"name": name, "arguments": arguments}, cancelled)
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream: stream.close()
