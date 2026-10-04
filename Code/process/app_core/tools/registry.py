from __future__ import annotations

import inspect
from copy import deepcopy
import json
import logging
import os
import subprocess
import threading
import urllib.request
import uuid
import sys
import queue
import math
import time
from pathlib import Path
from concurrent.futures import TimeoutError
from ..runtime.workers import DaemonExecutor
from dataclasses import dataclass
from typing import Any
from typing import get_type_hints, get_args, get_origin
import types

from ..conversation.messages import ToolResult

logger = logging.getLogger(__name__)


def _schema_for(annotation):
    if get_origin(annotation) is types.UnionType:
        return {'anyOf': [_schema_for(value) if value is not type(None) else {'type': 'null'} for value in get_args(annotation)]}
    origin = getattr(annotation, "__origin__", None)
    if annotation in (int,): return {"type": "integer"}
    if annotation in (float,): return {"type": "number"}
    if annotation in (bool,): return {"type": "boolean"}
    if annotation in (list,): return {"type": "array"}
    if annotation in (dict,): return {"type": "object"}
    if origin is list: return {"type": "array"}
    return {"type": "string"}


def local_definition(tool) -> dict[str, Any]:
    signature = inspect.signature(tool._call)
    annotations = get_type_hints(tool._call)
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        properties[name] = {**_schema_for(annotations.get(name, parameter.annotation)), "description": f"Parameter: {name}"}
        if parameter.default is inspect.Parameter.empty: required.append(name)
    schema = {"type": "object", "properties": properties, "required": required}
    for key, options in getattr(tool, 'CHOICES', {}).items():
        if key in properties: properties[key]['enum'] = list(options)
    return {"name": tool.TOOL_NAME, "description": tool.TOOL_DESCRIPTION, "inputSchema": schema}


@dataclass
class RegisteredTool:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Any
    remote: bool = False
    isolated: dict | None = None
    client: Any = None
    choices: Any = None
    prepare: Any = None


class StdioMCPClient:
    """Minimal MCP JSON-RPC stdio client with a persistent server process."""
    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None):
        self.command, self.args, self.env = command, args, env
        self._restart_lock = threading.Lock()
        self.process = subprocess.Popen([command, *(args or [])], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1, env={**os.environ, **(env or {})})
        self._counter = 0
        self._lock = threading.Lock()
        self._responses = queue.Queue(maxsize=256)
        process, responses = self.process, self._responses
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

    def _request(self, method, params=None):
        with self._lock:
            self._counter += 1
            request = {"jsonrpc": "2.0", "id": self._counter, "method": method, "params": params or {}}
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + 30
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError('MCP request deadline exceeded')
                response = self._read(remaining)
                if response.get("id") == request["id"]:
                    if "error" in response: raise RuntimeError(response["error"])
                    return response.get("result", {})

    def _initialize(self):
        self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "riko", "version": "0.1"}})
        with self._lock:
            self.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) + '\n')
            self.process.stdin.flush()

    def list_tools(self): return self._request("tools/list").get("tools", [])
    def call(self, name, arguments):
        if self.process.poll() is not None:
            with self._restart_lock:
                if self.process.poll() is not None:
                    replacement = StdioMCPClient(self.command, self.args, self.env)
                    self.process, self._responses, self._counter, self._lock = replacement.process, replacement._responses, replacement._counter, replacement._lock
        return self._request("tools/call", {"name": name, "arguments": arguments})
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream: stream.close()


class HTTPMCPClient:
    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self.url, self.headers, self.counter = url, headers or {}, 0
        self._initialize()

    def _request(self, method, params=None):
        self.counter += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self.counter, "method": method, "params": params or {}}).encode()
        request = urllib.request.Request(self.url, body, {"Content-Type": "application/json", **self.headers})
        with urllib.request.urlopen(request, timeout=30) as response: payload = json.loads(response.read())
        if "error" in payload: raise RuntimeError(payload["error"])
        return payload.get("result", {})

    def _initialize(self):
        self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "riko", "version": "0.1"}})
    def list_tools(self): return self._request("tools/list").get("tools", [])
    def call(self, name, arguments): return self._request("tools/call", {"name": name, "arguments": arguments})
    def close(self): pass


class ToolRegistry:
    def __init__(self, *, timeout_seconds=30.0, require_approval=False):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0: raise ValueError('Tool timeout must be positive and finite')
        self.tools: dict[str, RegisteredTool] = {}
        self.timeout_seconds = timeout_seconds
        self.require_approval = require_approval
        self.clients = []
        self.executor = DaemonExecutor(max_workers=4, thread_name_prefix='tool')
        self._running = {}
        self._execution_lock = threading.Lock()
        self._closed = False
        self._processes = set()
        self.approvals = None
        self.choice_resolver = None

    def register_local(self, tool):
        definition = local_definition(tool)
        self.tools[definition["name"]] = RegisteredTool(definition["name"], definition["description"], definition["inputSchema"], lambda args, t=tool: t.execute(**args))
        self.tools[definition['name']].choices = getattr(tool, 'input_choices', None) or (lambda: getattr(tool, 'CHOICES', {}))
        self.tools[definition['name']].prepare = getattr(tool, 'prepare_arguments', None)
        if type(tool).__module__.startswith('process.app_core.tools.builtin.'):
            self.tools[definition['name']].isolated = {'module': type(tool).__module__, 'class': type(tool).__name__,
                'config': getattr(tool, 'config', {}), 'context': getattr(tool, 'context', {})}

    def register_mcp(self, client):
        if client not in self.clients: self.clients.append(client)
        for definition in client.list_tools():
            self.tools[definition["name"]] = RegisteredTool(definition["name"], definition.get("description", ""), definition.get("inputSchema", {"type": "object"}), lambda args, n=definition["name"], c=client: c.call(n, args), True)
            self.tools[definition['name']].client = client

    def definitions(self, provider="openai"):
        result = []
        for tool in self.tools.values():
            schema = deepcopy(tool.schema)
            if tool.choices:
                for key, options in tool.choices().items():
                    if key in schema.get('properties', {}) and options: schema['properties'][key]['enum'] = list(options)[:128]
            if provider == "openai":
                result.append({"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": schema}})
            else:
                result.append({"name": tool.name, "description": tool.description, "parameters": schema})
        return result

    def close(self):
        if self.choice_resolver: self.choice_resolver.close()
        if self.approvals: self.approvals.close()
        with self._execution_lock:
            self._closed = True
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None: process.kill()
        self.executor.shutdown(wait=False, cancel_futures=True)
        for client in self.clients:
            try: client.close()
            except Exception: logger.exception('Unable to close MCP client')

    def _isolated_call(self, tool, arguments):
        command = [sys.executable, '--tool-worker'] if getattr(sys, 'frozen', False) else [sys.executable, str(Path(__file__).with_name('worker.py'))]
        process = subprocess.Popen(command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        with self._execution_lock:
            if self._closed:
                process.kill(); process.wait()
                raise RuntimeError('Tool registry closed')
            self._processes.add(process)
        try:
            try:
                output, _ = process.communicate(json.dumps({**tool.isolated, 'arguments': arguments}), timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill(); process.communicate()
                raise TimeoutError('Isolated tool terminated at deadline')
            payload = json.loads(output.splitlines()[-1])
            if 'error' in payload: raise RuntimeError(payload['error'])
            return payload['result']
        finally:
            with self._execution_lock: self._processes.discard(process)

    def execute(self, name: str, arguments: dict[str, Any], call_id: str | None = None, *, cancelled=lambda: False) -> ToolResult:
        from ..desktop.state import get_desktop_state
        desktop_state = get_desktop_state()
        tool = self.tools.get(name)
        if not tool: return ToolResult(call_id or str(uuid.uuid4()), name, f"Unknown tool: {name}", True)
        if self._closed or cancelled(): return ToolResult(call_id or str(uuid.uuid4()), name, 'Tool registry closed or call cancelled', True)
        arguments = deepcopy(arguments)
        corrections = []
        if self.choice_resolver and tool.choices:
            choices = tool.choices()
            if name == 'visual_effect' and (arguments.get('asset') or arguments.get('action') in {'list','stop'}): choices = {'action':choices['action']}
            try: arguments, corrections = self.choice_resolver.normalize(name, arguments, choices)
            except ValueError as exc: return ToolResult(call_id or str(uuid.uuid4()), name, str(exc), True)
        if tool.prepare:
            try:
                arguments, path_corrections = tool.prepare(arguments)
                corrections.extend(path_corrections)
            except (ValueError, OSError) as exc: return ToolResult(call_id or str(uuid.uuid4()), name, str(exc), True)
        if corrections:
            from ..events.bus import event_bus
            event_bus.publish('tool.call_normalized', name=name, arguments=arguments, corrections=corrections)
        if self.approvals:
            if not self.approvals.authorize(name, arguments, call_id, cancelled):
                return ToolResult(call_id or str(uuid.uuid4()), name, 'Tool approval denied, expired or cancelled; tool was not executed', True)
        elif self.require_approval: return ToolResult(call_id or str(uuid.uuid4()), name, "Tool execution requires approval", True)
        with self._execution_lock:
            if cancelled(): return ToolResult(call_id or str(uuid.uuid4()), name, 'Tool cancelled before execution', True)
            if self._closed: return ToolResult(call_id or str(uuid.uuid4()), name, 'Tool registry closed', True)
            previous = self._running.get(name)
            if previous is not None and not previous.done():
                return ToolResult(call_id or str(uuid.uuid4()), name, 'Previous call is still running; retry blocked to prevent duplicate side effects', True)
            future = self.executor.submit(self._isolated_call, tool, arguments) if tool.isolated else self.executor.submit(tool.handler, arguments)
            self._running[name] = future
        activity_id = desktop_state.tool_started(name, arguments)
        try:
            # Isolated workers enforce their own hard deadline, including teardown.
            result = future.result(timeout=self.timeout_seconds + (1 if tool.isolated else 0))
            error = bool(tool.remote and isinstance(result, dict) and result.get('isError'))
            if tool.remote and isinstance(result, dict):
                result = result.get('structuredContent', result.get('content', result))
                if isinstance(result, list): result = '\n'.join(item.get('text', '') for item in result if isinstance(item, dict) and item.get('type') == 'text')
            desktop_state.tool_finished(name, result, error, activity_id=activity_id)
            if corrections: result = {'result':result,'effective_arguments':arguments,'input_corrections':corrections}
            return ToolResult(call_id or str(uuid.uuid4()), name, result, error)
        except TimeoutError:
            if isinstance(tool.client, StdioMCPClient): tool.client.close()
            message = ('Tool timed out; isolated worker terminated. Prior side effects are not rolled back.' if tool.isolated or isinstance(tool.client, StdioMCPClient)
                       else 'Tool timed out; its worker may still finish and cause side effects. Retries are blocked until it finishes.')
            desktop_state.tool_finished(name, message, True, activity_id=activity_id)
            return ToolResult(call_id or str(uuid.uuid4()), name, message, True)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            desktop_state.tool_finished(name, str(exc), True, activity_id=activity_id)
            return ToolResult(call_id or str(uuid.uuid4()), name, str(exc), True)

    @classmethod
    def from_config(cls, config):
        raw = json.loads(config.tools.mcp_config.read_text(encoding='utf-8')) if config.tools.mcp_config and config.tools.mcp_config.exists() else {}
        registry = cls(timeout_seconds=config.tools.timeout_seconds, require_approval=config.tools.require_approval)
        from .approval import ToolApprovals
        registry.approvals = ToolApprovals(config.root / 'persistent_memories' / 'tool_approvals.json', config.tools.require_approval)
        try:
            from .builtin import iter_tools
            for tool in iter_tools(): registry.register_local(tool)
            from ..desktop.tools import iter_tools as desktop_tools
            for tool in desktop_tools(): registry.register_local(tool)
            for name, server in raw.get('mcpServers', raw.get('servers', {})).items():
                client = None
                try:
                    client = HTTPMCPClient(server['url'], server.get('headers')) if server.get('url') else StdioMCPClient(server['command'], server.get('args'), server.get('env'))
                    registry.register_mcp(client)
                except Exception as exc:
                    if client: client.close()
                    logger.warning('Could not load MCP server %s: %s', name, exc)
            return registry
        except BaseException:
            registry.close()
            raise
