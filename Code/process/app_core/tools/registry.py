"""Tool policy: one register() for every tool and its name rules, then approval, de-duplication, deadlines and cancellation
for each call. The Tool shape is tools/tool.py; schemas, MCP transports and worker processes live in tools/schema.py,
tools/mcp and tools/isolation.py, and their old names here (local_definition, StdioMCPClient, ...) stay for one release."""
from __future__ import annotations

from concurrent.futures import TimeoutError, wait
from copy import deepcopy
import logging
import math
import re
import threading
import time
from typing import Any
import uuid

from ..kernel.messages import ToolResult
from ..kernel.workers import DaemonExecutor
from .isolation import IsolatedWorkers
from .mcp import (HTTPMCPClient, MCPTool, StdioMCPClient, approval_defaults, configured_servers, connect, resolve_command, server_tools,
    untrusted_description, untrusted_output, untrusted_schema)
from .schema import local_definition
from .tool import NAME, RIKO, RegisteredTool, Tool, ToolActivity, ToolCancelled, local_tool

logger = logging.getLogger(__name__)
INVALID = re.compile(r'[^a-zA-Z0-9_-]')


class ToolRegistry:
    def __init__(self, *, timeout_seconds=30.0, require_approval=False, activity=None, disabled=()):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0: raise ValueError('Tool timeout must be positive and finite')
        self.tools: dict[str, Tool] = {}
        self.disabled = frozenset(disabled)  # tools.disabled: never registered, so never offered to the model or run
        self.timeout_seconds = timeout_seconds
        self.require_approval = require_approval
        self.activity = activity if activity is not None else ToolActivity()
        self.clients = []
        self.executor = DaemonExecutor(max_workers=4, thread_name_prefix='tool')
        self.workers = IsolatedWorkers()
        self._running = {}
        self._execution_lock = threading.Lock()
        self._closed = False
        self.approvals = None
        self.choice_resolver = None

    def register(self, tool: Tool, *, source, replace=False):
        """Add tool from source (RIKO, or mcp:<server> for a configured server) while the runtime starts, and return the
        name the model calls it by. Riko's own tool keeps its name: an invalid one, or one another of Riko's tools has
        (unless replace), raises ValueError, and a server's tool that has it is renamed. A server's tool whose own name
        is invalid or taken runs as <server>__<name>, with a warning, or is left out when that is invalid or taken too."""
        tool.source = source
        original = tool.remote_name or tool.name
        names = {tool.name, original, *([f'{INVALID.sub("_", source.removeprefix("mcp:"))}__{INVALID.sub("_", str(original))}'] if source != RIKO else [])}
        if self.disabled & names:
            logger.info('Tool %r from %s is disabled (tools.disabled)', original, source)
            return None
        if source != RIKO: return self._add_external(tool)
        if not isinstance(tool.name, str) or not NAME.fullmatch(tool.name): raise ValueError(f'Invalid tool name {tool.name!r}: use 1-64 letters, digits, _ or -')
        held = self.tools.get(tool.name)
        if held is not None and held.source == RIKO and not replace: raise ValueError(f'Tool {tool.name!r} is already registered')
        self.tools[tool.name] = tool
        if held is not None and held.source != RIKO: self._add_external(held)  # Riko's tool takes its name back
        return tool.name

    def _add_external(self, tool):
        tool.read_only = False  # the MCP specification treats a server's annotations as untrusted
        original = tool.remote_name or tool.name
        try:  # its parameter text is capped like its description
            schema = untrusted_schema(tool.input_schema)
            if isinstance(tool, RegisteredTool): tool.schema = schema
            else: tool.input_schema = schema
        except ValueError as exc:
            logger.warning('MCP tool %r from %s was left out: %s', original, tool.source, exc)
            self.activity.notify('tools', f'Tool {original!r} from {tool.source} was not loaded: {exc}', 'error')
            return None
        server = INVALID.sub('_', tool.source.removeprefix('mcp:'))
        for name in [original, f'{server}__{INVALID.sub("_", str(original))}']:
            if isinstance(name, str) and NAME.fullmatch(name) and name not in self.tools:
                if name != original:
                    tool.remote_name = original
                    logger.warning('MCP tool %r from %s runs as %r: its own name is invalid or another tool has it', original, tool.source, name)
                tool.name, self.tools[name] = name, tool
                return name
        logger.warning('MCP tool %r from %s was left out: its name is invalid or taken', original, tool.source)
        self.activity.notify('tools', f'Tool {original!r} from {tool.source} was not loaded: its name is invalid or taken', 'error')
        return None

    def register_local(self, tool, *, owner=''):
        """register() for a local tool object (tools.tool.local_tool), always one of Riko's own."""
        return self.register(local_tool(tool, owner=owner), source=RIKO)

    def register_mcp(self, client, *, source, owner=''):
        """register() for each tool an MCP client lists; source is RIKO only for Riko's own in-process server (TaskMCP)."""
        if client not in self.clients: self.clients.append(client)
        return [self.register(tool, source=source) for tool in server_tools(client, trusted=source == RIKO, owner=owner)]

    def definitions(self, provider="openai", *, read_only=False):
        """Each tool's definition for the model, with its current choices as enums; read_only keeps the read-only ones. A
        configured server's description is framed as untrusted and capped (mcp.untrusted_description), and its schema was
        capped when it was registered (mcp.untrusted_schema)."""
        result = []
        for tool in list(self.tools.values()):
            if read_only and not tool.read_only: continue
            schema = deepcopy(tool.input_schema)
            if tool.choices:
                for key, options in tool.choices().items():
                    if key in schema.get('properties', {}) and options: schema['properties'][key]['enum'] = list(options)[:128]
            description = tool.description if tool.source == RIKO else untrusted_description(tool.source.removeprefix('mcp:'), tool.description)
            if provider == "openai":
                result.append({"type": "function", "function": {"name": tool.name, "description": description, "parameters": schema}})
            else:
                result.append({"name": tool.name, "description": description, "parameters": schema})
        return result

    def close(self):
        if self.choice_resolver: self.choice_resolver.close()
        if self.approvals: self.approvals.close()
        with self._execution_lock: self._closed = True
        self.workers.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        for client in self.clients:
            try: client.close()
            except Exception: logger.exception('Unable to close MCP client')

    def execute(self, name: str, arguments: dict[str, Any], call_id: str | None = None, *, cancelled=lambda: False) -> ToolResult:
        activity = self.activity
        tool = self.tools.get(name)
        if not tool: return ToolResult(call_id or str(uuid.uuid4()), name, f"Unknown tool: {name}", True)
        if self._closed or cancelled(): return ToolResult(call_id or str(uuid.uuid4()), name, 'Tool registry closed or call cancelled', True)
        arguments = deepcopy(arguments)
        corrections = []
        if self.choice_resolver and tool.choices:
            try: arguments, corrections = self.choice_resolver.normalize(name, arguments, tool.choices(arguments))
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
            if not self.approvals.authorize(name, arguments, call_id, cancelled, key=tool.approval_key):
                return ToolResult(call_id or str(uuid.uuid4()), name, 'Tool approval denied, expired or cancelled; tool was not executed', True)
        elif self.require_approval: return ToolResult(call_id or str(uuid.uuid4()), name, "Tool execution requires approval", True)
        with self._execution_lock:
            if cancelled(): return ToolResult(call_id or str(uuid.uuid4()), name, 'Tool cancelled before execution', True)
            if self._closed: return ToolResult(call_id or str(uuid.uuid4()), name, 'Tool registry closed', True)
            previous = self._running.get(name)
            if previous is not None and not previous.done():
                return ToolResult(call_id or str(uuid.uuid4()), name, 'Previous call is still running; retry blocked to prevent duplicate side effects', True)
            # This call's own stop: Stop kills only its worker or cancels only its MCP request.
            stop, stoppable = threading.Event(), bool(tool.isolated) or tool.cancellable
            future = (self.executor.submit(self.workers.run, tool.isolated, arguments, stop, self.timeout_seconds) if tool.isolated
                else self.executor.submit(tool.execute, arguments, stop.is_set))
            self._running[name] = future
        activity_id = activity.tool_started(name, arguments)
        try:
            # Isolated workers enforce their own hard deadline, including teardown. Waiting in slices lets Stop end
            # the call now: the turn holds the turn lock meanwhile, so every new message would be refused as busy.
            deadline = time.monotonic() + self.timeout_seconds + (1 if tool.isolated else 0)
            while not wait([future], max(0, min(.05, deadline - time.monotonic()))).done:
                if time.monotonic() >= deadline: raise TimeoutError('Tool deadline exceeded')
                if cancelled(): raise ToolCancelled('Tool call cancelled')
            result, error = tool.result(future.result())
            activity.tool_finished(name, result, error, activity_id=activity_id)
            # A configured server's result (an error too) reaches the model framed as untrusted data; the activity panel
            # above keeps it raw. Messages the registry writes itself (cancelled, timed out) are not the server's.
            if tool.source != RIKO: result = untrusted_output(tool.source.removeprefix('mcp:'), tool.remote_name or tool.name, result)
            if corrections: result = {'result':result,'effective_arguments':arguments,'input_corrections':corrections}
            return ToolResult(call_id or str(uuid.uuid4()), name, result, error)
        except ToolCancelled:
            self.workers.cancel(stop)
            future.cancel()  # still queued: it never runs
            if stoppable: wait([future], 1)  # a killed worker or a cancelled MCP request ends within moments
            message = ('Tool cancelled because its turn was stopped; ' + ('its worker was terminated.' if tool.isolated
                else 'the MCP server was asked to stop it.' if stoppable else 'it may still finish and cause side effects. Retries are blocked until it finishes.')
                + ' Prior side effects are not rolled back.')
            activity.tool_finished(name, message, True, activity_id=activity_id)
            return ToolResult(call_id or str(uuid.uuid4()), name, message, True)
        except TimeoutError:
            tool.abandon()
            message = ('Tool timed out; isolated worker terminated. Prior side effects are not rolled back.' if stoppable
                       else 'Tool timed out; its worker may still finish and cause side effects. Retries are blocked until it finishes.')
            activity.tool_finished(name, message, True, activity_id=activity_id)
            return ToolResult(call_id or str(uuid.uuid4()), name, message, True)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            activity.tool_finished(name, str(exc), True, activity_id=activity_id)
            # A server's failure carries its text as well (a JSON-RPC error message, an HTTP reason), so it is framed too.
            message = str(exc) if tool.source == RIKO else untrusted_output(tool.source.removeprefix('mcp:'), tool.remote_name or tool.name, str(exc), key='error')
            return ToolResult(call_id or str(uuid.uuid4()), name, message, True)

    @classmethod
    def from_config(cls, config, activity=None, local_tools=None):
        """Riko's built-in tools, then local_tools ({owner: tools}: the factory passes the desktop tools, so tools/ never
        imports desktop/), then the MCP servers mcp.json configures, which a local name never yields to (the factory adds
        TaskMCP's tools, SessionManager its own)."""
        servers = configured_servers(config.tools.mcp_config)
        registry = cls(timeout_seconds=config.tools.timeout_seconds, require_approval=config.tools.require_approval, activity=activity,
            disabled=config.tools.disabled)
        from .approval import ToolApprovals
        # A configured server's tools ask before each call unless its mcp.json entry says "require_approval": false.
        registry.approvals = ToolApprovals(config.paths.tool_approvals, config.tools.require_approval, approval_defaults(servers))
        try:
            from .builtin import iter_tools
            for tool in iter_tools(config.paths.todo_list): registry.register_local(tool, owner='builtin')
            for owner, tools in (local_tools or {}).items():
                for tool in tools: registry.register_local(tool, owner=owner)
            for name, server in servers.items():
                client = None
                try:
                    if 'require_approval' in server and type(server['require_approval']) is not bool: raise ValueError('require_approval must be true or false')
                    client = connect(server)
                    registry.register_mcp(client, source=f'mcp:{name}', owner=name)
                except Exception as exc:
                    if client: client.close()
                    logger.warning('Could not load MCP server %s: %s', name, exc)
                    registry.activity.notify('tools', f'MCP server {name} was not loaded: {exc}', 'error')
            return registry
        except BaseException:
            registry.close()
            raise


__all__ = ['HTTPMCPClient', 'MCPTool', 'RIKO', 'RegisteredTool', 'StdioMCPClient', 'Tool', 'ToolActivity', 'ToolCancelled', 'ToolRegistry',
    'local_definition', 'resolve_command']
