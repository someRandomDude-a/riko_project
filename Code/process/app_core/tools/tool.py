"""The one shape every tool takes in ToolRegistry (Tool), its implementation for in-process and isolated tools
(RegisteredTool; tools/mcp.MCPTool extends it), and the adapter for local tool objects: the built-ins and desktop tools."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Protocol

from .schema import signature_schema

RIKO = 'riko'  # the source of Riko's own tools (built-in, desktop, session, tasks); a configured MCP server's is mcp:<name>
NAME = re.compile(r'[a-zA-Z0-9_-]{1,64}')  # the OpenAI function-name rule, which the Responses API and llama.cpp accept too


class ToolCancelled(RuntimeError): pass


class ToolActivity:
    """What ToolRegistry reports while tools run and load; desktop/activity.ActivityLog implements it for the UI. This one records nothing."""
    def tool_started(self, name, arguments): return None  # an id tool_finished receives back
    def tool_finished(self, name, result, error=False, activity_id=None): pass
    def notify(self, source, text, level='info'): pass


class Tool(Protocol):
    """What ToolRegistry.register takes. read_only mirrors MCP's readOnlyHint: it never changes its environment, and the
    initiative check offers exactly the read_only tools, so a tool opts in. Only Riko's own tools can be read_only, since
    the MCP specification treats a server's annotations as untrusted."""
    name: str  # NAME; register may rename a configured server's tool, which keeps the server's name in remote_name
    description: str
    input_schema: dict[str, Any]
    isolated: dict | None  # the request a disposable worker runs it from (tools/isolation.py); None runs it in-process
    read_only: bool
    owner: str  # the part of Riko that provides it (builtin, desktop, session, tasks) or the MCP server's name
    source: str  # RIKO or mcp:<server>, set by register
    remote_name: str
    approval_key: tuple[str, str]  # (source, the name its source knows): approval rules never belong to a name alone
    cancellable: bool  # execute stops the call itself once cancelled() turns true
    choices: Callable[..., dict] | None  # choices(arguments=None): enums per parameter, only those that apply to a call's arguments
    prepare: Callable[[dict], tuple[dict, list]] | None  # checks or corrects a call's arguments before approval
    def execute(self, arguments: dict, cancelled: Callable[[], bool]) -> Any: ...
    def result(self, value) -> tuple[Any, bool]: ...  # what the model sees, and whether the call failed
    def abandon(self) -> None: ...  # the call outlived its deadline


@dataclass
class RegisteredTool:
    """A Tool run in-process by handler(arguments), or in a disposable worker from its isolated request."""
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict], Any] | None = None
    isolated: dict | None = None
    choices: Callable[..., dict] | None = None
    prepare: Callable[[dict], tuple[dict, list]] | None = None
    read_only: bool = False
    owner: str = ''
    source: str = RIKO
    remote_name: str = ''
    cancellable = False

    @property
    def input_schema(self): return self.schema
    @property
    def approval_key(self): return self.source, self.remote_name or self.name
    def execute(self, arguments, cancelled=lambda: False): return self.handler(arguments)
    def result(self, value): return value, False
    def abandon(self): pass


def local_tool(tool, *, owner=''):
    """A RegisteredTool for an object with TOOL_NAME, TOOL_DESCRIPTION and execute(**arguments): a built-in (BaseTool, which
    describes its own input_schema and runs ISOLATED) or a desktop tool, whose execute signature is its schema. CHOICES,
    input_choices(arguments=None), prepare_arguments(arguments) and READ_ONLY are optional."""
    static = getattr(tool, 'CHOICES', {})
    schema = tool.input_schema if hasattr(tool, 'input_schema') else signature_schema(tool.execute, static)
    return RegisteredTool(tool.TOOL_NAME, tool.TOOL_DESCRIPTION, schema, lambda arguments: tool.execute(**arguments),
        isolated=tool.worker_request() if getattr(tool, 'ISOLATED', False) else None,
        choices=getattr(tool, 'input_choices', None) or (lambda arguments=None: static),
        prepare=getattr(tool, 'prepare_arguments', None), read_only=getattr(tool, 'READ_ONLY', False), owner=owner)
