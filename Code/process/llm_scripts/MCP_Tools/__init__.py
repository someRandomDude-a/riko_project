import pathlib
from importlib import import_module
from typing import Dict, Any, List
from .base import BaseTool, ToolType

# Configuration
CONFIG: Dict[str, Any] = {}
CONTEXT: Dict[str, Any] = {}

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
tools: Dict[str, BaseTool] = {}
function_tools: Dict[str, BaseTool] = {}
resource_tools: Dict[str, BaseTool] = {}
prompt_only_tools: Dict[str, BaseTool] = {}

for file_path in TOOLS_DIR.glob("*.py"):
    if file_path.name in ("__init__.py", "base.py"):
        continue

    module_name = file_path.stem
    try:
        module = import_module(f".{module_name}", package=__package__)
    except Exception as e:
        print(f"[ERROR] Could not import {module_name}: {e}")
        continue

    if not hasattr(module, "Tool"):
        continue
    tool_class = getattr(module, "Tool")
    if not (isinstance(tool_class, type) and issubclass(tool_class, BaseTool)):
        continue

    try:
        tool_instance = tool_class(CONFIG, CONTEXT)
    except Exception as e:
        print(f"[ERROR] Failed to instantiate {module_name}: {e}")
        continue

    name = tool_instance.TOOL_NAME
    tools[name] = tool_instance

    # Categorise
    tool_type = tool_instance.TOOL_TYPE
    if tool_type == ToolType.FUNCTION:
        function_tools[name] = tool_instance
    elif tool_type == ToolType.RESOURCE:
        resource_tools[name] = tool_instance
    else:
        prompt_only_tools[name] = tool_instance

# Build the MCP_PROMPT
MCP_PROMPT_LINES = ["You have the following tools available:"]

if function_tools:
    MCP_PROMPT_LINES.append("\n--- CALLABLE FUNCTIONS (you can call these) ---")
    for name, tool in function_tools.items():
        MCP_PROMPT_LINES.append(tool.build_prompt_context())

if resource_tools:
    MCP_PROMPT_LINES.append("\n--- RESOURCE PROCESSORS (auto-processed by the system) ---")
    for name, tool in resource_tools.items():
        MCP_PROMPT_LINES.append(tool.build_prompt_context())

if prompt_only_tools:
    MCP_PROMPT_LINES.append("\n--- CONTEXT PROVIDERS (information injected into prompt) ---")
    for name, tool in prompt_only_tools.items():
        MCP_PROMPT_LINES.append(tool.build_prompt_context())

MCP_PROMPT = "\n".join(MCP_PROMPT_LINES)

COMMAND_NAMES = list(tools.keys())
FUNCTION_NAMES = list(function_tools.keys())
RESOURCE_NAMES = list(resource_tools.keys())

def call_tool(name: str, **kwargs) -> Any:
    """Entry point for calling any tool. Compatible with MCP tools/call."""
    tool = tools.get(name)
    if tool is None:
        raise KeyError(f"Unknown tool: {name}")
    return tool.execute(**kwargs)

def get_openai_function_definitions() -> List[Dict[str, Any]]:
    """Return a list of OpenAI-style function definitions for FUNCTION tools."""
    return [tool.to_mcp_definition() for tool in function_tools.values()]

__all__ = [
    "tools", "MCP_PROMPT", "COMMAND_NAMES", "FUNCTION_NAMES", "RESOURCE_NAMES",
    "call_tool", "get_openai_function_definitions"
]

print(f"[INFO] Loaded {len(tools)} tools ({len(function_tools)} callable, {len(resource_tools)} resources)")