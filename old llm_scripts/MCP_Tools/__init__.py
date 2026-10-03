import pathlib
import os
from importlib import import_module
from typing import Dict, Any, List
from .base import BaseTool, ToolType

# Global configuration & shared context
CONFIG: Dict[str, Any] = {}
CONTEXT: Dict[str, Any] = {}

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
tools: Dict[str, BaseTool] = {}
function_tools: Dict[str, BaseTool] = {}
resource_tools: Dict[str, BaseTool] = {}
prompt_only_tools: Dict[str, BaseTool] = {}

# Recursive tool discovery
def discover_tools(root: pathlib.Path, package: str) -> None:
    """
    Recursively find all .py files in `root` (relative to the package),
    import them, and register any class named `Tool` that subclasses BaseTool.
    """
    for file_path in root.rglob("*.py"):
        # Skip the top‑level __init__.py and base.py (they are not tools)
        if file_path.parent == root and file_path.name in ("__init__.py", "base.py"):
            continue

        # Compute the dotted module name relative to the package
        rel_path = file_path.relative_to(root)
        module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")

        # Attempt to import the module
        try:
            # Use a relative import from the current package
            module = import_module(f".{module_name}", package=package)
        except Exception as e:
            print(f"[WARN] Could not import {module_name}: {e}")
            continue

        # Look for a `Tool` class that is a subclass of BaseTool
        if not hasattr(module, "Tool"):
            continue
        tool_class = getattr(module, "Tool")
        if not (isinstance(tool_class, type) and issubclass(tool_class, BaseTool)):
            continue

        # Instantiate the tool with the global config and context
        try:
            tool_instance = tool_class(CONFIG, CONTEXT)
        except Exception as e:
            print(f"[ERROR] Failed to instantiate {module_name}.Tool: {e}")
            continue

        # Store the tool
        name = tool_instance.TOOL_NAME
        if name in tools:
            print(f"[WARN] Duplicate tool name '{name}' from {module_name} – skipping")
            continue

        tools[name] = tool_instance

        # Categorise by type
        if tool_instance.TOOL_TYPE == ToolType.FUNCTION:
            function_tools[name] = tool_instance
        elif tool_instance.TOOL_TYPE == ToolType.RESOURCE:
            resource_tools[name] = tool_instance
        else:
            prompt_only_tools[name] = tool_instance

# Run discovery
if __package__ is None:
    raise AttributeError("__package__ is none")
discover_tools(TOOLS_DIR, __package__)

# Build the MCP_PROMPT (with sections)
MCP_PROMPT_LINES = ["\n### You have the following tools available:"]

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

# Exposed lists and call functions
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

# Package exports
__all__ = [
    "tools",
    "MCP_PROMPT",
    "COMMAND_NAMES",
    "FUNCTION_NAMES",
    "RESOURCE_NAMES",
    "call_tool",
    "get_openai_function_definitions",
]

print(f"[INFO] Loaded {len(tools)} tools ({len(function_tools)} callable, "
      f"{len(resource_tools)} resources, {len(prompt_only_tools)} prompt-only)")
