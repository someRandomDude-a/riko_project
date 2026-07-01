import pathlib
from importlib import import_module

# global configuration
CONFIG = {
    "WEATHER_API_KEY": "abc123",
    "DB_CONNECTION_STRING": "sqlite:///local.db",
}

# shared context for tools to exchange state if needed
CONTEXT = {}

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
tools = {}
MCP_PROMPT_LINES = ["You have the following tools available to you:"]

for file_path in TOOLS_DIR.glob("*.py"):
    if file_path.name == "__init__.py":
        continue

    module_name = file_path.stem
    try:
        module = import_module(f".{module_name}", package=__package__)
    except Exception as e:
        print(f"[ERROR] Could not import {module_name}: {e}")
        continue

    # initiaalize tools
    if hasattr(module, "initialize") and callable(module.initialize):
        try:
            # pass config and the shared context
            module.initialize(CONFIG, CONTEXT)
        except Exception as e:
            print(f"[ERROR] Initialization failed for tool '{module_name}': {e}")
            continue

    # extract the standard interface
    name = getattr(module, "TOOL_NAME", module_name)
    desc = getattr(module, "TOOL_DESCRIPTION", "")
    execute_func = getattr(module, "execute", None)
    if not execute_func:
        print(f"[WARNING] {name} has no execute() – skipping")
        continue

    prompt = getattr(module, "MCP_PROMPT", f"{name}: {desc}")

    # store the tools
    tools[name] = {
        "description": desc,
        "execute": execute_func,
        "prompt": prompt,
    }
    MCP_PROMPT_LINES.append(prompt)

MCP_PROMPT = "\n".join(MCP_PROMPT_LINES)
COMMAND_NAMES = list(tools.keys())

def call_tool(name, **kwargs):
    return tools[name]["execute"](**kwargs)