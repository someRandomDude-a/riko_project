from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional
import inspect


class ToolType(str, Enum):
    FUNCTION = "function"
    RESOURCE = "resource"
    PROMPT_ONLY = "prompt_only"


class BaseTool(ABC):
    # Metadata (override in subclasses)
    TOOL_NAME: str = ""
    TOOL_DESCRIPTION: str = ""
    TOOL_TYPE: ToolType = ToolType.FUNCTION
    MCP_PROMPT: str = ""   # Detailed usage instructions for the LLM

    def __init__(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        self.config = config
        self.context = context or {}
        self._setup()

    def _setup(self):
        """Override for initialization (e.g., API key validation)."""
        pass

    # Public entry point - not meant to be overridden
    def execute(self, **kwargs) -> Any:
        """
        Extract arguments and call the tool's _call method.
        This is the only method the external caller (loader) should invoke.
        """
        # Get the signature of _call (the tool's implementation)
        sig = inspect.signature(self._call)
        params = sig.parameters

        # Build a dict of arguments we'll pass to _call
        call_args = {}
        missing = []

        for name, param in params.items():
            if name == "self":
                continue
            # If the argument was provided in kwargs, use it
            if name in kwargs:
                call_args[name] = kwargs[name]
            # Else if it has a default, use that
            elif param.default is not inspect.Parameter.empty:
                call_args[name] = param.default
            else:
                missing.append(name)

        if missing:
            raise TypeError(f"Missing required arguments for {self.TOOL_NAME}: {', '.join(missing)}")

        # Call the tool's implementation with the extracted arguments
        return self._call(**call_args)

    # The actual tool logic - to be overridden by subclasses
    @abstractmethod
    def _call(self, **kwargs) -> Any:
        """
        Implement the tool logic here with explicit parameters and type hints.
        Example:
            def _call(self, a: int, b: int) -> int:
                return a + b
        """
        pass

    # MCP-compatible tool definition
    def to_mcp_definition(self) -> Dict[str, Any]:
        """Generate a tool definition from the _call signature."""
        sig = inspect.signature(self._call)
        params = sig.parameters
        properties = {}
        required = []

        for name, param in params.items():
            if name == "self":
                continue
            # Determine JSON Schema type from type annotation
            type_hint = param.annotation
            if type_hint is inspect.Parameter.empty:
                json_type = "string"
            else:
                # Map Python types to JSON Schema
                if type_hint in (str, bytes):
                    json_type = "string"
                elif type_hint is int:
                    json_type = "integer"
                elif type_hint is float:
                    json_type = "number"
                elif type_hint is bool:
                    json_type = "boolean"
                elif type_hint is list:
                    json_type = "array"
                elif type_hint is dict:
                    json_type = "object"
                else:
                    json_type = "string"

            properties[name] = {
                "type": json_type,
                "description": f"Parameter: {name}"
            }

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    # LLM-friendly prompt builder
    def build_prompt_context(self) -> str:
        """Generate an LLM-readable prompt section for this tool."""
        if self.MCP_PROMPT:
            return self.MCP_PROMPT

        # Auto-generate from metadata and signature
        lines = [f"{self.TOOL_NAME}: {self.TOOL_DESCRIPTION}"]
        lines.append(f"  Type: {self.TOOL_TYPE.value}")
        sig = inspect.signature(self._call)
        params = sig.parameters
        if params:
            lines.append("  Parameters:")
            for name, param in params.items():
                if name == "self":
                    continue
                default = "" if param.default is inspect.Parameter.empty else f" (default: {param.default})"
                lines.append(f"    - {name}{default}")
        return "\n".join(lines)

    def __repr__(self):
        return f"<{self.TOOL_TYPE.value}:{self.TOOL_NAME}>"