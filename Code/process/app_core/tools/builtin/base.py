from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import inspect


class BaseTool(ABC):
    """A built-in tool. The registry describes it by input_schema (from _call's signature) and runs each call in a
    disposable worker (tools/isolation.py) that imports only the subclass's module, so subclasses keep their imports light."""
    TOOL_NAME: str = ""
    TOOL_DESCRIPTION: str = ""
    CHOICES: Dict[str, list] = {}
    ISOLATED = True  # explicit, not inferred from where the module lives: run in a worker under any import root
    READ_ONLY = False  # declares that it never changes its environment (MCP readOnlyHint), so initiative may call it too

    def __init__(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        self.config = config
        self.context = context or {}
        self._setup()

    def _setup(self):
        """Override for initialization (e.g., API key validation)."""
        pass

    @property
    def input_schema(self) -> Dict[str, Any]:
        from ..schema import signature_schema  # here, not at import: the worker loads only base.py and the tool's module
        return signature_schema(self._call, self.CHOICES)

    def worker_request(self) -> Dict[str, Any]:
        """What tools/worker.py rebuilds this tool from in its own process."""
        return {'module': type(self).__module__, 'class': type(self).__name__, 'config': self.config, 'context': self.context}

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

    def __repr__(self):
        return f"<tool:{self.TOOL_NAME}>"
