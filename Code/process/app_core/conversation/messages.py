"""Shim: moved to kernel/messages.py. Old imports keep working, with the same objects, for one release; import from kernel."""
from ..kernel.messages import ChatMessage, ModelResponse, Role, ToolCall, ToolResult, conversation_sections  # noqa: F401
