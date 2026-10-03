from __future__ import annotations

from dataclasses import dataclass, field
import json
from datetime import datetime
from dataclasses import replace
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: Any
    is_error: bool = False


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    timestamp: str | None = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    context_kind: str | None = None # Internal only; never serialized to providers/history.

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in self.tool_calls
            ]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    def as_record(self):
        return {**self.as_dict(), 'timestamp': self.timestamp}


def conversation_sections(messages, gap_seconds=300):
    """Render sparse time markers without mutating messages or API metadata.

    Tools/system messages neither create sections nor reset conversational gaps.
    Legacy undated messages break time continuity rather than inventing dates.
    """
    result, previous, seen = [], None, False
    for message in messages:
        if message.role not in {'user', 'assistant'}:
            result.append(message)
            continue
        try:
            current = datetime.fromisoformat(message.timestamp) if message.timestamp else None
            if current and current.tzinfo is None: current = None
        except (ValueError, TypeError): current = None
        marker = None
        if current is None:
            if not seen or previous is not None: marker = '[timestamp unavailable]'
        elif previous is None or (current - previous).total_seconds() >= gap_seconds or current < previous:
            marker = f'[{current.astimezone().strftime("%Y-%m-%dT%H:%M")}]'
        result.append(replace(message, content=f'{marker}\n{message.content}') if marker else message)
        previous, seen = current, True
    return result


@dataclass
class ModelResponse:
    message: ChatMessage
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    context_messages: list[ChatMessage] | None = None
