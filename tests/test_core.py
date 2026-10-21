
from process.app_core.conversation.chat import ChatService
from process.app_core.kernel.messages import ChatMessage, ModelResponse, ToolCall
from process.app_core.tools.registry import ToolRegistry


class FakeProvider:
    def __init__(self): self.calls = 0
    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(ChatMessage("assistant", tool_calls=[ToolCall("1", "add", {"a": 2, "b": 3})]))
        return ModelResponse(ChatMessage("assistant", "done"))
    def count_tokens(self, messages): return sum(len(m.content.split()) for m in messages)
    def close(self): pass


class AddTool:
    TOOL_NAME = "add"
    TOOL_DESCRIPTION = "Add two numbers"
    def execute(self, a: int, b: int): return a + b


def test_tool_loop():
    registry = ToolRegistry()
    registry.register_local(AddTool())
    service = ChatService(FakeProvider(), system_prompt="test", tool_registry=registry)
    result = service.respond("calculate")
    assert result.message.content == "done"


def test_registry_schema():
    registry = ToolRegistry()
    registry.register_local(AddTool())
    definition = registry.definitions()[0]
    assert definition["function"]["parameters"]["required"] == ["a", "b"]
