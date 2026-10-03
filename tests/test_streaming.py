from types import SimpleNamespace

from process.app_core.inference.providers import assemble_stream
from process.app_core.emotion.compat import compatible_engine


def test_fragmented_tools_and_text():
    received = []
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "test", "arguments": '{"x":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '1}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"total_tokens": 9}},
    ]
    response = assemble_stream(chunks, received.append)
    assert received == ["Hello"]
    assert response.message.tool_calls[0].arguments == {"x": 1}
    assert response.usage == {"total_tokens": 9}


def test_julia_restores_original_forward_only_when_needed():
    original = lambda: "standard"
    encoder = SimpleNamespace(_julia_original_forward=original, forward=lambda: "optimized")
    engine = SimpleNamespace(model=SimpleNamespace(encoder=encoder))
    assert compatible_engine(engine) is engine
    assert encoder.forward() == "standard"
    assert engine.encoder_specialized is False
