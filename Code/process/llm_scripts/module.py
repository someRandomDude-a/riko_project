from __future__ import annotations

import json
import re
from datetime import datetime
import pathlib
import os
import tempfile

from process.common.config import char_config
from process.llm_scripts.llama_server import (
    call_llm_api,
    stream_chat,
    get_llm_token_length,
)
from process.llm_scripts.MCP_Tools import MCP_PROMPT, call_tool, FUNCTION_NAMES
from process.llm_scripts.Memory_system.long_term_memory import (
    get_RAG_context,
    add_message_to_memory,
)


# Chat history persistence

_HISTORY_FILE = pathlib.Path(char_config["history_file"]).absolute()
_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_history():
    if _HISTORY_FILE.is_file():
        try:
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                hist = json.load(f)
                if isinstance(hist, list):
                    return hist
        except json.JSONDecodeError:
            print("[WARN] History file is corrupted. Starting fresh history.")
    return []


_history = _load_history()


def _save_history():
    with tempfile.NamedTemporaryFile(
        "w", dir=_HISTORY_FILE.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(_history, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = pathlib.Path(tmp.name)
    temp_path.replace(_HISTORY_FILE)


# Static config

_SYSTEM_PROMPT = char_config["presets"]["default"]["system_prompt"] + MCP_PROMPT
_ASSISTANT_NAME = char_config["presets"]["default"]["name"]
_MAX_TOOL_ITERATIONS = 5

_THINK_RE = re.compile(r"^(?:.*?\n)*?\s*", re.DOTALL)  # placeholder
_THINK_RE = re.compile(r"((?:<\|im_start\|>)*\s*)?(?:<think>)?(.*?)(?:</think>)?", re.DOTALL)
# Robust cleaner:
_THINK_BLOCK = re.compile(r"<think>.*?</think>|<\|begin▁of▁thinking\|>.*?<\|end▁of▁thinking\|>",
                          re.DOTALL)


def _strip_thinking(text: str) -> tuple[str, str]:
    """Strip `` / `` blocks. Returns (clean_text, joined_thinking)."""
    chunks = _THINK_BLOCK.findall(text)
    if not chunks:
        return text, ""
    thinking = "\n---\n".join(c.strip() for c in chunks if c.strip())
    clean = _THINK_BLOCK.sub("", text).strip()
    return clean, thinking


# Streaming tool-call extraction

def _extract_tool_calls(response) -> list:
    """
    Extract function_call items from a Responses-API or Chat-Completions
    response. Works with both shapes because we only read standard fields.
    """
    out = []
    # Responses API: list of items
    if hasattr(response, "output") and response.output is not None:
        for item in response.output:
            if getattr(item, "type", "") == "function_call":
                out.append(item)
        return out

    # Chat Completions streaming chunks: collected list of choices
    if isinstance(response, list):
        try:
            msg = response[-1].choices[0].message
            if msg and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    class _Wrap:
                        type = "function_call"
                        name = tc.function.name
                        arguments = tc.function.arguments
                        call_id = tc.id
                    out.append(_Wrap())
        except Exception:
            pass
        return out

    # Chat Completions non-streaming
    try:
        msg = response.choices[0].message
        if msg and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                class _Wrap:
                    type = "function_call"
                    name = tc.function.name
                    arguments = tc.function.arguments
                    call_id = tc.id
                out.append(_Wrap())
    except Exception:
        pass
    return out


# Public entry point

def _noop(*_a, **_kw): pass


def llm_response(
    user_message: str,
    user_name: str,
    time_now: str | None = None,
    on_token=None,
    on_sentence=None,
) -> tuple[str, str]:
    """
    Streaming LLM response with tool-call loop.

    Args:
        user_message: the user's input text
        user_name:    display name (e.g., "Senpai")
        time_now:     ISO timestamp (auto-generated if omitted)
        on_token:     optional callback(str) called per text delta
        on_sentence:  optional callback(str) called per complete sentence

    Returns: (assistant_text, reasoning_text)
    """
    on_token = on_token or _noop
    on_sentence = on_sentence or _noop

    if time_now is None:
        time_now = datetime.now().isoformat(timespec='minutes')

    # Rolling window + memory retrieval
    global _history
    handle_rolling_window()
    memory_text = get_RAG_context(user_message)
    header = "\n### Conversation History\n"

    # Build the prompt
    messages: list = [
        {
            "role": "system",
            "content": [{"type": "input_text",
                          "text": _SYSTEM_PROMPT + memory_text + header}],
        }
    ]
    if _history:
        messages.extend(_history)
    messages.append({
        "role": "user",
        "content": [{"type": "input_text",
                      "text": f'[{time_now}] {user_name}: {user_message}'}],
        "tokens": get_llm_token_length(f'[{time_now}] {user_name}: {user_message}'),
    })

    # Tool definitions (auto-built from MCP_Tools discovery)
    from process.llm_scripts.MCP_Tools import get_openai_function_definitions
    tools = get_openai_function_definitions() or None

    # First pass — stream text. If the model wants tools, fall back to
    final_response = stream_chat(messages, on_token=on_token, tools=tools)
    tool_calls = _extract_tool_calls(final_response) if final_response is not None else []

    # Tool-call iterations (non-streaming; latency already paid on pass 1)
    for _ in range(_MAX_TOOL_ITERATIONS):
        if not tool_calls:
            break
        # Append each call to the conversation
        for call in tool_calls:
            messages.append(_call_to_message(call))

        # Execute each tool and append the result
        for call in tool_calls:
            try:
                args = json.loads(call.arguments) if isinstance(call.arguments, str) else dict(call.arguments)
            except json.JSONDecodeError:
                args = {}
            try:
                result = call_tool(call.name, **args)
                output_str = str(result)
            except KeyError:
                output_str = f"Error: unknown tool '{call.name}'"
            except Exception as e:  # noqa: BLE001
                output_str = f"Error executing {call.name}: {e}"

            messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output_str,
            })

        # Re-call without streaming — tool loop is not on the hot path
        response = call_llm_api(messages, tools=tools)
        tool_calls = _extract_tool_calls(response)

    # Extract final text
    raw = _response_output_text(final_response if tool_calls is None or not tool_calls else response)
    if not raw:
        # Empty stream (e.g. network glitch). Final call without tools.
        response = call_llm_api(messages, tools=None)
        raw = _response_output_text(response)

    clean, reasoning = _strip_thinking(raw.strip())

    if not clean:
        clean = "..."
    elif clean.startswith("[") and "]" in clean:
        _, _, clean = clean.partition("]")
    clean = clean.strip()
    if not clean.startswith(f"{_ASSISTANT_NAME}:"):
        clean = f"{_ASSISTANT_NAME}: " + clean

    messages.append({
        "role": "assistant",
        "content": [{"type": "output_text", "text": f"[{time_now}] {clean}"}],
        "tokens": get_llm_token_length(f"[{time_now}] {clean]"),
    })
    _history = messages[1:]
    _save_history()

    # Sentence segmentation — fire callback per sentence if provided
    final_reply = clean.removeprefix(f"{_ASSISTANT_NAME}:").strip()
    if on_sentence and final_reply:
        # Final emit for any text accumulated after the streaming pass
        for piece in _split_into_sentences(final_reply):
            on_sentence(piece)

    if not reasoning:
        reasoning = "Could not fetch reasoning"

    return final_reply, reasoning


def _response_output_text(response) -> str:
    """Read output text from a Responses-API or Chat-Completions response."""
    if response is None:
        return ""
    # Responses API streaming final carries `response.output_text`
    if hasattr(response, "output_text"):
        return response.output_text or ""
    # Chat Completions
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _call_to_message(call) -> dict:
    """Convert a function_call item to an input message the SDK will accept."""
    return {
        "type": "function_call",
        "id": getattr(call, "call_id", ""),
        "call_id": getattr(call, "call_id", ""),
        "name": getattr(call, "name", ""),
        "arguments": getattr(call, "arguments", ""),
    }


def _split_into_sentences(text: str) -> list[str]:
    cfg = char_config.get("sentence_streamer", {})
    terms = cfg.get("terminators", [". ", "! ", "? "])
    import re as _re
    pattern = "(" + "|".join(_re.escape(t) for t in sorted(terms, key=len, reverse=True)) + ")"
    parts = _re.split(pattern, text)
    out, buf = [], ""
    for p in parts:
        buf += p
        if p and any(p.endswith(t) for t in terms):
            s = buf.strip()
            if s:
                out.append(s)
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


# Context-window management

_MAX_HISTORY_TOKENS = char_config['presets']['default']['model_params']['context_window_token_limit']
_SYSTEM_INSTRUCTIONS_TOKENS = get_llm_token_length(_SYSTEM_PROMPT)


def handle_rolling_window():
    """
    When context window is full, archive old messages into long-term memory.
    """
    token_count = _SYSTEM_INSTRUCTIONS_TOKENS
    for msg in _history:
        if "tokens" not in msg:
            try:
                text = msg["content"][0]["text"]
                msg["tokens"] = get_llm_token_length(text)
            except Exception:
                msg["tokens"] = 1
        token_count += msg["tokens"]

    if token_count <= _MAX_HISTORY_TOKENS:
        return

    while _history and (token_count >= _MAX_HISTORY_TOKENS or _history[0]["role"] != "user"):
        dropped_message = _history.pop(0)
        token_count -= dropped_message["tokens"]

        if dropped_message["role"] == "system":
            continue

        message_tokens = dropped_message["tokens"]
        try:
            message = dropped_message["content"][0]["text"]
        except Exception:
            continue

        if message.startswith("[") and "]" in message:
            message_time, _, message_text = message.partition("]")
            message_time = message_time[1:]
        else:
            message_time = datetime.now().isoformat(timespec="minutes")
            message_text = message

        try:
            datetime.fromisoformat(message_time)
        except ValueError:
            message_time = datetime.now().isoformat(timespec="minutes")

        add_message_to_memory(message_text, message_time, message_tokens, _history)

    print(f"[INFO] Context window managed. Token count: {token_count}")
