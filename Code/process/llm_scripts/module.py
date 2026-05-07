import json
from openai import OpenAI
from datetime import datetime
import pathlib
import os
import tempfile
from collections import deque

from process.llm_scripts.Memory_system.long_term_memory import get_RAG_context, add_message_to_memory 
from process.llm_scripts.utils import get_llm_token_length
from process.common.config import char_config


# === Utility: Load and Save Chat History ===
_HISTORY_FILE = char_config['history_file']
_HISTORY_FILE = pathlib.Path(_HISTORY_FILE).resolve()
_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

def _load_history() -> deque[dict] | None:
    if _HISTORY_FILE.is_file():
        try:
            with open(_HISTORY_FILE, "r") as f:
                hist = json.load(f)
                if isinstance(hist, list):
                    return hist
        except json.JSONDecodeError:
            print("[WARN] History file is corrupted. Starting fresh history.")
    return None

_history = _load_history()

def _save_history():
    with tempfile.NamedTemporaryFile(
        "w",
        dir=_HISTORY_FILE.parent,
        delete=False,
        encoding="utf-8"
    ) as tmp:
        json.dump(_history, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = pathlib.Path(tmp.name)

    temp_path.replace(_HISTORY_FILE)



_SYSTEM_INSTRUCTIONS = char_config['presets']['default']['system_prompt']  
_ASSISTANT_NAME = char_config['presets']['default']['name']
def Riko_Response(user_input: str, time_now = None):
    """
    Handles user input, manages context, queries memory, and returns model output.
    Must always include a speaker name in format:
    `speaker_name: message_text` 
    """
    if time_now == None:
        time_now = datetime.now().isoformat(timespec='minutes')

    global _history
    handle_rolling_window()
    memory_text = get_RAG_context(user_input.split(":", 1)[1].strip())
    header = """
### Conversation History
"""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": _SYSTEM_INSTRUCTIONS + memory_text + header
                }
            ]
        }
    ]

    if _history:
        messages.extend(_history)
    # Add new user message
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": f'[{time_now}] {user_input}'}
        ],
        "tokens": get_llm_token_length(f'[{time_now}] {user_input}')
    })
    response = _call_llm_api(messages)


    response_text = response.output_text.strip()
    if not response_text:
        response_text = f"{_ASSISTANT_NAME}: ..."

    #Remove the AI parroted timestamp
    elif response_text.startswith("[") and "]" in response_text:
        _, _, response_text = response_text.partition("]")
    
    response_text = response_text.strip()
    #Ensure speaker name is always included properly
    if not response_text.startswith(f"{_ASSISTANT_NAME}:"):
        response_text = f"{_ASSISTANT_NAME}: " + response_text


    # Save assistant's response
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": f"[{time_now}] {response_text}"}
        ],
        "tokens": get_llm_token_length(f"[{time_now}] {response_text}")
    })
    _history = messages[1:]# Skip the first element as it's the system setup message and RAG memories
    _save_history() 
    
    reasoning = f"Could not fetch reasoning"
    try:
        for item in response.output:
            if getattr(item, "type", "") == "reasoning":
                content_list = getattr(item, "content", [])
                if content_list:
                    reasoning = content_list[0].text
                break
    except Exception:
        pass
    final_response = response_text.removeprefix(f"{_ASSISTANT_NAME}:").strip()
    return final_response, reasoning


# === Handle context overflow ===
_MAX_HISTORY_TOKENS = char_config['presets']['default']['model_params']['context_window_token_limit']
_SYSTEM_INSTRUCTIONS_TOKENS = get_llm_token_length(_SYSTEM_INSTRUCTIONS)
def handle_rolling_window():
    """When context window is full, archive old messages into long-term memory."""

    if not _history:
        print("[INFO] No history to manage.")
        return
    
    token_count = _SYSTEM_INSTRUCTIONS_TOKENS
    for msg in _history:
        if "tokens" not in msg:
            msg["tokens"] = get_llm_token_length(msg["content"][0]["text"])
        token_count += msg["tokens"]

    if token_count <= _MAX_HISTORY_TOKENS:
        return
    
    while token_count >= _MAX_HISTORY_TOKENS or _history[0]["role"] != "user":
        # Pop oldest non-system message
        dropped_message = _history.pop(0)
        token_count -= dropped_message["tokens"]

        if dropped_message["role"] == "system":
            continue

        message_tokens = dropped_message["tokens"]
        message = dropped_message["content"][0]["text"]
        message_time, _, message_text = message.partition("]")
        
        message_time, message_text = message_time[1:], message_text.strip()

        # ensure every message has a valid timestamp
        try:
            datetime.fromisoformat(message_time)
        except ValueError:
            message_time = datetime.now().isoformat(timespec="minutes")

        add_message_to_memory(message_text, message_time, message_tokens, _history)

    print(f"[INFO] Context window managed. Updated history saved. final history token count: {token_count}")
    _save_history()


_client = OpenAI(api_key=char_config['api_key'], base_url=char_config['base_url'])
_MODEL = char_config['model']
_MAX_OUTPUT_TOKENS = char_config['presets']['default']['model_params']['max_output_tokens']
_TEMPERATURE = char_config['presets']['default']['model_params']['temperature']
def _call_llm_api(messages):
    """Core LLM Call"""
    response = _client.responses.create(
        model=_MODEL,
        input=messages,
        max_output_tokens= _MAX_OUTPUT_TOKENS,
        temperature=_TEMPERATURE,
        stream=False,
        text={
            "format": {
            "type": "text"
            }
        },
        store=False,
    )
    return response

