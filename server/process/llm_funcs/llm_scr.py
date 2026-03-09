import yaml
import json
import os
from openai import OpenAI
from process.llm_funcs.Memory_system.long_term_memory import get_RAG_context, add_message_to_memory 
from datetime import datetime
from transformers import AutoTokenizer

with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)

# Constants
HISTORY_FILE = char_config['history_file']
MODEL = char_config['model']
SYSTEM_INSTRUCTIONS = char_config['presets']['default']['system_prompt']  
MAX_HISTORY_TOKENS = char_config['presets']['default']['model_params']['context_window_token_limit']
MAX_OUTPUT_TOKENS = char_config['presets']['default']['model_params']['max_output_tokens']
TEMPERATURE = char_config['presets']['default']['model_params']['temperature']

client = OpenAI(api_key=char_config['api_key'], base_url=char_config['base_url'])
tokenizer = AutoTokenizer.from_pretrained(char_config['tokenizer_model'])


# === Utility: Load and Save Chat History ===
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                hist = json.load(f)
                if isinstance(hist, list):
                    return hist
        except json.JSONDecodeError:
            print("[WARN] History file is corrupted. Starting fresh history.")
    return None

history = load_history()

def save_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_llm_token_length(text : str | list[str]) -> int | list[int]:
    """Returns the number of tokens in a given string, or an array of lengths for each string in a string array"""
    lengths = tokenizer(text, add_special_tokens = False, return_length=True)["length"]
    if isinstance(text, str):
        return lengths[0]
    return lengths
# === Handle context overflow ===
def handle_rolling_window():
    """When context window is full, archive old messages into long-term memory."""
    if not history:
        print("[INFO] No history to manage.")
        return
    
    token_count = 0 
    for msg in history:
        if "tokens" not in msg:
            msg["tokens"] = get_llm_token_length(msg["content"][0]["text"])
        token_count += msg["tokens"]

    if token_count <= MAX_HISTORY_TOKENS:
        return
    
    while token_count >= MAX_HISTORY_TOKENS:
        # Pop oldest non-system message
        dropped_message = history.pop(0)
        token_count -= dropped_message["tokens"]

        if dropped_message["role"] == "system":
            continue

        message = dropped_message["content"][0]["text"]
        message_text = message.split(" timestamp:", 1)[0]
        message_time = message.rsplit(" timestamp:", 1)[-1].strip()

        # ensure every message has a valid timestamp
        try:
            datetime.fromisoformat(message_time)
        except ValueError:
            message_time = datetime.now().isoformat(timespec="minutes")

        add_message_to_memory(message_text,message_time)
    print("[INFO] Context window managed. Updated history saved. final history token count: ",token_count)
    save_history()



def call_llm_api(messages):
    """Core LLM Call"""
    response = client.responses.create(
        model=MODEL,
        input=messages,
        max_output_tokens= MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
        text={
            "format": {
            "type": "text"
            }
        },
        store=False,
    )
    return response


def Riko_Response(user_input: str, time_now = datetime.now().isoformat(timespec='minutes')):
    """
    Handles user input, manages context, queries memory, and returns model output.
    """
    global history

    handle_rolling_window()
    memory_text = get_RAG_context(user_input)
    
    header = """
### Conversation History
"""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": SYSTEM_INSTRUCTIONS + memory_text + header
                }
            ]
        }
    ]

    if history:
        messages.extend(history)
    # Add new user message
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": user_input + " timestamp:" + time_now}
        ],
        "tokens": get_llm_token_length(user_input + " timestamp:" + time_now)
    })
    
    response = call_llm_api(messages)
    
    # replace the AI's parroted timestamp with an accurate timestamp
    response = response.output_text.rsplit("timestamp:")[0].strip()
    if not response.startswith("(Riko)"):
        response = "(Riko) " + response
    # Save assistant's response
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": response + " timestamp:" + time_now}
        ],
        "tokens": get_llm_token_length(response + " timestamp:" + time_now)
    })
    messages = messages[1:]  # Skip the first element as it's the system setup message and RAG memories
    history = messages
    save_history() 
    return response