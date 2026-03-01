# OpenAI tool calling with history 
### Uses a sample function
import yaml
import json
import os
from openai import OpenAI
from typing import cast
from process.llm_funcs.Memory_system.long_term_memory import load_faiss_index, load_memory_store, add_embeddings_to_faiss, save_faiss_index, save_memory_store, get_relevant_memories, get_embedding 
import numpy as np
from datetime import datetime



with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)

client = OpenAI(api_key="", base_url=char_config['base_url'])

# Constants
HISTORY_FILE = char_config['history_file']
MODEL = char_config['model']
SYSTEM_PROMPT =  [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": char_config['presets']['default']['system_prompt'] 
                },
            ]
        }
    ]

MAX_HISTORY_TOKENS = char_config['presets']['default']['model_params']['context_window_token_limit']

# === Initialize RAG Memory System ===
embedding_dim = char_config['RAG_params']['text_embedding_dim']
memory_store = load_memory_store()
faiss_index = load_faiss_index(embedding_dim)

if faiss_index.ntotal == 0:
    add_embeddings_to_faiss(faiss_index, memory_store)
    save_faiss_index(faiss_index)


# === Utility: Load and Save Chat History ===
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                hist = json.load(f)
                if hist:
                    return hist
        except json.JSONDecodeError:
            print("[WARN] History file is corrupted. Starting fresh history.")
    return None


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# === Send old messages to long-term memory ===
def add_message_to_memory(message_text, message_time=datetime.now().isoformat(timespec='minutes')):
    """Adds a message to long-term memory with default importance."""
    global memory_store, faiss_index
    
    # === Skip duplicates ===
    if any(m['text'] == message_text for m in memory_store):
        print("[Memory] Duplicate message detected — skipping.")
        return
    
    # format new memory entry
    new_memory = {
        "text": message_text,
        "importance_score": char_config['RAG_params']['default_importance_score'] ,
        "timestamp": message_time,
        "access_count": 0,
        "detailed": True,
    }
    memory_store.append(new_memory)
    
    embedding = np.array(get_embedding(message_text), dtype=np.float32).reshape(1, -1)
        
    faiss_index.add(embedding)
    save_memory_store(memory_store)
    save_faiss_index(faiss_index)


# === Handle context overflow ===
def handle_rolling_window(time_exceeded):
    """When context window is full, archive old messages into long-term memory."""

    # print(f"[INFO] Context window exceeded at {time_exceeded}. Archiving old messages.")
    history = load_history()
    if not history:
        print("[INFO] No history to manage.")
        return
    while True:
        approx_token_count = sum(len(msg["content"][0]["text"].split()) for msg in history)
        if approx_token_count <= MAX_HISTORY_TOKENS:
            break

        # Pop oldest non-system message
        dropped_message = history.pop(0)
        if dropped_message["role"] == "system":
            continue

        message = ":".join([dropped_message["role"], dropped_message["content"][0]["text"]])
        message_text = message.split(" timestamp:", 1)[0]
        message_time = message.rsplit(" timestamp:", 1)[-1].strip()

        # ensure every message has a valid timestamp
        try:
            datetime.fromisoformat(message_time)
        except ValueError:
            message_time = datetime.now().isoformat(timespec="minutes")

        add_message_to_memory(message_text,message_time)
    print("[INFO] Context window managed. Updated history saved. final history token count: ",approx_token_count)
    save_history(history)


# === Retrieve relevant memories for new prompt ===
def get_RAG_context(user_input):
    """
      Query long-term memory for related past experiences.
    """
    print("fetching relevant long term memories")
    ranked_memories, _, _ = get_relevant_memories(user_input, memory_store, faiss_index)
    top_k = char_config['RAG_params']['default_top_k']    
    top_memories = ranked_memories[:top_k]  # Top-K relevant memories
    if not top_memories:
        return ""
    
    memory_snippets = ",".join([f"[({m['text']} timestamp:{m['timestamp']})" for m in top_memories])
    return f"Relevant memories:[{memory_snippets}]"


# === Core LLM call ===
def call_llm_api(messages):
    response = client.responses.create(
        model=MODEL,
        input=messages,
        max_output_tokens= char_config['presets']['default']['model_params']['max_output_tokens'],
        stream=False,
        text={
            "format": {
            "type": "text"
            }
        },
    )
    return response

def Riko_Response(user_input, time_now = datetime.now().isoformat(timespec='minutes')):
    """
    Handles user input, manages context, queries memory, and returns model output.
    """

    handle_rolling_window(time_now)
    messages = SYSTEM_PROMPT[:]  # Start with system prompt

    memory_text = get_RAG_context(user_input)
    
    if memory_text:
        messages.append({
            "role": "system",
            "content": [{"type": "input_text", "text": memory_text}]
        })    
    
    history = load_history()
    if history:
        messages.extend(history)

    # Add new user message
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": user_input + " timestamp:" + time_now}
        ]
    })
    
    # print("\n\nFinal prompt: ", messages, "\n\n") # This print command can cause a lot of lag, only use for debugging
    response = call_llm_api(messages)
    
    #This is basically us replacing the AI's parroted timestamp with an accurate timestamp
    response = response.output_text.rsplit("timestamp:",1)[0].strip()
    # Save assistant's response
    messages.append({
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": response + " timestamp:" + time_now}
    ]    
    })

    # Remove the system prompt from the messages (splice the list directly)
    messages = messages[2:]  # Skip the first element as it's the system setup message
    # Change from 1 to 2 to also skip RAG system messages from being appended to history

    save_history(messages) # The part where we actually save the history from llm response
    print(response.output_text)
    return response
