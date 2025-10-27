# OpenAI tool calling with history 
### Uses a sample function
import yaml
import gradio as gr
import json
import os
from openai import OpenAI

from process.llm_funcs.Memory_system.long_term_memory import *

with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)

client = OpenAI(api_key=char_config['OPENAI_API_KEY'] , base_url=char_config['base_url']
                )

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
                }
            ]
        }
    ]

MAX_HISTORY_TOKENS = char_config['presets']['default']['model_params']['context_window_token_limit']  # Approximate context window limit, adjust for your model

# === Initialize RAG Memory System ===
embedding_dim = char_config['RAG_params']['text_embedding_dim']  # Sentence-BERT ('all-MiniLM-L6-v2') output dimension
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
def add_message_to_memory(message_text):
    """Adds a message to long-term memory with default importance."""
    global memory_store, faiss_index
    new_memory = {
        "text": message_text,
        "importance_score": char_config['RAG_params']['default_importance_score'] ,
        "timestamp": time.time(),
        "access_count": 0,
        "detailed": True,
    }
    memory_store.append(new_memory)
    embedding = np.array([get_embedding(message_text)]).astype("float32")
    faiss_index.add(embedding)
    save_memory_store(memory_store)
    save_faiss_index(faiss_index)


# === Handle context overflow ===
def handle_rolling_window(time_exceeded):
    return
    """When context window is full, archive old messages into long-term memory."""
    print(f"[INFO] Context window exceeded at {time_exceeded}. Archiving old messages.")
    history = load_history()
    while True:
        token_count = sum(len(msg["content"][0]["text"].split()) for msg in history)
        if token_count <= MAX_HISTORY_TOKENS:
            break
        # Pop oldest non-system message and store it
        dropped_message = history.pop(1)  # Keep system prompt at index 0
        message_text = dropped_message["content"][0]["text"]
        add_message_to_memory(message_text)
    print("[INFO] Context window managed. Updated history saved.",token_count)
    save_history(history)


# === Retrieve relevant memories for new prompt ===
def get_additional_memory(user_input):
    """Query long-term memory for related past experiences."""
    ranked_memories, _, _ = get_relevant_memories(user_input, memory_store, faiss_index)
    top_k = char_config['RAG_params']['default_top_k']    
    top_memories = ranked_memories[:top_k]  # Top-K relevant memories
    if not top_memories:
        return ""
    memory_snippets = "\n".join([m["text"] for m in top_memories])
    return f"Relevant long-term memories:\n{memory_snippets}"


# === Core LLM call ===
def get_riko_response_no_tool(messages):

    # Call OpenAI with system prompt + history
    response = client.responses.create(
        model=MODEL,
        input=messages,
        temperature= char_config['presets']['default']['model_params']['temperature'],
        top_p= char_config['presets']['default']['model_params']['top_p']  ,
        max_output_tokens= char_config['presets']['default']['model_params']['max_output_tokens'],
        stream=False,
        text={
            "format": {
            "type": "text"
            }
        },
    )

    return response


def llm_response(user_input):
    """Handles user input, manages context, queries memory, and returns model output."""

    handle_rolling_window(time.time())

    
    messages = SYSTEM_PROMPT[:]  # Start with system prompt
    
    # 🧠 Retrieve relevant RAG context
    memory_text = get_additional_memory(user_input)
    if memory_text:
        messages.append({
            "role": "system",
            "content": [{"type": "input_text", "text": memory_text}]
        })
        
    
    history = load_history()
    if history:
        messages.extend(history)# Load history excluding system prompt
    # Add new user message
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": user_input}
        ]
    })


    riko_test_response = get_riko_response_no_tool(messages)
    print("\n\nRiko Response: ", messages, "\n\n")

    # Send request to LLM
    riko_response = get_riko_response_no_tool(messages)
    # print("\nRiko Response:", riko_response.output_text, "\n")

    # Save assistant's response
    messages.append({
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": riko_test_response.output_text}
    ]
    })
    # Remove the system prompt from the messages (splice the list directly)
    messages = messages[1:]  # Skip the first element as it's the system setup message
    # Comment out the above line and uncomment the line below to also skip RAG memory system message.
    # messages = messages[2:] to also skip the RAG memory system message.
    save_history(messages)
    print(riko_test_response.output_text)
    return riko_test_response.output_text


if __name__ == "__main__":
    print('running main')