import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import json
import pathlib
from transformers import pipeline
import torch
import yaml
from datetime import datetime
import copy

# === Load Character Configuration ===
with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)

# Constants
MODEL_NAME = char_config['RAG_params']['embedding_model_id'] # Sentence-BERT model name
BART_MODEL_NAME = char_config['RAG_params']['summarization_model_id']   # BART model name for summarization
FAISS_INDEX_PATH = './persistant_memories/faiss_index.index' # File path for FAISS index
MEMORY_STORE_PATH = './persistant_memories/memory_store.json' # File path for memory store
EMBEDDING_DIM = char_config['RAG_params']['text_embedding_dim'] # Embedding dimension from config  
FAISS_DECAY_FACTOR_HIGH = -char_config['RAG_params']['high_importance_decay_factor'] # Decay factor for high-importance memories
FAISS_DECAY_FACTOR_LOW = -char_config['RAG_params']['low_importance_decay_factor'] # Decay factor for low-importance memories 
SUMMARY_MIN_LENGTH = char_config['RAG_params']['summary_min_length'] # Minimum length for summary
SUMMARY_MAX_LENGTH = char_config['RAG_params']['summary_max_length'] # Maximum length for summary
SUMMARY_NUM_BEAMS = char_config['RAG_params']['summary_beam_size'] # Beam search size for summarization
SUMMARY_MAX_TOKENS = char_config['RAG_params']['summary_max_tokens'] # Maximum number of tokens for BART input
MEMORY_CLEANUP_THRESHOLD_DAYS = char_config['RAG_params']['memory_cleanup_threshold'] # Days before memory is eligible for cleanup
MEMORY_IMPORTANCE_THRESHOLD = char_config['RAG_params']['memory_importance_threshold'] # Threshold below which memories are discarded
DEFAULT_MEMORY_IMPORTANCE = char_config['RAG_params']['default_importance_score']

# Load SentenceTransformer model
model = SentenceTransformer(MODEL_NAME)

summarizerPipeline = None

def get_RAG_context(user_input):
    """
      Query long-term memory for related past experiences.
    """
    # print("fetching relevant long term memories....")
    ranked_memories, _, _ = get_relevant_memories(user_input, memory_store, faiss_index)
    top_k = char_config['RAG_params']['default_top_k']    
    top_memories = ranked_memories[:top_k]  # Top-K relevant memories
    if not top_memories:
        return ""
    
    memory_snippets = "\n".join([f"- {m['text']} (timestamp:{m['timestamp']})" for m in top_memories])
    return f"""
### Relevant Memories
These are past interactions that may be relevant.
Use them only if useful.

{memory_snippets}    
"""


def summarize_text(text):
    """summarize memory using BART"""
    print("Summarizer called!")
    global summarizerPipeline
    if summarizerPipeline is None:
        summarizerPipeline = pipeline("summarization",model=BART_MODEL_NAME,torch_dtype=torch.float16)

    
    summary = summarizerPipeline(text,num_beams=SUMMARY_NUM_BEAMS,max_length= SUMMARY_MAX_LENGTH, min_length=SUMMARY_MIN_LENGTH,truncation=True)
    summary_text = ""
    for text in summary:
        summary_text += text['summary_text']
    
    return summary_text


def load_memory_store():
    """Load memorystore from file if it exists, else load defaults"""
    
    memory_store = []
    if pathlib.Path.exists(MEMORY_STORE_PATH):
        with open(MEMORY_STORE_PATH, 'r') as f:
            try:
                memory_store = json.load(f)
                if not isinstance(memory_store, list):
                    raise ValueError("Memory_store is not a list")
                print(f"Loaded {len(memory_store)} memories from file.")

                if len(memory_store) != load_faiss_index().ntotal:
                    index = create_faiss_cpu_index()
                    add_faiss_embeddings(index, memory_store)
                    save_faiss_index(index)

            except (json.JSONDecodeError, ValueError) as e:
                print("[WARN] Memory store file is empty or corrupted", e)
                memory_store = []

    
    
    if memory_store:
        return memory_store
    
    # Fetch default memories
    default_memories = char_config["presets"]["default"]["memories"]

    # Build memory store dynamically
    currentTime = datetime.now().isoformat(timespec='minutes')

    for mem in default_memories:
        memory_store.append({
            "text": mem["text"],
            "importance_score": mem["importance_score"],
            "timestamp": currentTime,
            "access_count": mem.get("access_count", 0),
            "detailed": mem.get("detailed", False),
        })

    print("No memories found, Loaded memory store from YAML.")
    index = create_faiss_cpu_index()
    add_faiss_embeddings(index, memory_store)
    save_faiss_index(index)
    save_memory_store(memory_store)
    return memory_store



def save_memory_store(memory_store):
    """Save memory store to disk"""

    with open(MEMORY_STORE_PATH, 'w') as f:
        json.dump(memory_store, f)
    print(f"Saved {len(memory_store)} memories to file.")

def create_faiss_cpu_index():
    """Create FAISS-CPU index"""
    M = 64  # Number of bi-directional links per node (tradeoff: accuracy vs. memory)
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, M)
    index.hnsw.efConstruction = 200  # Controls build accuracy (higher = better, slower)
    index.hnsw.efSearch = 100         # Controls query accuracy (higher = better, slower)
    print(f"Created HNSWFlat FAISS index (dim={EMBEDDING_DIM}, M={M})")
    return index

def save_faiss_index(index):
    """Save FAISS index to disk"""
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index saved to file.")

def load_faiss_index():
    """Load FAISS index from disk"""
    if pathlib.Path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        index = create_faiss_cpu_index()
    return index

def get_embedding(text: str | list[str]):
    """get the embedding of a text"""
    embedding = model.encode(text, convert_to_numpy=True).astype("float32")
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    return embedding

def add_faiss_embeddings(index, memory_store):
    """add embeddings to FAISS index"""
    if not memory_store:
        return
    texts = [memory['text'] for memory in memory_store]

    embeddings = get_embedding(texts)
    index.add(embeddings)

def query_faiss_cpu(index, query_embedding, k=5):
    """query FAISS-CPU for relevant memories"""
    distances, indices = index.search(query_embedding, k)  # Get top-k indices and distances
    return indices, distances

age_div_factor = 60 * 60 * 24
def get_age_in_days(memory):
    return  (time.time() - datetime.fromisoformat(memory['timestamp']).timestamp()) / age_div_factor

def decay_memory(memory):
    """apply timestamp decay and rank memories"""
    # Decay based on importance: high importance memories decay slower
    decay = np.exp(
        (FAISS_DECAY_FACTOR_HIGH if memory['importance_score'] > 0.8 else FAISS_DECAY_FACTOR_LOW) * get_age_in_days(memory))
    memory['importance_score'] *= decay

    # Transition from detailed to summarized when importance is low
    if memory['importance_score'] < 0.3 and memory['detailed'] and len(memory['text']) > 300:  # Low importance, summarized
        print("summarizing memory: " + memory['text'])
        memory['text'] = summarize_text(memory['text'])
        memory['detailed'] = False
        
    return memory

def cleanup_memory_store(memory_store):
    """
    Function to clean up low-importance memories (optional cleanup process)
    MEMORY_CLEANUP_THRESHOLD_DAYS  # Clean memories older than 30 days
    MEMORY_IMPORTANCE_THRESHOLD  # Discard memories with importance lower than 0.1
    """
    # Filter out low-importance or very old memories
    trimmed_memory_store = [
        memory for memory in memory_store
        if get_age_in_days(memory) < MEMORY_CLEANUP_THRESHOLD_DAYS or memory['importance_score'] >= MEMORY_IMPORTANCE_THRESHOLD
    ]
    if trimmed_memory_store == memory_store:
        return memory_store, load_faiss_index()
    
    index = create_faiss_cpu_index()
    add_faiss_embeddings(index, trimmed_memory_store)
    
    print(f"Cleaned up memory store, {len(trimmed_memory_store)} memories remaining.")
    return trimmed_memory_store, index


def update_memory_importance(memory):
    """Function to update memory importance based on access count"""
    access_boost = 0.1 * (memory['access_count'] ** 0.5)  # Exponential growth for access
    memory['importance_score'] += access_boost
    memory['importance_score'] = min(memory['importance_score'], 1.0)  # Cap at 1.0
    return memory

def get_relevant_memories(prompt, memory_store, index, k=5):
    """retrieve relevant memories based on a prompt, returns K memories"""

    indices, distances = query_faiss_cpu(index, get_embedding(prompt), k)
    
    # Rank memories by importance, decay, and similarity
    ranked_memories = []
    for idx, distance in zip(indices[0], distances[0]):
        memory = copy.copy(memory_store[idx])
        decay = np.exp(FAISS_DECAY_FACTOR_HIGH if memory['importance_score'] > 0.8 else FAISS_DECAY_FACTOR_LOW  * get_age_in_days(memory))
        memory['ranked_score'] = memory['importance_score'] * decay
        memory['similarity_score'] = 1 / (1 + distance)  # Inverse distance for higher similarity
        ranked_memories.append(memory)

    ranked_memories = sorted(ranked_memories, key=lambda x: x['ranked_score'], reverse=True)

    # Update access count and importance
    for idx in indices[0]:
        memory_store[idx]['access_count'] += 1
        update_memory_importance(memory_store[idx])

    # Update memory store with decayed importance    
    memory_store =  [decay_memory(memory) for memory in memory_store]
    return ranked_memories, indices, distances


def add_message_to_memory(message_text : str, message_time : str):
    """Adds a message to long-term memory with default importance."""
    # === Skip tiny messages ===
    if len(message_text) < 25:
        return

    # === Skip duplicates ===
    if any(m['text'] == message_text for m in memory_store):
        print("[Memory] Duplicate message detected — skipping.")
        return
    
    # format new memory entry
    new_memory = {
        "text": message_text,
        "importance_score": DEFAULT_MEMORY_IMPORTANCE ,
        "timestamp": message_time,
        "access_count": 0,
        "detailed": True,
    }
    memory_store.append(new_memory)
    
    embedding = get_embedding(message_text)
        
    faiss_index.add(embedding)
    save_memory_store(memory_store)
    save_faiss_index(faiss_index)

# === Initialize RAG Memory System ===
memory_store = load_memory_store()
faiss_index = load_faiss_index()

if faiss_index.ntotal == 0:
    add_faiss_embeddings(faiss_index, memory_store)
    save_faiss_index(faiss_index)


# Main script execution
if __name__ == "__main__":
    prompt = "Tell me more about yourself?"
    ranked_memories, indices, distances = get_relevant_memories(prompt, memory_store, faiss_index)

    print("Ranked Memories:")
    for memory in ranked_memories:
        print(f"Memory Text: {memory['text']}")
        print(f"Importance Score: {memory['importance_score']:.4f}, Similarity Score: {memory['similarity_score']:.4f}")
        print(f"Decay Score: {memory['ranked_score']:.4f}, Access Count: {memory['access_count']}")
        print("-" * 50)

    

    # Cleanup mem store WIL break yur file, do not use
    memory_store, faiss_index = cleanup_memory_store(memory_store)
    add_message_to_memory("example sentence!!! awesome", datetime.now().isoformat(timespec="minutes"))
    save_memory_store(memory_store)
    save_faiss_index(faiss_index)
