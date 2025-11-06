import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import json
import os
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import yaml

import copy
# === Load Character Configuration ===
with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)



# Constants (now converted to variables)
MODEL_NAME = char_config['RAG_params']['embedding_model_id']  # Sentence-BERT model name
BART_MODEL_NAME = char_config['RAG_params']['summarization_model_id']   # BART model name for summarization
FAISS_INDEX_PATH = 'faiss_index.index'  # File path for FAISS index
MEMORY_STORE_PATH = 'memory_store.json'  # File path for memory store
EMBEDDING_DIM = char_config['RAG_params']['text_embedding_dim']  # Embedding dimension from config  
FAISS_DECAY_FACTOR_HIGH = char_config['RAG_params']['high_importance_decay_factor']  # Decay factor for high-importance memories
FAISS_DECAY_FACTOR_LOW = char_config['RAG_params']['low_importance_decay_factor']   # Decay factor for low-importance memories
SUMMARY_MIN_LENGTH = char_config['RAG_params']['summary_min_length']  # Minimum length for summary
SUMMARY_MAX_LENGTH = char_config['RAG_params']['summary_max_length'] # Maximum length for summary
SUMMARY_NUM_BEAMS = char_config['RAG_params']['summary_beam_size']  # Beam search size for summarization
SUMMARY_MAX_TOKENS = char_config['RAG_params']['summary_max_tokens'] # Maximum number of tokens for BART input
MEMORY_CLEANUP_THRESHOLD_DAYS = char_config['RAG_params']['memory_cleanup_threshold']  # Days before memory is eligible for cleanup
MEMORY_IMPORTANCE_THRESHOLD = char_config['RAG_params']['memory_importance_threshold']  # Threshold below which memories are discarded

_summarizer_tokenizer = None
_summarizer_model = None

# Load SentenceTransformer model
model = SentenceTransformer(MODEL_NAME)

def get_summarizer():
    """Load the summarizer lazily (only when needed)."""
    global _summarizer_tokenizer, _summarizer_model
    if _summarizer_tokenizer is None or _summarizer_model is None:
        print("[INFO] Loading BART summarizer model...")
        _summarizer_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_NAME)
        _summarizer_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)
    return _summarizer_tokenizer, _summarizer_model

# Load or initialize memory store
def load_memory_store():
    if os.path.exists(MEMORY_STORE_PATH):
        with open(MEMORY_STORE_PATH, 'r') as f:
            memory_store = json.load(f)
        print(f"Loaded {len(memory_store)} memories from file.")
    else:
        memory_store = [
            {"text": "I Love Senpai very much", "importance_score": 0.9, "timestamp": time.time(), "access_count": 0, "detailed": True},
            {"text": "I am Riko", "importance_score": 0.7, "timestamp": time.time(), "access_count": 0, "detailed": True},
            {"text": "I should use jokes and sarcasm ", "importance_score": 0.8, "timestamp": time.time(), "access_count": 0, "detailed": True},
            {"text": "I like to tease Senpai", "importance_score": 0.9, "timestamp": time.time(), "access_count": 0, "detailed": True},
            {"text": "I love spending time with Senpai", "importance_score": 0.9, "timestamp": time.time(), "access_count": 0, "detailed": True},
            ]
        print("No memory store found, using default memories.")
    return memory_store

# Save memory store to disk
def save_memory_store(memory_store):
    with open(MEMORY_STORE_PATH, 'w') as f:
        json.dump(memory_store, f)
    print(f"Saved {len(memory_store)} memories to file.")

# Create FAISS-CPU index
def create_faiss_cpu_index(dim: int):
    index = faiss.IndexFlatL2(dim)
    return index

# Save FAISS index to disk
def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index saved to file.")

# Load FAISS index from disk
def load_faiss_index(dim: int):
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        print("FAISS index loaded from file.")
    else:
        index = create_faiss_cpu_index(dim)
        print("No FAISS index found, creating new index.")
    return index

# Function to get the embedding of a text
def get_embedding(text: str):
    embedding = model.encode(text)
    return embedding

# Function to add embeddings to FAISS index
def add_embeddings_to_faiss(index, memory_store):
    embeddings = np.array([get_embedding(memory['text']) for memory in memory_store]).astype('float32')
    index.add(embeddings)

# Function to query FAISS-CPU for relevant memories
def query_faiss_cpu(index, query_embedding, k=5):
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')  # Reshape for single query
    distances, indices = index.search(query_embedding, k)  # Get top-k indices and distances
    return indices, distances

# Function to summarize memory using BART
def summarize_text(text):
    """Summarize text using BART."""
    tokenizer, model = get_summarizer()
    inputs = tokenizer([text], max_length=SUMMARY_MAX_TOKENS, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=SUMMARY_NUM_BEAMS,
        min_length=SUMMARY_MIN_LENGTH,
        max_length=SUMMARY_MAX_LENGTH,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to apply timestamp decay and rank memories
def decay_memory(memory):
    current_time = time.time()
    age_in_days = (current_time - memory['timestamp']) / (60 * 60 * 24)

    # Decay based on importance: high importance memories decay slower
    decay_factor = FAISS_DECAY_FACTOR_LOW
    if memory['importance_score'] > 0.8:
        decay_factor = FAISS_DECAY_FACTOR_HIGH  # Slow decay for high-importance memories

    decay = np.exp(-decay_factor * age_in_days)
    memory['importance_score'] *= decay

    # Transition from detailed to summarized when importance is low
    if memory['importance_score'] < 0.3 and memory['detailed']:  # Low importance, summarized
        memory['text'] = summarize_text(memory['text'])
        memory['detailed'] = False

    # After a certain period, discard the memory if importance is very low
    if memory['importance_score'] < MEMORY_IMPORTANCE_THRESHOLD and not memory['detailed']:  # Threshold for discarding
        return None  # This memory should be discarded

    return memory

# Function to clean up low-importance memories (optional cleanup process)
def cleanup_memory_store(memory_store):
    current_time = time.time()
    threshold_age_days = MEMORY_CLEANUP_THRESHOLD_DAYS  # Clean memories older than 30 days
    threshold_importance = MEMORY_IMPORTANCE_THRESHOLD  # Discard memories with importance lower than 0.1

    # Filter out low-importance or very old memories
    memory_store = [
        memory for memory in memory_store
        if (current_time - memory['timestamp']) < (threshold_age_days * 60 * 60 * 24) or memory['importance_score'] >= threshold_importance
    ]

    print(f"Cleaned up memory store, {len(memory_store)} memories remaining.")
    return memory_store

# Function to update memory importance based on access count
def update_memory_importance(memory):
    access_boost = 0.1 * (memory['access_count'] ** 0.5)  # Exponential growth for access
    memory['importance_score'] += access_boost
    memory['importance_score'] = min(memory['importance_score'], 1.0)  # Cap at 1.0
    return memory

# Function to retrieve relevant memories based on a prompt
def get_relevant_memories(prompt, memory_store, index, k=5):
    prompt_embedding = get_embedding(prompt)
    indices, distances = query_faiss_cpu(index, prompt_embedding, k)
    
    # Rank memories by importance, decay, and similarity
    ranked_memories = []
    for idx, distance in zip(indices[0], distances[0]):
        memory = copy.copy(memory_store[idx])
        age_in_days = (time.time() - memory['timestamp']) / (60 * 60 * 24)
        decay_factor = FAISS_DECAY_FACTOR_LOW
        if memory['importance_score'] > 0.8:
            decay_factor = FAISS_DECAY_FACTOR_HIGH
        decay = np.exp(-decay_factor * age_in_days)
        
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

# Main script execution
if __name__ == "__main__":
    # Initialize memory store and FAISS index
    memory_store = load_memory_store()
    faiss_index = load_faiss_index(EMBEDDING_DIM)

    # If index is empty, add all memory embeddings to the FAISS index
    if faiss_index.ntotal == 0:
        add_embeddings_to_faiss(faiss_index, memory_store)
        save_faiss_index(faiss_index)

    # Example prompt and retrieval
    prompt = "Tell me more about yourself?"
    ranked_memories, indices, distances = get_relevant_memories(prompt, memory_store, faiss_index)

    # Print the ranked memories
    print("Ranked Memories:")
    for memory in ranked_memories:
        print(f"Memory Text: {memory['text']}")
        print(f"Importance Score: {memory['importance_score']:.4f}, Similarity Score: {memory['similarity_score']:.4f}")
        print(f"Decay Score: {memory['ranked_score']:.4f}, Access Count: {memory['access_count']}")
        print("-" * 50)

    # Cleanup and save updated memory store
    memory_store = cleanup_memory_store(memory_store)
    save_memory_store(memory_store)
