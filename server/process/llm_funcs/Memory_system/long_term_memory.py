import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import json
import pathlib
import torch
import yaml
from datetime import datetime
import tempfile
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uuid

_DEBUG = True

def _debug_out(text: str):
    if _DEBUG:
        print (text)

# === Load Character Configuration ===
with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)

# === Summarizer functions ===
_SUMMARY_MODEL_ID = char_config['RAG_params']['model_id']
_SUMMARY_NUM_BEAMS = char_config['RAG_params']['summary_beam_size']
_SUMMARY_MAX_TOKENS = char_config['RAG_params']['summary_max_tokens']

_summary_tokenizer = AutoTokenizer.from_pretrained(_SUMMARY_MODEL_ID)
_summary_model = AutoModelForSeq2SeqLM.from_pretrained(
    _SUMMARY_MODEL_ID,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
if torch.cuda.is_available():
    _summary_model = _summary_model.to("cuda")

_summary_model.eval()

def _summarize_text(text: str) -> str:
    """
    Summarize memories into concise first-person features.
    - text: str, detailed memory text (from self-reflection)
    Returns: str, summarized memory ready for FAISS
    """
    # Build prompt for summarization
    prompt = (f"Summarize the following detailed memories into concise, first-person factual features suitable for retrieval:\n\n{text}")

    # Tokenize input
    inputs = _summary_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=_SUMMARY_MAX_TOKENS
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate summary with beam search
    with torch.inference_mode():
        summary_ids = _summary_model.generate(
            **inputs,
            max_length=_SUMMARY_MAX_TOKENS,
            num_beams=_SUMMARY_NUM_BEAMS,
            early_stopping=True
        )

    # Decode output
    summary_text = _summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Optional: split into separate lines for FAISS, then recombine with newline
    summary_lines = [line.strip("{} ") for line in summary_text.split("\n") if line.strip()]
    summary_text = "\n".join(summary_lines)
    return summary_text

# ===================FAISS functions===============

_EMBEDDING_DIM = char_config['RAG_params']['text_embedding_dim']
def _create_faiss_cpu_index():
    """Create FAISS-CPU index"""

    M = 64  # Number of bi-directional links per node (tradeoff: accuracy vs. memory)
    index = faiss.IndexHNSWFlat(_EMBEDDING_DIM, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200  # Controls build accuracy (higher = better, slower)
    index.hnsw.efSearch = 100         # Controls query accuracy (higher = better, slower)
    index = faiss.IndexIDMap2(index)
    _debug_out(f"Created HNSWFlat FAISS index (dim={_EMBEDDING_DIM}, M={M})")
    return index

_FAISS_INDEX_PATH = './persistant_memories/faiss_index.index' # File path for FAISS index
_FAISS_INDEX_PATH = pathlib.Path(_FAISS_INDEX_PATH).resolve()
def _save_faiss_index():
    """Save FAISS index to disk"""
    faiss.write_index(_faiss_index, _FAISS_INDEX_PATH.as_posix())
    _debug_out("FAISS index saved to file.")

def _load_faiss_index():
    """Load FAISS index from disk"""
    if _FAISS_INDEX_PATH.is_file():
        index = faiss.read_index(_FAISS_INDEX_PATH.as_posix())
    else:
        index = _create_faiss_cpu_index()
    return index

_MODEL_NAME = char_config['RAG_params']['embedding_model_id']
embedding_model = SentenceTransformer(_MODEL_NAME)
def _get_embedding(text: str | list[str]):
    """get the embedding of a text"""
    embedding = embedding_model.encode(text, convert_to_numpy=True).astype("float32")
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    faiss.normalize_L2(embedding)
    return embedding 

def _add_entire_memory_store():
    """add embeddings to FAISS index for every memory in the memory_store.
        Returns immediately if memory store is `None`"""
    global _memory_store
    
    if not _memory_store:
        return

    _memory_store = [memory for memory in _memory_store if memory]
    texts = [memory['text'] for memory in _memory_store]
    
    # Ensure each memory has an ID and convert to FAISS int64
    ids = []
    memory_changed = False
    for memory in _memory_store:
        if 'id' not in memory:
            memory_changed = True
            memory['id'] = int(uuid.uuid4().int % (2**63))  # inline UUID
        ids.append(memory['id'])  # FAISS requires int64

    ids = np.array(ids, dtype=np.int64)
    embeddings = _get_embedding(texts)
    _faiss_index.add_with_ids(embeddings, ids)
    
    if memory_changed:
        _save_memory_store()

# ============= RAG memory functions ===================

_MEMORY_CLEANUP_THRESHOLD_DAYS = char_config['RAG_params']['memory_cleanup_threshold'] # Days before memory is eligible for cleanup
_MEMORY_IMPORTANCE_THRESHOLD = char_config['RAG_params']['memory_importance_threshold'] # Threshold below which memories are discarded
def cleanup_memory_store():
    """
    Function to clean up low-importance memories (optional cleanup process)
    MEMORY_CLEANUP_THRESHOLD_DAYS  # Clean memories older than 30 days
    MEMORY_IMPORTANCE_THRESHOLD  # Discard memories with importance lower than 0.1
    """
    # Filter out low-importance or very old memories
    global _memory_store
    global _faiss_index

    to_remove = [
        memory for memory in _memory_store
        if not (_get_age_in_days(memory) < _MEMORY_CLEANUP_THRESHOLD_DAYS
                or memory['importance_score'] >= _MEMORY_IMPORTANCE_THRESHOLD)
    ]

    for memory in to_remove:
        _faiss_index.remove_ids(np.array([memory['id']], dtype=np.int64))

    _memory_store = [m for m in _memory_store if m not in to_remove]

    _debug_out(f"Cleaned up memory store, {len(_memory_store)} memories remaining.")
    _save_faiss_index()
    _save_memory_store()




def get_RAG_context(user_input):
    """
      Query long-term memory for related past experiences.
    """
    ranked_memories = _get_relevant_memories(user_input)
    if not ranked_memories:
        return ""
    
    memory_snippets = "\n".join([f"- {m['text']} (timestamp:{m['timestamp']})" for m in ranked_memories])
    return f"""
### Relevant Memories
These are past interactions that may be relevant.
Use them only if useful.
{memory_snippets}    
"""

_TOP_K = char_config['RAG_params']['default_top_k']    
def _get_relevant_memories(prompt):
    """retrieve relevant memories based on a prompt, returns K memories"""

    indices, similarity = _query_faiss_cpu(prompt, _TOP_K)
    id_to_index = {m['id']: i for i, m in enumerate(_memory_store)}
    # Rank memories by importance, decay, and similarity
    ranked_memories = []
    for idx, similarity in zip(indices[0], similarity[0]):
        if idx < 0:
            continue
        # build ranked memory array
        memory = _memory_store[id_to_index[idx]]
        decay = np.exp((_MEMORY_DECAY_FACTOR_HIGH if memory['importance_score'] > 0.8 else _MEMORY_DECAY_FACTOR_LOW)  * _get_age_in_days(memory))
        similarity_score = (similarity + 1) / 2

        ranked_score = memory['importance_score'] * decay * 0.6 + similarity_score * 0.4
        ranked_memories.append({
            "text": memory["text"],
            "timestamp": memory["timestamp"],
            "ranked_score": ranked_score,
        })

        # Update access count and importance
        _memory_store[id_to_index[idx]]['access_count'] += 1
        _update_memory_importance(_memory_store[id_to_index[idx]])

    ranked_memories.sort(key=lambda x: x['ranked_score'], reverse=True)

    _decay_memory_store()
    return ranked_memories

def _query_faiss_cpu(text, k=5):
    """query FAISS-CPU for relevant memories"""
    query_embedding = _get_embedding(text)
    similarity, indices = _faiss_index.search(query_embedding, k)  # Get top-k indices and distances
    return indices, similarity

_AGE_DIV_FACTOR = 60 * 60 * 24
def _get_age_in_days(memory):
    return  (time.time() - datetime.fromisoformat(memory['timestamp']).timestamp()) / _AGE_DIV_FACTOR

def _update_memory_importance(memory):
    """Function to update memory importance based on access count"""
    access_boost = 0.1 * (memory['access_count'] ** 0.5)  # Exponential growth for access
    memory['importance_score'] += access_boost
    memory['importance_score'] = min(memory['importance_score'], 1.0)  # Cap at 1.0
    return memory

_MEMORY_DECAY_FACTOR_HIGH = -char_config['RAG_params']['high_importance_decay_factor'] # Decay factor for high-importance memories
_MEMORY_DECAY_FACTOR_LOW = -char_config['RAG_params']['low_importance_decay_factor'] # Decay factor for low-importance memories 
def _decay_memory_store():
    """apply timestamp decay and rank memories"""
    # Decay based on importance: high importance memories decay slower
    for memory in _memory_store:
        # skip empty (deleted) memories
        if not memory:
            continue

        decay = np.exp(
            (_MEMORY_DECAY_FACTOR_HIGH if memory['importance_score'] > 0.8 else _MEMORY_DECAY_FACTOR_LOW) * _get_age_in_days(memory)
        )

        memory['importance_score'] = max(memory['importance_score'] * decay, 0)
        # Transition from detailed to summarized when importance is low
        if memory['importance_score'] < 0.3 and memory['detailed'] and len(memory['text']) > 300:  # Low importance, summarized
            _debug_out("summarizing memory: " + memory['text'])
            memory['text'] = _summarize_text(memory['text'])
            memory['detailed'] = False




_DEFAULT_MEMORY_IMPORTANCE = char_config['RAG_params']['default_importance_score']
def add_message_to_memory(message_text : str, message_time : str, context):
    """Adds a message to long-term memory with default importance."""
    # === Skip tiny messages ===
    if len(message_text) < 25:
        return

    # === Skip duplicates ===
    embedding = _get_embedding(message_text)
    D, I = _faiss_index.search(embedding, 1)
    if D[0][0] > 0.8:  # Threshold depends on embedding space
        _debug_out("Duplicate detected via embedding similarity.")
        return

    memory_text = _self_reflection(message_text, context)
    # format new memory entry
    new_memory = {
        "id": int(uuid.uuid4().int % (2**63)),  # inline UUID
        "text": memory_text,
        "importance_score": _DEFAULT_MEMORY_IMPORTANCE ,
        "timestamp": message_time,
        "access_count": 0,
        "detailed": True,
    }
    _memory_store.append(new_memory)
    
    embedding = _get_embedding(memory_text)
    _faiss_index.add_with_ids(embedding, np.array([new_memory['id']], dtype=np.int64))

    _save_memory_store()
    _save_faiss_index()


_MAX_REFLECTION_TOKENS = char_config["Self_reflection_params"]["token_limit"]
def _self_reflection(message: str, context) -> str:
    """
    Generate a detailed first-person reflection for a message given the current context.
    - message: str, the latest user message
    - context: list, previous messages in rolling context
    Returns: str, detailed self-reflection
    """
    # Flatten context safely
    context_text = ""
    for msg in context:
        for item in msg.get("content", []):
            text = item.get("text", "").strip()
            text = text.split(" timestamp:", 1)[0].strip()

            if text:  # skip empty
                if text.startswith("("):
                    head, sep, tail = text.partition(")")
                    if sep:  # found closing parenthesis
                        speaker = head[1:].strip()
                        text = f"{speaker}: {tail.strip()}"
            
                context_text += text + "\n"

    # Build robust prompt
    prompt = (
        f"Current context (older messages):\n{context_text}\n\n"
        f"New message (latest):\n{message}\n\n"
        "Instruction: Write a detailed, first-person reflection summarizing this new message. "
        "Include explicit facts, nuances, and relationships for future memory retrieval. "
        "Focus on the user's text first, but include relevant context if necessary."
    )

    # Tokenize with truncation: prioritize keeping the latest message
    inputs = _summary_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_REFLECTION_TOKENS
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate reflection
    with torch.inference_mode():    
        output_ids = _summary_model.generate(
            **inputs,
            max_length=_MAX_REFLECTION_TOKENS,
            num_beams=4,
            early_stopping=True
        )

    # Decode output
    self_reflection = _summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return self_reflection

# ================ Memory Store Functions ===================

_MEMORY_STORE_PATH = './persistant_memories/memory_store.json'
_MEMORY_STORE_PATH = pathlib.Path(_MEMORY_STORE_PATH).resolve()
_MEMORY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_memory_store():
    """Load memorystore from file if it exists, else load defaults"""
    global _memory_store 
    global _faiss_index

    _memory_store = []
    if _MEMORY_STORE_PATH.is_file():
        with open(_MEMORY_STORE_PATH, 'r') as f:
            try:
                _memory_store = json.load(f)
                if not isinstance(_memory_store, list):
                    raise ValueError("Memory_store is not a list")
                _debug_out(f"Loaded {len(_memory_store)} memories from file.")

                if len(_memory_store) != _faiss_index.ntotal:
                    _debug_out("[ERROR]memory Store and vector store index mismatch!, Rebuilding FAISS index.\n[INFO]This can take a long time, do not worry!")
                    _faiss_index = _create_faiss_cpu_index()
                    _add_entire_memory_store()
                    _save_faiss_index()

            except (json.JSONDecodeError, ValueError) as e:
                _debug_out("[WARN] Memory store file is empty or corrupted", e)
                _memory_store = []

    
    # return if the memory store is not empty
    if _memory_store:
        return _memory_store
    
    # Fetch default memories
    default_memories = char_config["presets"]["default"]["memories"]

    # Build memory store dynamically
    currentTime = datetime.now().isoformat(timespec='minutes')

    for mem in default_memories:
        _memory_store.append({
            "text": mem["text"],
            "importance_score": mem["importance_score"],
            "timestamp": currentTime,
            "access_count": mem.get("access_count", 0),
            "detailed": mem.get("detailed", True),
        })

    _debug_out("No memories found, Loaded memory store from YAML.")
    _faiss_index = _create_faiss_cpu_index()
    _add_entire_memory_store()
    _save_faiss_index()
    _save_memory_store()
    return _memory_store

def _save_memory_store():
    """Save memory store to disk"""
    with tempfile.NamedTemporaryFile(
        'w',
        dir=_MEMORY_STORE_PATH.parent,
        delete=False,
        encoding='utf-8'
    ) as tmp:
        json.dump(_memory_store, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = pathlib.Path(tmp.name)

    temp_path.replace(_MEMORY_STORE_PATH)
    _debug_out(f"Saved {len(_memory_store)} memories to file.")



# === Initialize RAG Memory System ===
_faiss_index = _load_faiss_index()
_memory_store = _load_memory_store()

if _faiss_index.ntotal == 0:
    _add_entire_memory_store()
    _save_faiss_index()


# Test script
if __name__ == "__main__":
    prompt = "Tell me more about yourself?"
    ranked_memories = _get_relevant_memories(prompt)

    print("Ranked Memories:")
    for memory in ranked_memories:
        print(f"Memory Text: {memory['text']}")
    
    cleanup_memory_store()
