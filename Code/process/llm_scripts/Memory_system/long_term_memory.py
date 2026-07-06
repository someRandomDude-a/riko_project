from __future__ import annotations

import re
import time
import json
import pathlib
import uuid
import tempfile
import os
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from process.common.config import char_config
from process.llm_scripts.llama_server import (
    get_llm_token_length,
    call_llm_api,
    summarize,
)

_DEBUG = True


def _debug_out(text: str):
    if _DEBUG:
        print(text)


# Summarizer

def _summarize_text(text: str) -> str:
    """Summarize a long detailed memory into a compact snippet. No-op fallback."""
    try:
        out = summarize(text).strip()
    except Exception as e:
        _debug_out(f"[WARN] summarization failed: {e!r}")
        return text
    out_lines = [
        line.strip("{} ")
        for line in out.split("\n")
        if line.strip()
    ]
    return "\n".join(out_lines) if out_lines else out


# FAISS

_EMBEDDING_DIM = char_config['RAG_params']['text_embedding_dim']


def _create_faiss_cpu_index():
    M = 64
    index = faiss.IndexHNSWFlat(_EMBEDDING_DIM, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 100
    index = faiss.IndexIDMap2(index)
    _debug_out(f"Created HNSWFlat FAISS index (dim={_EMBEDDING_DIM}, M={M})")
    return index


_FAISS_INDEX_PATH = pathlib.Path('./persistant_memories/faiss_index.index').absolute()
_FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def _save_faiss_index():
    faiss.write_index(_faiss_index, _FAISS_INDEX_PATH.as_posix())
    _debug_out("FAISS index saved to file.")


def _load_faiss_index():
    if _FAISS_INDEX_PATH.is_file():
        return faiss.read_index(_FAISS_INDEX_PATH.as_posix())
    return _create_faiss_cpu_index()


_MODEL_NAME = char_config['RAG_params']['embedding_model_id']
embedding_model = SentenceTransformer(_MODEL_NAME)


def _get_embedding(text):
    embedding = embedding_model.encode(text, convert_to_numpy=True).astype("float32")
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    faiss.normalize_L2(embedding)
    return embedding


def _add_entire_memory_store():
    global _memory_store
    if not _memory_store:
        return
    _memory_store = [m for m in _memory_store if m]
    texts = [m['text'] for m in _memory_store]

    ids, changed = [], False
    for memory in _memory_store:
        if 'id' not in memory:
            changed = True
            memory['id'] = int(uuid.uuid4().int % (2 ** 63))
        ids.append(memory['id'])
    ids = np.array(ids, dtype=np.int64)
    embeddings = _get_embedding(texts)
    _faiss_index.add_with_ids(embeddings, ids)  # type: ignore
    if changed:
        _save_memory_store()


# Memory lifecycle

_MEMORY_CLEANUP_THRESHOLD_DAYS = char_config['RAG_params']['memory_cleanup_threshold']
_MEMORY_IMPORTANCE_THRESHOLD = char_config['RAG_params']['memory_importance_threshold']


def cleanup_memory_store():
    global _memory_store, _faiss_index
    to_remove = [
        m for m in _memory_store
        if not (_get_age_in_days(m) < _MEMORY_CLEANUP_THRESHOLD_DAYS
                or m['importance_score'] >= _MEMORY_IMPORTANCE_THRESHOLD)
    ]
    for m in to_remove:
        _faiss_index.remove_ids(np.array([m['id']], dtype=np.int64))  # type: ignore
    _memory_store = [m for m in _memory_store if m not in to_remove]
    _debug_out(f"Cleaned up memory store, {len(_memory_store)} memories remaining.")
    _save_faiss_index()
    _save_memory_store()


_HEADER_TEXT = "### Relevant Memories. These are past interactions that may be relevant.\n"
_MAX_MEMORY_TOKENS = char_config['RAG_params']['max_token_budget']
_HEADER_TOKENS = get_llm_token_length(_HEADER_TEXT)


def get_RAG_context(user_input):
    memories = _get_relevant_memories(user_input)
    if not memories:
        return ""
    token_count = sum((m["tokens"] for m in memories), _HEADER_TOKENS)
    while token_count > _MAX_MEMORY_TOKENS and memories:
        removed = memories.pop()
        token_count -= removed["tokens"]
    snippets = "\n".join(f"- [{m['created_on']}] {m['text']}" for m in memories)
    return f"{_HEADER_TEXT}{snippets}"


def _get_relevant_memories(prompt):
    indices, similarity = _query_faiss_cpu(prompt)
    id_to_index = {m['id']: i for i, m in enumerate(_memory_store)}
    ranked = []
    for idx, sim in zip(indices[0], similarity[0]):
        if idx < 0:
            continue
        memory = _memory_store[id_to_index[idx]]
        decay = np.exp(
            (_MEMORY_DECAY_FACTOR_HIGH if memory['importance_score'] > 0.8 else _MEMORY_DECAY_FACTOR_LOW)
            * _get_age_in_days(memory)
        )
        similarity_score = (sim + 1) / 2
        ranked_score = memory['importance_score'] * decay * 0.6 + similarity_score * 0.4
        ranked.append({
            "text": memory["text"],
            "created_on": memory["created_on"],
            "ranked_score": ranked_score,
            "tokens": memory["tokens"],
        })
        memory['access_count'] += 1
        memory['last_access'] = datetime.now().isoformat()
        _update_memory_importance(memory)
    ranked.sort(key=lambda x: x['ranked_score'], reverse=True)
    _decay_memory_store()
    return ranked


_TOP_K = char_config['RAG_params']['max_memories']


def _query_faiss_cpu(text):
    query_embedding = _get_embedding(text)
    similarity, indices = _faiss_index.search(query_embedding, _TOP_K)  # type: ignore
    return indices, similarity


_AGE_DIV_FACTOR = 60 * 60 * 24


def _get_age_in_days(memory):
    return (time.time() - datetime.fromisoformat(memory['last_access']).timestamp()) / _AGE_DIV_FACTOR


def _update_memory_importance(memory):
    boost = 0.1 * (memory['access_count'] ** 0.5)
    memory['importance_score'] += boost
    memory['importance_score'] = min(memory['importance_score'], 1.0)
    return memory


_MEMORY_DECAY_FACTOR_HIGH = -char_config['RAG_params']['high_importance_decay_factor']
_MEMORY_DECAY_FACTOR_LOW = -char_config['RAG_params']['low_importance_decay_factor']


def _decay_memory_store():
    for memory in _memory_store:
        if not memory:
            continue
        decay = np.exp(
            (_MEMORY_DECAY_FACTOR_HIGH if memory['importance_score'] > 0.8 else _MEMORY_DECAY_FACTOR_LOW)
            * _get_age_in_days(memory)
        )
        memory['importance_score'] = max(memory['importance_score'] * decay, 0)
        if memory['importance_score'] < 0.3 and memory['detailed'] and len(memory['text']) > 300:
            _debug_out("summarizing memory: " + memory['text'])
            memory['text'] = _summarize_text(memory['text'])
            memory['detailed'] = False
            memory['tokens'] = get_llm_token_length(f"- [{memory['created_on']}] {memory['text']}")


_DEFAULT_MEMORY_IMPORTANCE = char_config['RAG_params']['default_importance_score']


def add_message_to_memory(message_text, message_time, message_tokens, context):
    if len(message_text) < 25:
        return
    embedding = _get_embedding(message_text)
    D, I = _faiss_index.search(embedding, 1)  # type: ignore
    if D[0][0] > 0.8:
        _debug_out("Duplicate detected via embedding similarity.")
        return

    memory_text = _self_reflection(message_text, context)

    new_memory = {
        "id": int(uuid.uuid4().int % (2 ** 63)),
        "text": memory_text,
        "importance_score": _DEFAULT_MEMORY_IMPORTANCE,
        "created_on": message_time,
        "last_access": message_time,
        "access_count": 0,
        "tokens": message_tokens,
        "detailed": True,
    }
    _memory_store.append(new_memory)
    embedding = _get_embedding(memory_text)
    _faiss_index.add_with_ids(embedding, np.array([new_memory['id']], dtype=np.int64))  # type: ignore
    _save_memory_store()
    _save_faiss_index()


# Self-reflection

_REFLECTION_MODEL_ID = char_config["Self_reflection_params"]["model_id"]
_MAX_REFLECTION_INPUT = char_config["Self_reflection_params"]["context_limit"]
_MAX_REFLECTION_OUTPUT = char_config["Self_reflection_params"]["token_limit"]


def _self_reflection(message: str, context) -> str:
    """
    Generate a first-person reflection for memory storage. Catches any
    LLM failure and falls back to the raw message — never let a reflection
    call break the higher-level save_memory path.
    """
    raw_texts = []
    for msg in context:
        try:
            content_text = msg["content"][0]["text"]
        except Exception:
            continue
        cleaned = re.sub(r'^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}\]\s*', '', content_text)
        raw_texts.append(cleaned)
    context_text = "\n".join(raw_texts)

    prompt = (
        "Instruction: Write a detailed, first-person reflection summarizing this new message. "
        "Include explicit facts, nuances, and relationships for future memory retrieval.\n\n"
        f"Current context (older messages):\n{context_text}\n\n"
        f"New message (latest):\n{message}\n\n"
        "Reflection:"
    )

    messages = [
        {"role": "system", "content": [
            {"type": "input_text",
             "text": "You are a helpful assistant that generates concise, first-person reflections for memory storage."}
        ]},
        {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
    ]

    try:
        response = call_llm_api(messages, tools=[])
        reflection = response.output_text.strip()
    except Exception as e:
        _debug_out(f"[WARN] _self_reflection failed: {e!r}; falling back to raw message")
        return message

    # Strip thinking blocks; Qwen3.5 sometimes returns `` only.
    reflection = re.sub(r"", "", reflection, flags=re.DOTALL).strip()
    return reflection or message


# Persistence

_MEMORY_STORE_PATH = pathlib.Path('./persistant_memories/memory_store.json').absolute()
_MEMORY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_memory_store():
    global _memory_store, _faiss_index
    _memory_store = []
    if _MEMORY_STORE_PATH.is_file():
        try:
            with open(_MEMORY_STORE_PATH, 'r') as f:
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
            _debug_out(f"[WARN] Memory store file is empty or corrupted: {e}")
            _memory_store = []

    if _memory_store:
        return _memory_store

    default_memories = char_config["presets"]["default"]["memories"]
    current_time = datetime.now().isoformat(timespec='minutes')
    for mem in default_memories:
        created_on = mem.get("date", current_time)
        _memory_store.append({
            "text": mem["text"],
            "importance_score": mem.get("importance_score", _DEFAULT_MEMORY_IMPORTANCE),
            "created_on": created_on,
            "last_access": created_on,
            "access_count": mem.get("access_count", 0),
            "detailed": mem.get("detailed", True),
            "tokens": get_llm_token_length(f"- [{created_on}] {mem['text']}"),
        })
    _debug_out("No memories found, loaded memory store from YAML.")
    _faiss_index = _create_faiss_cpu_index()
    _add_entire_memory_store()
    _save_faiss_index()
    _save_memory_store()
    return _memory_store


def _save_memory_store():
    with tempfile.NamedTemporaryFile(
        'w', dir=_MEMORY_STORE_PATH.parent, delete=False, encoding='utf-8'
    ) as tmp:
        json.dump(_memory_store, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = pathlib.Path(tmp.name)
    temp_path.replace(_MEMORY_STORE_PATH)
    _debug_out(f"Saved {len(_memory_store)} memories to file.")


def migrate_memories():
    msg_text = [f'- [{m["created_on"]}] {m["text"]}' for m in _memory_store]
    tokens = get_llm_token_length(msg_text)
    for memory, token_count in zip(_memory_store, tokens):
        memory["tokens"] = token_count
    _save_memory_store()
    _debug_out("Memory tokens updated and saved.")


# Init

_faiss_index = _load_faiss_index()
_memory_store = _load_memory_store()

if _faiss_index.ntotal == 0:
    _add_entire_memory_store()
    _save_faiss_index()


def test_script():
    migrate_memories()
    prompt = "Tell me more about yourself?"
    memories = get_RAG_context(prompt)
    print(memories)
