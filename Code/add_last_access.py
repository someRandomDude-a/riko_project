import pathlib
import json
import tempfile
import os
from process.llm_scripts.Memory_system.long_term_memory import test_script, _memory_store

_MEMORY_STORE_PATH = './persistant_memories/memory_store.json'
_MEMORY_STORE_PATH = pathlib.Path(_MEMORY_STORE_PATH).resolve()
_MEMORY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            
for memory in _memory_store:
    if not memory.get("last_access"):
        memory["last_access"] = memory["created_on"]

test_script()