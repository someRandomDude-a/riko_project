"""/api/memories: list, correct and delete long-term memories (persistence/memory.py)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .backend import Services

router = APIRouter()


@router.get("/api/memories")
def memories(backend: Services):
    memory = backend.chat.memory_store
    return {"records": memory.list_records(), "pipeline": memory.status()}

class MemoryUpdate(BaseModel):
    text: str | None = None
    memory_type: str | None = None
    importance: float | None = None
    tags: list[str] | None = None
    active: bool | None = None

@router.patch("/api/memories/{record_id}")
def update_memory(record_id: str, request: MemoryUpdate, backend: Services):
    try:
        return backend.chat.memory_store.update(record_id, **request.model_dump(exclude_unset=True))
    except KeyError: raise HTTPException(404, "Memory not found")
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))

@router.delete("/api/memories/{record_id}")
def delete_memory(record_id: str, backend: Services):
    try: backend.chat.memory_store.delete(record_id)
    except KeyError: raise HTTPException(404, "Memory not found")
    return {"deleted": record_id}
