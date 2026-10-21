"""/api/tasks: the user's view of the revision-checked task store (persistence/tasks.py); a stale revision is 409."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..persistence.tasks import TaskConflict
from .backend import Services

router = APIRouter()


@router.get('/api/tasks')
def tasks(backend: Services, include_closed: bool = True, query: str = ''):
    return {'tasks': backend.chat.deps.task_store.list(include_closed=include_closed, query=query, limit=100)}

@router.get('/api/tasks/{task_id}')
def get_task(task_id: str, backend: Services):
    try: return backend.chat.deps.task_store.get(task_id)
    except KeyError: raise HTTPException(404, 'Task not found')

class TaskCreateRequest(BaseModel):
    title: str
    description: str = ''
    next_step: str = ''

@router.post('/api/tasks')
def create_task(request: TaskCreateRequest, backend: Services):
    try: return backend.chat.deps.task_store.create(**request.model_dump(), actor='user_api', reason='Explicit user creation')
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))

class TaskUpdateRequest(BaseModel):
    expected_revision: int
    changes: dict
    reason: str = 'Explicit user correction'

@router.patch('/api/tasks/{task_id}')
def update_task(task_id: str, request: TaskUpdateRequest, backend: Services):
    try: return backend.chat.deps.task_store.update(task_id, **request.model_dump(), actor='user_api')
    except TaskConflict as exc: raise HTTPException(409, str(exc))
    except KeyError: raise HTTPException(404, 'Task not found')
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))
