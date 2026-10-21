"""/api/initiative: the background initiative's settings and state (runtime/initiative.py), and user.* trigger events."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .backend import Services

router = APIRouter()


@router.get('/api/initiative')
def initiative_status(backend: Services): return backend.initiative()

@router.put('/api/initiative')
def initiative_settings(request: dict, backend: Services):
    try: return backend.session.initiative.update(request)
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))

class CustomEventRequest(BaseModel):
    event: str = 'user.custom'
    context: str = ''

@router.post('/api/initiative/event')
def custom_event(request: CustomEventRequest, backend: Services):
    if request.event not in backend.session.initiative.triggers or not request.event.startswith('user.'):
        raise HTTPException(400, 'Only registered user.* events can be published here')
    if len(request.context) > 2000: raise HTTPException(400, 'Context exceeds 2000 characters')
    event = backend.bus.publish(request.event, context=request.context)
    return {'event_id': event.id}
