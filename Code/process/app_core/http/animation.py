"""/api/animation: the avatar's animation library, previews, pointer interaction, renderer capabilities and walking
(animation/runtime.py); 503 while the animation service is off."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StrictInt

from .backend import Services

router = APIRouter()


@router.get('/api/animation')
def animation_status(backend: Services): return backend.animation()

class AnimationImport(BaseModel):
    path: str = Field(max_length=2048)
    metadata: dict = Field(default_factory=dict)

@router.post('/api/animation/import')
def animation_import(request: AnimationImport, backend: Services):
    service = backend.animation_service()
    try: entry = service.library.import_file(request.path, request.metadata)
    except (ValueError, OSError, TypeError) as exc: raise HTTPException(400, str(exc))
    service.invalidate_library()
    return entry

@router.patch('/api/animation/assets/{identifier}')
def animation_metadata(identifier: str, request: dict, backend: Services):
    service = backend.animation_service()
    try: entry = service.library.update(identifier, request)
    except (ValueError, TypeError, OSError) as exc: raise HTTPException(400, str(exc))
    service.invalidate_library()
    return entry

@router.get('/api/animation/assets/{identifier}/file')
def animation_asset(identifier: str, backend: Services):
    try: path = backend.animation_service().library.path(identifier)
    except ValueError as exc: raise HTTPException(404, str(exc))
    return FileResponse(path)

@router.post('/api/animation/assets/{identifier}/preview')
def animation_preview(identifier: str, backend: Services):
    try: action = backend.animation_service().preview(identifier)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'action_id': action.id}

class AnimationInteraction(BaseModel):
    kind: str
    pointer: dict | None = None
    target: dict | None = None

@router.post('/api/animation/interaction')
def animation_interaction(request: AnimationInteraction, backend: Services):
    try: backend.animation_service().interact(request.kind, request.pointer, request.target)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'ok': True}

class AnimationCapabilities(BaseModel):
    bones: list[str]
    expressions: list[str]

@router.post('/api/animation/capabilities')
def animation_capabilities(request: AnimationCapabilities, backend: Services):
    try: backend.animation_service().report_capabilities(request.bones, request.expressions)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'ok': True}

class WalkDestination(BaseModel):
    x: StrictInt
    y: StrictInt

@router.post('/api/animation/walk')
def animation_walk(request: WalkDestination, backend: Services):
    try: action = backend.animation_service().walk_to(request.x, request.y)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'action_id': action.id}

@router.post('/api/animation/stop')
def animation_stop(backend: Services):
    service, session = backend.animation_service(), backend.session
    service.stop_movement()
    for action in session.actions.active():
        if action['kind'] == 'motion.preview': session.actions.cancel(action['id'])
    return {'ok': True}
