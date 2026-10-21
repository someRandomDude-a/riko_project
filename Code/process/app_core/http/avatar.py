"""/api/avatar: the configured VRM, the imported avatar library, wake animations and the renderer's animation results."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..desktop.media import resolve_media
from ..kernel.safe_paths import local_path
from .backend import Services
from .settings import PathCheck

router = APIRouter()


@router.get('/api/avatar/model')
def avatar_model(backend: Services):
    config = backend.config
    # Only the explicitly configured VRM, not arbitrary renderer-supplied paths; a network path is refused before it resolves.
    try: asset = local_path(config.avatar.get('model', 'character_files/Mita.vrm'), config.root).resolve()
    except ValueError: raise HTTPException(404, 'Configured VRM model not found') from None
    if asset.suffix.lower() != '.vrm' or not asset.is_file(): raise HTTPException(404, 'Configured VRM model not found')
    return FileResponse(asset, media_type='model/gltf-binary', headers={'Cache-Control': 'no-store'})

@router.get('/api/avatar/models')
def avatar_models(backend: Services): return backend.avatar_models()

@router.post('/api/avatar/models/import')
def import_avatar_model(request: PathCheck, backend: Services):
    from ..desktop.avatar_models import AvatarModels
    try: result = AvatarModels(backend.config.root).import_model(request.value)
    except (ValueError, OSError) as exc: raise HTTPException(400, str(exc)) from exc
    backend.bus.publish('avatar.models_changed')
    return result

@router.get('/api/avatar/animation')
def avatar_animation(path: str, backend: Services):
    try:
        asset = resolve_media(backend.config.root, path, backend.effects_directory(), extensions={'.vrma'})
    except ValueError as exc: raise HTTPException(400, str(exc))
    if asset.stat().st_size > 32 * 1024 * 1024: raise HTTPException(400, 'Animation asset exceeds 32 MiB')
    return FileResponse(asset, media_type='model/gltf-binary')

class AnimationResult(BaseModel):
    action_id: str
    status: str
    error: str = ''

@router.post('/api/avatar/animation/result')
def animation_result(result: AnimationResult, backend: Services):
    session = backend.session
    if session is None: raise HTTPException(503, 'Runtime is not ready')
    action = next((item for item in session.actions.active() if item['id'] == result.action_id and (item['kind'] == 'wake_animation' or item['kind'].startswith('motion.'))), None)
    if action is None: raise HTTPException(404, 'Animation action expired or replaced')
    if result.status not in {'started', 'completed', 'cancelled', 'error'}: raise HTTPException(400, 'Invalid animation status')
    if result.status == 'completed': session.actions.complete(result.action_id)
    elif result.status in {'error', 'cancelled'}:
        session.actions.cancel(result.action_id)
        if result.status == 'error' and getattr(session, 'animation', None): session.animation.renderer_error(action, result.error)
    backend.bus.publish('wake.feedback.error' if result.status == 'error' else 'wake.feedback.animation',
        action_id=result.action_id, asset='animation', status=result.status, error=result.error[:1000])
    return {'ok': True}
