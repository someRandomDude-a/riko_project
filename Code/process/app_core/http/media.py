"""/api/media: an image or video under the approved asset roots (desktop/media.resolve_media), never an arbitrary path."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..desktop.media import resolve_media
from .backend import Services

router = APIRouter()


@router.get('/api/media')
def media(path: str, backend: Services):
    try: asset = resolve_media(backend.config.root, path, backend.effects_directory())
    except ValueError as exc: raise HTTPException(400, str(exc))
    return FileResponse(asset)
