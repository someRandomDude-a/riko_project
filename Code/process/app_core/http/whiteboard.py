"""/api/whiteboard/image: the board as a PNG for the model, and the renderer's capture of it (desktop/whiteboard_image.py)."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from .backend import Services

router = APIRouter()


@router.get('/api/whiteboard/image')
def whiteboard_image(backend: Services):
    return Response(backend.board_images.image(backend.state.snapshot(), backend.config.root), media_type='image/png', headers={'Cache-Control': 'no-store'})

@router.post('/api/whiteboard/image')
async def whiteboard_capture(request: Request, revision: str, backend: Services):
    if request.headers.get('content-type', '').split(';')[0] != 'image/png': raise HTTPException(415, 'Send a PNG capture of the whiteboard only')
    png = bytearray()
    async for chunk in request.stream():
        if len(png) + len(chunk) > 8 * 1024 * 1024: raise HTTPException(413, 'Board image exceeds 8 MiB')
        png.extend(chunk)
    # PIL verification, hashing and the publish run off the event loop.
    try: accepted = await run_in_threadpool(lambda: backend.board_images.capture(revision, bytes(png), backend.state.snapshot(), backend.bus))
    except (ValueError, OSError) as exc: raise HTTPException(400, 'Invalid board image') from exc
    return {'accepted': accepted}
