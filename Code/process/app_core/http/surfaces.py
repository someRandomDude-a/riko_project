"""/api/surfaces and /api/displays: the renderer's acknowledgements, the avatar and whiteboard windows' geometry and the
displays Electron reports (desktop/geometry.py, desktop/whiteboard.py)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictInt

from .backend import Services

router = APIRouter()


class SurfaceResult(BaseModel):
    surface: str
    command_id: str
    status: str
    error: str = ''
    bounds: dict[str, float] | None = None

@router.post('/api/surfaces/result')
def surface_result(request: SurfaceResult, backend: Services):
    if len(request.error) > 2000: raise HTTPException(400, 'Error exceeds 2000 characters')
    try: return {'accepted': backend.state.surface_result(**request.model_dump())}
    except ValueError as exc: raise HTTPException(400, str(exc))

class BoardSurface(BaseModel):
    visible: bool | None = None
    geometry: dict[str, StrictInt] | None = None

@router.patch('/api/surfaces/whiteboard')
def board_surface(request: BoardSurface, backend: Services):
    try: backend.state.set_whiteboard_surface(**request.model_dump())
    except ValueError as exc: raise HTTPException(400, str(exc))
    return backend.snapshot()

class DisplayBounds(BaseModel):
    x: StrictInt
    y: StrictInt
    width: StrictInt = Field(gt=0)
    height: StrictInt = Field(gt=0)

class Display(BaseModel):
    index: StrictInt = Field(ge=0)
    id: StrictInt
    label: str = Field(max_length=512)
    primary: bool
    bounds: DisplayBounds
    scaleFactor: float = Field(gt=0, allow_inf_nan=False)

@router.post('/api/displays')
def displays(request: list[Display], backend: Services):
    indices = [item.index for item in request]
    if not request or len(request) > 32 or indices != list(range(len(request))) or sum(item.primary for item in request) != 1:
        raise HTTPException(400, 'Provide ordered display indices and exactly one primary display')
    backend.state.set_displays([item.model_dump() for item in request])
    return {'accepted': True}

@router.patch('/api/surfaces/avatar')
def avatar_surface(request: dict[str, StrictInt], backend: Services):
    try: backend.state.update_geometry('avatar', **request)
    except ValueError as exc: raise HTTPException(400, str(exc))
    backend.state.save_avatar_geometry()  # desktop_settings.json, in order and outside the geometry's lock
    return backend.snapshot()
