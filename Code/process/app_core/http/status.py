"""/api/status and /ws/events: the full state snapshot, then resource.snapshot, then every bus event (events/stream.py)."""
from fastapi import APIRouter, WebSocket

from ..events.stream import stream_events
from .backend import Services

router = APIRouter()


@router.get("/api/status")
def status(backend: Services): return backend.snapshot()

@router.websocket("/ws/events")
async def events(websocket: WebSocket, backend: Services):
    await stream_events(websocket, backend.bus, backend.snapshot, initial=lambda: backend.resource_snapshot() or {})
