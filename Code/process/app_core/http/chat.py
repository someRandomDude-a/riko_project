"""/api/chat and /ws/chat: a typed turn through the session (a busy session is 409), Stop, and the archived history."""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..kernel.cancellation import TurnBusy, TurnCancelled
from .backend import Services

router = APIRouter()
BUSY = 'Riko is still handling another reply; send again when it finishes'


class ChatRequest(BaseModel):
    text: str
    user_name: str = "User"

@router.post("/api/chat")
def chat_endpoint(request: ChatRequest, backend: Services):
    # Plain def: FastAPI runs it in the thread pool, so the state/snapshot calls below
    # (which take session locks) never block the event loop.
    try:
        response = backend.session.respond(request.text, request.user_name)
    except TurnCancelled:
        return {"cancelled": True}
    except TurnBusy as exc:
        # Another turn (voice, Discord, a reply still stopping) holds the session: a conflict, not a server failure.
        raise HTTPException(409, BUSY) from exc
    backend.state.set_speech(response.message.content)
    return {"text": response.message.content, "emotion": backend.snapshot()["emotion"]}

@router.post("/api/chat/stop")
def stop_chat(backend: Services):
    backend.session.cancel()
    return {"stopping": True}

@router.get('/api/chat/history')
def chat_history(backend: Services, before: int | None = None, limit: int = 40):
    if before is not None and before < 1 or not 1 <= limit <= 100: raise HTTPException(400, 'Invalid history page')
    if backend.conversation_store is None: raise HTTPException(503, 'History store is not ready')
    return backend.conversation_store.page(before, limit)

@router.websocket("/ws/chat")
async def chat_stream(websocket: WebSocket, backend: Services):
    await websocket.accept()
    try:
        while True:
            request = await websocket.receive_json()
            text = str(request.get("text", "")).strip()
            if not text: continue
            await websocket.send_json({"type": "chat.started"})
            # Tool-enabled turns use the reliable completion path; this endpoint
            # is reserved for providers that support token streaming.
            try: response = await run_in_threadpool(backend.session.respond, text, request.get("user_name", "User"))
            except TurnBusy:  # the socket's 409: say so and keep the connection for the next message
                await websocket.send_json({"type": "chat.busy", "payload": {"status": 409, "detail": BUSY}}); continue
            except TurnCancelled:
                await websocket.send_json({"type": "chat.cancelled", "payload": {}}); continue
            await websocket.send_json({"type": "chat.completed", "payload": {"text": response.message.content}})
    except (WebSocketDisconnect, RuntimeError):
        return
