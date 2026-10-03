"""ASGI event transport, independent of model/service construction."""
import asyncio

from starlette.websockets import WebSocketDisconnect


async def stream_events(websocket, bus, snapshot, *, initial=None, event_filter=lambda event: True, on_message=None):
    await websocket.accept()
    queue = asyncio.Queue(maxsize=100)
    loop = asyncio.get_running_loop()
    closed = False
    overflowed = False

    def enqueue(payload):
        nonlocal overflowed
        if closed or overflowed: return
        if queue.full():
            # Never silently lose permission/state transitions. A reconnect
            # receives fresh snapshots and chat history rather than stale state.
            asyncio.create_task(websocket.close(code=1013))
            overflowed = True
            return
        queue.put_nowait(payload)

    def listener(event):
        if not event_filter(event): return
        try: loop.call_soon_threadsafe(enqueue, event.as_dict())
        except RuntimeError: pass

    async def send():
        await websocket.send_json({'type': 'state.snapshot', 'payload': snapshot()})
        watermark = 0
        if initial:
            watermark = bus.cursor()
            resources = await asyncio.to_thread(initial)
            await websocket.send_json({'type': 'resource.snapshot', 'payload': resources, 'sequence': watermark})
        while True:
            event = await queue.get()
            # Resource updates queued before bootstrap are superseded by its
            # fresh snapshots. Dialogue deltas/history are never discarded here.
            if initial and event['type'].startswith('resource.') and event.get('sequence', 0) <= watermark: continue
            await websocket.send_json(event)

    async def receive():
        # Waiting only on outgoing events leaves idle disconnected clients
        # subscribed forever. ASGI reports disconnect through receive().
        while True:
            message = await websocket.receive()
            if message['type'] == 'websocket.disconnect': return
            if on_message and message['type'] == 'websocket.receive':
                import json
                text = message.get('text', '')
                if not text or len(text) > 16384: await websocket.close(code=1009); return
                try: await on_message(json.loads(text))
                except (ValueError, TypeError, KeyError): await websocket.close(code=1008); return

    unsubscribe = bus.subscribe(listener)
    workers = [asyncio.create_task(send()), asyncio.create_task(receive())]
    try:
        done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
        for task in done: task.result()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        closed = True
        unsubscribe()
        for task in workers: task.cancel()
        try: await asyncio.gather(*workers, return_exceptions=True)
        except asyncio.CancelledError: pass # ASGI may cancel again during disconnect cleanup.
