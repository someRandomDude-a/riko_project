"""One event subscription, no status polling and no secondary model process."""
import asyncio
import contextlib
import json
import logging
from uuid import uuid4

import aiohttp

logger = logging.getLogger(__name__)


class BackendError(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


class BackendClient:
    def __init__(self, base_url, on_event, on_disconnect):
        self.base_url, self.on_event, self.on_disconnect = base_url, on_event, on_disconnect
        self.http = None
        self.worker = None
        self.ready = asyncio.Event()
        self.closed = False
        self.instance = str(uuid4())
        self.socket = None

    async def report(self, value):
        socket = self.socket
        if socket is None or socket.closed: return False
        try: await socket.send_json(value); return True
        except (aiohttp.ClientError, ConnectionError): return False

    async def open(self):
        if self.http is not None: return
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600, connect=5))
        self.worker = asyncio.create_task(self._events(), name='discord-backend-events')

    async def request(self, method, path, *, body=None, data=None, binary=False):
        if self.http is None: raise BackendError(503, 'Backend connection is not open')
        headers = {'Content-Type': 'application/octet-stream'} if data is not None else None
        try:
            async with self.http.request(method, self.base_url + path, json=body, data=data, headers=headers) as response:
                if response.status >= 400:
                    detail = 'Companion backend unavailable; check the Python server.'
                    if response.status in {400, 409, 413, 415, 422}:
                        with contextlib.suppress(ValueError, aiohttp.ContentTypeError):
                            detail = str((await response.json()).get('detail', detail))[:1000]
                    raise BackendError(response.status, detail)
                if binary:
                    audio = bytearray()
                    async for chunk in response.content.iter_chunked(8192):
                        if len(audio) + len(chunk) > 8 * 1024 * 1024 + 128: raise BackendError(413, 'Media export is too large')
                        audio.extend(chunk)
                    return bytes(audio)
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise BackendError(503, 'Backend connection failed; the request is not automatically retried.') from exc

    async def _events(self):
        while not self.closed:
            try:
                async with self.http.ws_connect(self.base_url + '/ws/discord/client?instance=' + self.instance, max_msg_size=4 * 1024 * 1024) as socket:
                    self.socket = socket
                    async for message in socket:
                        if message.type != aiohttp.WSMsgType.TEXT: continue
                        try: event = json.loads(message.data)
                        except ValueError: continue
                        if not isinstance(event, dict) or not isinstance(event.get('type'), str): continue
                        if event['type'] == 'resource.snapshot': self.ready.set()
                        try: await self.on_event(event)
                        except Exception: logger.exception('Discord event delivery failed')
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                logger.warning('Companion event connection unavailable')
            finally:
                self.socket = None
                was_ready = self.ready.is_set()
                self.ready.clear()
                if was_ready and not self.closed: await self.on_disconnect()
            if not self.closed: await asyncio.sleep(1) # Recovery only, not a status poll.

    async def close(self):
        self.closed = True
        self.ready.clear()
        if self.worker:
            self.worker.cancel()
            with contextlib.suppress(asyncio.CancelledError): await self.worker
        if self.http: await self.http.close()
