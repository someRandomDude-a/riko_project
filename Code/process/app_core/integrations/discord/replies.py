"""Coalesced streamed Discord edits, driven by incoming model deltas."""
import asyncio

from .media import split_text


class StreamReply:
    def __init__(self, send, *, prefix=''):
        self.send, self.text, self.messages, self.rendered = send, '', [], []
        self.timer = None
        self.lock = asyncio.Lock()
        self.closed = False
        self.prefix = prefix
        self.error = None

    def feed(self, delta):
        if self.closed: return
        self.text = (self.text + delta)[:64000]
        if self.timer is None: self.timer = asyncio.create_task(self._scheduled())

    async def _scheduled(self):
        try:
            await asyncio.sleep(.8) # One coalesced edit after data arrives; not polling.
            await self.flush()
        except Exception as exc: self.error = exc
        finally: self.timer = None

    async def flush(self):
        async with self.lock:
            if not self.text: return
            for index, content in enumerate(split_text(self.text)):
                content = self.prefix + content
                if index >= len(self.messages):
                    self.messages.append(await self.send(content))
                    self.rendered.append(content)
                elif self.rendered[index] != content:
                    await self.messages[index].edit(content=content)
                    self.rendered[index] = content

    async def finish(self, text=None):
        self.closed = True
        if self.timer:
            self.timer.cancel()
            try: await self.timer
            except asyncio.CancelledError: pass
            self.timer = None
        if self.error: raise self.error
        if text is not None: self.text = text[:64000]
        await self.flush()
        # Authoritative final output can be shorter than speculative streamed text.
        count = len(split_text(self.text)) if self.text else 0
        for message in self.messages[count:]: await message.delete()
