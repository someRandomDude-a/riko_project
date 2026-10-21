"""Coalesced streamed Discord edits, driven by incoming model deltas."""
import asyncio
import logging
import secrets

from .media import split_text

logger = logging.getLogger(__name__)


class StreamReply:
    def __init__(self, send, *, prefix=''):
        self.send, self.text, self.messages, self.rendered = send, '', [], []
        self.nonces = []  # one per chunk: a send retried after an ambiguous failure gets Discord's existing message back
        self.timer = None
        self.lock = asyncio.Lock()
        self.closed = False
        self.finished = False
        self.prefix = prefix

    def feed(self, delta):
        if self.closed: return
        self.text = (self.text + delta)[:64000]
        if self.timer is None: self.timer = asyncio.create_task(self._scheduled())

    async def _scheduled(self):
        try:
            await asyncio.sleep(.8) # One coalesced edit after data arrives; not polling.
            await self.flush()
        # A failed streamed update costs only that update: the next one and finish() retry it.
        except Exception as exc: logger.warning('Discord streamed update failed; the final reply is still sent: %r', exc)
        finally: self.timer = None

    async def flush(self):
        """Every chunk is attempted; the first failure is raised afterwards. A failed send ends the
        pass, so a later chunk never posts ahead of it, and the next flush sends it again."""
        async with self.lock:
            if not self.text: return
            failure = None
            for index, content in enumerate(split_text(self.text)):
                content = self.prefix + content
                try:
                    if index >= len(self.messages):
                        retried = index < len(self.nonces)
                        while len(self.nonces) <= index: self.nonces.append(secrets.randbits(63))
                        self.messages.append(await self.send(content, nonce=self.nonces[index]))
                        # A retry may get back the message an earlier attempt made, with that attempt's text.
                        self.rendered.append(None if retried else content)
                        if retried:
                            await self.messages[index].edit(content=content)
                            self.rendered[index] = content
                    elif self.rendered[index] != content:
                        await self.messages[index].edit(content=content)
                        self.rendered[index] = content
                except Exception as exc:
                    failure = failure or exc
                    if index >= len(self.messages): break
            if failure: raise failure

    async def finish(self, text=None):
        if self.finished: return # Idempotent: the turn queue's cleanup finishes every reply again.
        self.closed = True
        if self.timer:
            self.timer.cancel()
            try: await self.timer
            except asyncio.CancelledError: pass
            self.timer = None
        if text is not None: self.text = text[:64000]
        failure = None
        try: await self.flush() # Always attempted: an earlier streamed failure never discards the reply.
        except Exception as exc: failure = exc
        # Authoritative final output can be shorter than speculative streamed text.
        count = len(split_text(self.text)) if self.text else 0
        for message in self.messages[count:]:
            try: await message.delete()
            except Exception as exc: logger.warning('Discord could not delete a superseded streamed message: %r', exc)
        self.finished = True # Not reached when cancelled mid-flush, so a later finish() completes it.
        if failure: raise failure
