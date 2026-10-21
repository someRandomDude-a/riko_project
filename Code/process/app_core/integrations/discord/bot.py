"""Discord is a transport client, never an owner of model/microphone workers."""
import asyncio
import contextlib
from dataclasses import dataclass, field
import io
import json
import logging
from pathlib import Path
from uuid import uuid4

import discord
from discord.ext import commands

from .backend import BackendClient, BackendError
from .media import AUDIO_EXTENSIONS, MAX_ATTACHMENT_BYTES, decode_audio, extract_pdf, split_text
from .preferences import Preferences
from .replies import StreamReply
from .views import ApprovalView
from .voice import VoiceCapture, downsample_call, receive_extension

logger = logging.getLogger(__name__)
CALLS_AVAILABLE = False # Paused until the custom Discord calling client is implemented.


@dataclass
class Job:
    channel: object
    user: object
    text: str = ''
    attachment: object = None
    pcm: bytes | None = None
    call_audio: bool = False
    transcribe_only: bool = False
    speech_only: bool = False
    message: object = None
    id: str = field(default_factory=lambda: str(uuid4()))
    stream: object = None
    reasoning: object = None
    task: object = None

    async def send(self, content=None, **kwargs):
        kwargs['allowed_mentions'] = discord.AllowedMentions.none()
        if self.message:
            try: return await self.message.reply(content, mention_author=False, **kwargs)
            except discord.NotFound: pass
        return await self.channel.send(content, **kwargs)


class CompanionBot(commands.Bot):
    def __init__(self, settings, *, preferences, backend_factory=BackendClient):
        """preferences: the channel preferences file (DataPaths.discord_preferences)."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(command_prefix='!', intents=intents, allowed_mentions=discord.AllowedMentions.none())
        self.settings = settings
        self.preferences = Preferences(preferences)
        self.backend = backend_factory(settings.backend_url, self.backend_event, self.backend_disconnected)
        self.queue = asyncio.Queue(maxsize=8)
        self.processor = None
        self.active = None
        self.approvals = {}
        self.capture = None
        self.call_text_channel = None
        self.closing = False
        self.voice_interrupted_job = None
        self.board_target = None
        self.board_update = None
        self.board_revision = None
        self.board_generation = 0

    async def setup_hook(self):
        from .commands import CompanionCommands
        await self.backend.open()
        await asyncio.wait_for(self.backend.ready.wait(), timeout=10)
        await self.add_cog(CompanionCommands(self))
        self.start_processor()
        if self.settings.sync_guild:
            guild = discord.Object(id=self.settings.sync_guild)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else: await self.tree.sync()

    async def on_ready(self):
        logger.info('Discord transport ready as %s; models belong to the Python backend', self.user)
        await self.report({'kind': 'ready', 'bot_name': str(self.user)})

    async def report(self, value):
        report = getattr(self.backend, 'report', None)
        if report: await report(value)

    async def on_disconnect(self): await self.report({'kind': 'disconnected'})

    async def on_resumed(self): await self.on_ready()

    async def observe_message(self, message):
        await self.report({'kind': 'message', 'message_id': str(message.id), 'user_id': str(message.author.id),
            'user_name': getattr(message.author, 'display_name', str(message.author)), 'channel_id': str(message.channel.id),
            'channel_name': str(getattr(message.channel, 'name', 'Direct message')),
            'guild_id': str(message.guild.id) if message.guild else None, 'text': message.content[:2000],
            'author_bot': bool(message.author.bot), 'webhook': bool(message.webhook_id),
            'attachments': [item.filename for item in message.attachments[:10]]})

    async def on_message(self, message):
        await self.observe_message(message)
        if message.author.bot or message.webhook_id: return
        if not self.settings.allows(message.author.id, message.channel.id, guild=message.guild is not None): return
        if message.content.startswith('!'):
            await self.process_commands(message)
            return
        attachments = message.attachments
        if len(attachments) > 1:
            await message.reply('Send at most one PDF attachment per message. Audio/calling are paused.', mention_author=False)
            return
        if not message.content.strip() and not attachments: return
        await self.enqueue(Job(message.channel, message.author, message.content,
            attachment=attachments[0] if attachments else None, message=message))

    async def enqueue(self, job):
        if self.closing: return False
        if not self.settings.allows(job.user.id, job.channel.id, guild=getattr(job.channel, 'guild', None) is not None): return False
        if job.message is None:
            guild = getattr(job.channel, 'guild', None)
            await self.report({'kind':'message','message_id':job.id,'user_id':str(job.user.id),
                'user_name':getattr(job.user,'display_name','Discord user'),'channel_id':str(job.channel.id),
                'channel_name':str(getattr(job.channel,'name','Direct message')),'guild_id':str(guild.id) if guild else None,'text':job.text[:2000]})
        try: self.queue.put_nowait(job)
        except asyncio.QueueFull:
            await job.send('The reply queue is full. Try again after the current replies finish.')
            return False
        return True

    def start_processor(self):
        self.processor = asyncio.create_task(self.process_queue(), name='discord-turn-queue')
        self.processor.add_done_callback(self.processor_stopped)

    def processor_stopped(self, task):
        # Cancelled means close() or loop shutdown; any other end is a bug that must not silence chat replies.
        if self.closing or task.cancelled() or task is not self.processor: return
        logger.error('Discord turn queue stopped unexpectedly; restarting it', exc_info=task.exception())
        self.start_processor()

    async def process_queue(self):
        # Discord's HTTP layer passes connection drops through as aiohttp/OS errors, not HTTPException,
        # so every Discord call below is guarded: one failed reply never ends the queue before close.
        while not self.closing:
            job = await self.queue.get()
            self.active = job
            try:
                job.task = asyncio.create_task(self.process_job(job))
                try: await job.task
                except asyncio.CancelledError: # A stopped turn ends here; the queue itself stops only when cancelled.
                    if self.closing or asyncio.current_task().cancelling(): raise
                except Exception as exc:
                    logger.exception('Discord turn failed')
                    text = str(exc) if isinstance(exc, (BackendError, ValueError)) else 'Request failed; check the bot/backend logs.'
                    try: await job.send(text[:1800])
                    except Exception as error: logger.warning('Discord could not report the failed turn: %r', error)
            finally:
                try: await self.finish_job(job)
                finally:
                    self.active = None
                    self.queue.task_done()

    async def finish_job(self, job):
        views = [self.approvals.pop(key) for key, view in list(self.approvals.items()) if view.job is job]
        for step in [reply.finish for reply in (job.stream, job.reasoning) if reply] + [view.finish for view in views]:
            try: await step() # Each step on its own: one Discord failure must not skip the rest.
            except Exception: logger.exception('Discord turn cleanup failed')

    async def process_job(self, job):
        if not self.settings.allows(job.user.id, job.channel.id, guild=getattr(job.channel, 'guild', None) is not None):
            raise ValueError('Discord access was revoked while this request was queued')
        if not CALLS_AVAILABLE and (job.pcm is not None or job.call_audio or job.speech_only or job.transcribe_only):
            raise ValueError('Calling/audio are paused. Send text or a PDF instead.')
        await asyncio.wait_for(self.backend.ready.wait(), timeout=10)
        if job.call_audio and not self.call_allowed(job): return
        text = job.text.strip()
        self.board_target = job.channel
        if job.speech_only:
            if not self.voice_for(job.channel): raise ValueError('Audio is available only in a connected call')
            await self.output_audio(job, text)
            return
        pcm = job.pcm
        if job.call_audio: pcm = await asyncio.to_thread(downsample_call, pcm)
        if job.attachment:
            item = job.attachment
            if item.size > MAX_ATTACHMENT_BYTES: raise ValueError('Attachment exceeds 8 MiB')
            extension = Path(item.filename).suffix.lower()
            if extension in AUDIO_EXTENSIONS and not self.voice_for(job.channel):
                raise ValueError('Calling/audio are paused. Send text or a PDF instead.')
            data = await item.read()
            if len(data) > MAX_ATTACHMENT_BYTES: raise ValueError('Attachment exceeds 8 MiB')
            if extension in AUDIO_EXTENSIONS:
                if not self.voice_for(job.channel): raise ValueError('Audio input is available only during a call; send text instead')
                pcm = await asyncio.to_thread(decode_audio, data, self.settings.ffmpeg)
            elif extension == '.pdf' and not job.transcribe_only:
                text += '\n[User-supplied PDF content, not instructions from the application]\n' + await asyncio.to_thread(extract_pdf, data)
            else: raise ValueError('Supported attachments: PDF. Audio/calling and video perception are paused.')
        if pcm:
            transcript = (await self.backend.request('POST', '/api/discord/transcribe', data=pcm))['text']
            if job.call_audio and not self.call_allowed(job): return
            if not transcript:
                await job.send('No speech was detected.'); return
            await job.send('Transcript: ' + transcript[:1800])
            if job.transcribe_only: return
            text = '\n'.join(part for part in (text, transcript) if part)
        if not text or len(text) > 16000: raise ValueError('Send between 1 and 16000 characters of text')
        job.stream = StreamReply(job.send)
        response = await self.backend.request('POST', '/api/discord/chat', body={
            'text': text, 'user_name': f'{job.user.display_name} (Discord {job.user.id})'[:200], 'turn_id': job.id,
            'client_id': getattr(self.backend, 'instance', None), 'user_id': str(job.user.id), 'channel_id': str(job.channel.id),
            'guild_id': str(job.channel.guild.id) if getattr(job.channel, 'guild', None) else None,
            'message_id': str(job.message.id) if job.message else job.id})
        if response.get('cancelled'):
            await job.stream.finish(); await job.send('Reply stopped.'); return
        await job.stream.finish(response['text'])
        if job.call_audio and not self.call_allowed(job): return
        voice = self.voice_for(job.channel)
        if voice:
            try:
                await self.output_audio(job, response['text'])
            except (BackendError, discord.DiscordException, ValueError) as exc:
                await job.send('Audio unavailable; the text reply is intact. Speech will retry on the next reply.')
                logger.warning('Discord speech unavailable: %s', type(exc).__name__)

    async def output_audio(self, job, text):
        for index, part in enumerate(split_text(text, 1800), 1):
            voice = self.voice_for(job.channel)
            if not voice: return # Leaving a call never falls back to audio attachments.
            wav = await self.backend.request('POST', '/api/discord/speech', body={'text': part}, binary=True)
            if self.voice_for(job.channel) is voice: await self.play_audio(voice, wav)

    def voice_for(self, channel):
        if not CALLS_AVAILABLE: return None
        guild = getattr(channel, 'guild', None)
        voice = getattr(guild, 'voice_client', None)
        return voice if voice and voice.is_connected() and self.call_text_channel and self.call_text_channel.id == channel.id else None

    def call_allowed(self, job):
        return bool(self.capture and not self.capture.closed and job.user.id in self.capture.consent
            and self.capture.accepts(job.user) and self.voice_for(job.channel))

    async def revoke_voice(self, channel_id, user_id):
        if self.capture: self.capture.revoke(user_id)
        await self.stop_channel(channel_id, user_id, call_only=True)

    async def play_audio(self, voice, wav):
        loop = asyncio.get_running_loop()
        finished = loop.create_future()
        def after(error):
            def done():
                if finished.done(): return
                if error: finished.set_exception(error)
                else: finished.set_result(None)
            with contextlib.suppress(RuntimeError): loop.call_soon_threadsafe(done)
        source = discord.FFmpegOpusAudio(io.BytesIO(wav), pipe=True, executable=self.settings.ffmpeg)
        try:
            voice.play(source, after=after)
            await asyncio.wait_for(finished, timeout=75)
        finally:
            if not finished.done(): finished.cancel()
            stop = getattr(voice, 'stop_playing', voice.stop)
            stop() # Do not stop the receive lane when cancelling output.
            source.cleanup()

    async def show_approval(self, request, job):
        if request.get('turn_id') != job.id or request['id'] in self.approvals: return
        arguments = json.dumps(request.get('arguments', {}), ensure_ascii=False, indent=2).encode()
        if len(arguments) > MAX_ATTACHMENT_BYTES:
            await self.backend.request('POST', '/api/tools/approvals/' + request['id'], body={'approved': False})
            await job.send('Tool arguments exceed the approval display limit; execution denied.'); return
        view = ApprovalView(self, request, job)
        self.approvals[request['id']] = view
        description = f"Tool approval required: **{discord.utils.escape_markdown(request['name'])}**\nRequest: `{request['id']}`\nOnly configured admins may approve this exact call."
        view.message = await job.send(description, view=view,
            file=discord.File(io.BytesIO(arguments), filename='tool-arguments.json'))

    async def backend_event(self, event):
        kind, payload = event['type'], event.get('payload', {})
        if kind in {'resource.snapshot', 'resource.discord'}:
            status = payload.get('discord', {}) if kind == 'resource.snapshot' else payload
            if status and status.get('settings'):
                from .access import apply_access
                self.settings = apply_access(self.settings, status['settings']['values'])
            if kind == 'resource.snapshot' and self.is_ready(): await self.on_ready()
        job = self.active
        if kind in {'whiteboard.changed', 'whiteboard.image'}:
            self.board_revision = payload.get('revision')
            self.schedule_board_update()
        if job and event.get('turn_id') == job.id:
            if kind == 'chat.delta' and job.stream: job.stream.feed(payload.get('text', ''))
            if kind == 'model.reasoning' and self.preferences.get(job.channel.id, 'reasoning'):
                if job.reasoning is None:
                    job.reasoning = StreamReply(job.send, prefix='**Reasoning (not the reply)**\n')
                job.reasoning.feed(payload.get('text', ''))
            if kind == 'tool.approval_requested':
                await self.show_approval({**payload, 'turn_id': job.id}, job)
        if job and kind in {'resource.snapshot', 'resource.approvals'}:
            resource = payload.get('approvals', {}) if kind == 'resource.snapshot' else payload
            for request in (resource or {}).get('pending', []): await self.show_approval(request, job)
        if kind in {'tool.approval_finished', 'tool.approval_resolved'}:
            view = self.approvals.pop(payload.get('id'), None)
            if view: await view.finish()
        target = self.preferences.initiative_channel
        if kind == 'initiative.presented' and target and target in self.settings.channels:
            channel = self.get_channel(target)
            if channel:
                for text in split_text(payload.get('message', '')): await channel.send(text)

    def schedule_board_update(self):
        self.board_generation += 1
        if self.board_update is None: self.board_update = asyncio.create_task(self.send_board_update())
        return self.board_update

    async def send_board_update(self):
        generation = self.board_generation
        try:
            await asyncio.sleep(.3) # Coalesce rendering acknowledgements from one mutation.
            generation = self.board_generation
            target = self.board_target
            if not target or self.closing: return
            if self.voice_for(target): return # The movable desktop surface is used during calls.
            image = await self.backend.request('GET', '/api/whiteboard/image', binary=True)
            await target.send('Whiteboard updated.', file=discord.File(io.BytesIO(image), filename='whiteboard.png'), allowed_mentions=discord.AllowedMentions.none())
        except (BackendError, discord.HTTPException): logger.warning('Whiteboard image delivery unavailable')
        finally:
            self.board_update = None
            if not self.closing and self.board_generation != generation:
                self.board_update = asyncio.create_task(self.send_board_update())

    async def backend_disconnected(self):
        if self.active:
            await self.stop_channel(self.active.channel.id)

    async def stop_channel(self, channel_id, user_id=None, *, call_only=False):
        job = self.active
        matched = job and job.channel.id == channel_id and (user_id is None or job.user.id == user_id) and (not call_only or job.call_audio)
        if matched:
            with contextlib.suppress(BackendError):
                await self.backend.request('POST', '/api/discord/stop', body={'turn_id': job.id})
            if job.task: job.task.cancel()
        for voice in self.voice_clients:
            if (matched or user_id is None) and self.call_text_channel and self.call_text_channel.id == channel_id:
                getattr(voice, 'stop_playing', voice.stop)()
        retained = []
        while not self.queue.empty():
            queued = self.queue.get_nowait(); self.queue.task_done()
            if queued.channel.id != channel_id or (user_id is not None and queued.user.id != user_id) or (call_only and not queued.call_audio): retained.append(queued)
        for queued in retained: self.queue.put_nowait(queued)

    async def join_call(self, interaction, receive=False):
        if not CALLS_AVAILABLE: raise ValueError('Calling is paused while the custom Discord client is planned. Chat and commands remain available.')
        channel = getattr(getattr(interaction.user, 'voice', None), 'channel', None)
        if not channel or channel.guild.id != interaction.guild_id: raise ValueError('Join a server voice channel first')
        if self.voice_clients: raise ValueError('Leave the existing call before joining another one')
        extension = receive_extension() if receive else None
        voice = await channel.connect(cls=extension.VoiceRecvClient if extension else discord.VoiceClient,
            self_deaf=not receive, timeout=20)
        self.call_text_channel = interaction.channel
        self.board_target = interaction.channel
        try: await self.backend.request('PATCH', '/api/surfaces/whiteboard', body={'visible': True})
        except BackendError: logger.warning('Desktop whiteboard surface unavailable')
        if receive:
            loop = asyncio.get_running_loop()
            def accepts(user):
                return self.settings.allows(user.id, interaction.channel_id, guild=True) and getattr(getattr(user, 'voice', None), 'channel', None) == channel
            def deliver(user, pcm):
                job = Job(interaction.channel, user, pcm=pcm, call_audio=True)
                if not self.closing and not self.queue.full(): self.queue.put_nowait(job)
                elif not self.closing:
                    asyncio.create_task(interaction.channel.send('Voice reply queue full; this utterance was not transcribed. Capture remains active.'))
            def activity(user, seconds):
                job = self.active
                if seconds >= 1.5 and job and job.channel.id == interaction.channel_id and job.id != self.voice_interrupted_job:
                    self.voice_interrupted_job = job.id
                    asyncio.create_task(self.stop_channel(interaction.channel_id))
            self.capture = VoiceCapture(loop, accepts, deliver, activity=activity)
            def after(error):
                if error:
                    def failed():
                        if self.capture: self.capture.close()
                        asyncio.create_task(interaction.channel.send('Call reception failed. STT has stopped; check the DAVE/voice extension installation.'))
                    with contextlib.suppress(RuntimeError): loop.call_soon_threadsafe(failed)
            try:
                voice.listen(extension.BasicSink(lambda user, data: self.capture.write(user, data.pcm) if self.capture else None), after=after)
            except Exception:
                self.capture.close(); self.capture = None
                await voice.disconnect(force=True)
                self.call_text_channel = None
                raise
        return 'Joined. Playback is available. ' + ('Each speaker must opt in with /listen enabled:true. Transcripts and replies appear in this text channel.' if receive else 'Call transcription is off; uploaded-audio STT remains available.')

    async def leave_call(self):
        if self.capture: self.capture.close(); self.capture = None
        if self.call_text_channel: await self.stop_channel(self.call_text_channel.id)
        for voice in list(self.voice_clients): await voice.disconnect(force=True)
        self.call_text_channel = None

    async def on_voice_state_update(self, member, before, after):
        if self.capture and before.channel != after.channel and self.call_text_channel:
            await self.revoke_voice(self.call_text_channel.id, member.id)
        if self.user and member.id == self.user.id and after.channel is None:
            if self.capture: self.capture.close(); self.capture = None
            if self.call_text_channel: await self.stop_channel(self.call_text_channel.id)
            self.call_text_channel = None

    async def close(self):
        if self.closing: return
        self.closing = True
        try:
            try:
                with contextlib.suppress(discord.DiscordException): await self.leave_call()
                if self.active: await self.stop_channel(self.active.channel.id)
                if self.processor:
                    self.processor.cancel()
                    with contextlib.suppress(asyncio.CancelledError): await self.processor
            finally: await self.backend.close()
        finally:
            if self.board_update:
                self.board_update.cancel()
                with contextlib.suppress(asyncio.CancelledError): await self.board_update
            await super().close()
