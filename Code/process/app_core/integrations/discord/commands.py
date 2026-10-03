"""Slash commands with one explicit access gate and narrow administrative edits."""
import json
from urllib.parse import quote

import discord
from discord import app_commands
from discord.ext import commands

from .backend import BackendError
from .bot import Job
from .config import BASIC_SETTINGS
from .views import ConfirmView


class CompanionCommands(commands.Cog):
    def __init__(self, bot): self.bot = bot

    def require(self, interaction, *, admin=False):
        if not self.bot.settings.allows(interaction.user.id, interaction.channel_id, guild=interaction.guild is not None, admin=admin):
            raise app_commands.CheckFailure('This command requires a configured trusted user/admin and an allowed channel.')

    async def reply(self, interaction, text):
        options = {'ephemeral': True, 'allowed_mentions': discord.AllowedMentions.none()}
        if interaction.response.is_done(): await interaction.followup.send(str(text)[:1900], **options)
        else: await interaction.response.send_message(str(text)[:1900], **options)

    async def cog_app_command_error(self, interaction, error):
        cause = getattr(error, 'original', error)
        text = str(cause) if isinstance(cause, (BackendError, ValueError, app_commands.CheckFailure)) else 'Command failed; check bot permissions and backend availability.'
        await self.reply(interaction, text)

    @app_commands.command(name='chat', description='Chat with the shared companion using streamed text.')
    async def chat(self, interaction: discord.Interaction, text: app_commands.Range[str, 1, 6000]):
        self.require(interaction)
        await interaction.response.defer(ephemeral=True)
        accepted = await self.bot.enqueue(Job(interaction.channel, interaction.user, text))
        await self.reply(interaction, 'Queued. The reply will appear in this channel.' if accepted else 'Not queued.')

    @app_commands.command(name='speak', description='Audio is paused; chat remains text-only.')
    async def speak(self, interaction: discord.Interaction, text: app_commands.Range[str, 1, 2000]):
        self.require(interaction)
        await self.reply(interaction, 'Calling/audio are paused. Text chat and whiteboard image updates remain available.')

    @app_commands.command(name='transcribe', description='Call transcription is paused; send text instead.')
    async def transcribe(self, interaction: discord.Interaction, audio: discord.Attachment, respond: bool = False):
        self.require(interaction)
        await self.reply(interaction, 'Calling/audio transcription are paused. Send text instead.')

    @app_commands.command(name='reasoning', description='Show/hide reasoning separately from assistant replies here.')
    async def reasoning(self, interaction: discord.Interaction, enabled: bool):
        self.require(interaction, admin=True)
        self.bot.preferences.set(interaction.channel_id, 'reasoning', enabled)
        await self.reply(interaction, f'Separate reasoning visibility: {enabled}.')

    @app_commands.command(name='join', description='Calling is paused pending the custom Discord client.')
    @app_commands.guild_only()
    async def join(self, interaction: discord.Interaction, receive: bool = False):
        self.require(interaction, admin=True)
        await self.reply(interaction, 'Calling is paused until the custom Discord client is implemented. Chat, tools and commands still work.')

    @app_commands.command(name='leave', description='Disconnect the companion from its server call.')
    @app_commands.guild_only()
    async def leave(self, interaction: discord.Interaction):
        self.require(interaction, admin=True)
        await interaction.response.defer(ephemeral=True)
        await self.bot.leave_call()
        await self.reply(interaction, 'Calling is paused; any leftover call connection was safely disconnected.')

    @app_commands.command(name='listen', description='Call transcription is paused; no voice is recorded.')
    @app_commands.guild_only()
    async def listen(self, interaction: discord.Interaction, enabled: bool):
        self.require(interaction)
        await self.reply(interaction, 'Calling/audio transcription are paused; no voice is being recorded.')

    @app_commands.command(name='stop', description='Stop your active and queued replies in this channel.')
    async def stop(self, interaction: discord.Interaction):
        self.require(interaction)
        await interaction.response.defer(ephemeral=True)
        admin = self.bot.settings.admin_actions and interaction.user.id in self.bot.settings.admins
        await self.bot.stop_channel(interaction.channel_id, None if admin else interaction.user.id)
        await self.reply(interaction, 'Stopped your replies/queue here. Desktop turns belonging to another request were not cancelled.')

    @app_commands.command(name='status', description='Show actual backend, model, emotion and Discord voice availability.')
    async def status(self, interaction: discord.Interaction):
        self.require(interaction)
        await interaction.response.defer(ephemeral=True)
        state = await self.bot.backend.request('GET', '/api/status')
        runtime = state.get('runtime', {})
        voice = self.bot.voice_for(interaction.channel)
        result = {'character': state.get('character_name'), 'generating': runtime.get('generating'),
            'backend_events': self.bot.backend.ready.is_set(), 'queue': self.bot.queue.qsize(),
            'emotion': state.get('emotion'), 'voice_connected': bool(voice),
            'call_stt': bool(self.bot.capture and not self.bot.capture.closed),
            'calling': 'paused', 'camera_transport': 'disabled', 'audio_transport': 'disabled', 'whiteboard': 'change-triggered PNG images'}
        await self.reply(interaction, json.dumps(result, ensure_ascii=False, indent=2))

    @app_commands.command(name='settings', description='Read/edit approved basic settings; model changes require backend restart.')
    async def settings(self, interaction: discord.Interaction, key: str | None = None, value: str | None = None):
        self.require(interaction, admin=True)
        if key and key not in BASIC_SETTINGS: raise ValueError('That setting is not available through Discord')
        if value is not None and not key: raise ValueError('Specify a setting key')
        await interaction.response.defer(ephemeral=True)
        snapshot = await self.bot.backend.request('GET', '/api/settings')
        if value is None:
            values = {k: v for k, v in snapshot['values'].items() if k in BASIC_SETTINGS and (not key or k == key)}
            await self.reply(interaction, json.dumps(values, ensure_ascii=False, indent=2)); return
        try: parsed = json.loads(value)
        except ValueError as exc: raise ValueError('Use a JSON number or boolean for this setting') from exc
        result = await self.bot.backend.request('PUT', '/api/settings', body={'changes': {key: parsed}, 'revision': snapshot['revision']})
        await self.reply(interaction, 'Saved. Restart Python to apply runtime/model settings.' if result.get('saved') else json.dumps(result.get('errors', {})))

    @app_commands.command(name='tool_policy', description='Require or remove approval for one exact registered tool name.')
    async def tool_policy(self, interaction: discord.Interaction, name: str, required: bool):
        self.require(interaction, admin=True)
        await interaction.response.defer(ephemeral=True)
        await self.bot.backend.request('PUT', '/api/tools/approvals', body={'policy': {name: required}})
        await self.reply(interaction, f'Approval requirement for {name}: {required}.')

    @app_commands.command(name='tasks', description='Find durable tasks when relevant, without injecting them into every reply.')
    async def tasks(self, interaction: discord.Interaction, query: str = ''):
        self.require(interaction)
        await interaction.response.defer(ephemeral=True)
        result = await self.bot.backend.request('GET', '/api/tasks?include_closed=false&query=' + quote(query[:200], safe=''))
        lines = [f"{task['id']} · revision {task['revision']} · {task['status']} · {task['title']}" for task in result['tasks'][:12]]
        await self.reply(interaction, '\n'.join(lines) or 'No matching tasks.')

    @app_commands.command(name='task_create', description='Create an explicit durable task.')
    async def task_create(self, interaction: discord.Interaction, title: app_commands.Range[str, 1, 200], next_step: str = ''):
        self.require(interaction, admin=True)
        await interaction.response.defer(ephemeral=True)
        task = await self.bot.backend.request('POST', '/api/tasks', body={'title': title, 'next_step': next_step[:2000]})
        await self.reply(interaction, f"Created {task['id']} at revision {task['revision']}.")

    @app_commands.command(name='task_update', description='Update a task using its current revision and an explicit reason.')
    async def task_update(self, interaction: discord.Interaction, task_id: str, revision: app_commands.Range[int, 1], next_step: str, reason: app_commands.Range[str, 1, 1000]):
        self.require(interaction, admin=True)
        await interaction.response.defer(ephemeral=True)
        task = await self.bot.backend.request('PATCH', '/api/tasks/' + quote(task_id, safe=''), body={
            'expected_revision': revision, 'changes': {'next_step': next_step[:2000]}, 'reason': reason})
        await self.reply(interaction, f"Updated task to revision {task['revision']}.")

    @app_commands.command(name='initiative', description='Read/toggle initiative and explicitly choose whether to relay it here.')
    async def initiative(self, interaction: discord.Interaction, enabled: bool | None = None, relay_here: bool | None = None):
        self.require(interaction, admin=True)
        if relay_here and interaction.channel_id not in self.bot.settings.channels: raise ValueError('Initiative relay requires a whitelisted server text channel')
        await interaction.response.defer(ephemeral=True)
        method, body = ('GET', None) if enabled is None else ('PUT', {'enabled': enabled})
        result = await self.bot.backend.request(method, '/api/initiative', body=body)
        if relay_here is not None:
            self.bot.preferences.initiative_channel = interaction.channel_id if relay_here else None
            self.bot.preferences.save()
        await self.reply(interaction, json.dumps({'enabled': result['settings']['enabled'], 'busy': result['busy'],
            'error': result['error'], 'relay_channel': self.bot.preferences.initiative_channel}, ensure_ascii=False))

    @app_commands.command(name='memories', description='Inspect a small page of shared companion memories (admin only).')
    async def memories(self, interaction: discord.Interaction):
        self.require(interaction, admin=True)
        await interaction.response.defer(ephemeral=True)
        result = await self.bot.backend.request('GET', '/api/memories')
        records = result['records'][:8]
        await self.reply(interaction, '\n'.join(f"{item['id']}: {item['text'][:150]}" for item in records) or 'No memories.')

    @app_commands.command(name='history', description='Read recent messages in this Discord channel; does not expose global memory.')
    async def history(self, interaction: discord.Interaction, count: app_commands.Range[int, 1, 10] = 5):
        self.require(interaction)
        await interaction.response.defer(ephemeral=True)
        messages = [f'{message.id} · {message.author.display_name}: {message.content[:120]}' async for message in interaction.channel.history(limit=count)]
        await self.reply(interaction, '\n'.join(reversed(messages)) or 'No messages.')

    @app_commands.command(name='memory', description='Inspect or explicitly enable/disable one shared memory (admin only).')
    async def memory(self, interaction: discord.Interaction, memory_id: str, active: bool | None = None):
        self.require(interaction, admin=True)
        await interaction.response.defer(ephemeral=True)
        if active is not None:
            result = await self.bot.backend.request('PATCH', '/api/memories/' + quote(memory_id, safe=''), body={'active': active})
            await self.reply(interaction, f'Memory active: {active}.'); return
        result = await self.bot.backend.request('GET', '/api/memories')
        record = next((item for item in result['records'] if item['id'] == memory_id), None)
        if record is None: raise ValueError('Memory not found')
        await self.reply(interaction, json.dumps({key: record.get(key) for key in ('id','text','active','importance','memory_type','tags')}, ensure_ascii=False))

    @app_commands.command(name='resources', description='Read GPU residency and owned-component estimates without loading models.')
    async def resources(self, interaction: discord.Interaction):
        self.require(interaction, admin=True)
        await interaction.response.defer(ephemeral=True)
        result = await self.bot.backend.request('GET', '/api/resources/gpu')
        allowed = ('name', 'total_mib', 'used_mib', 'free_mib', 'owned_mib', 'other_mib')
        summary = {'available': result.get('available'), 'gpus': [
            {key: gpu.get(key) for key in allowed if key in gpu} for gpu in result.get('gpus', [])]}
        await self.reply(interaction, json.dumps(summary, ensure_ascii=False, indent=2))

    @app_commands.command(name='animation', description='Read/preview the desktop animation library; Discord has no avatar renderer.')
    async def animation(self, interaction: discord.Interaction, preview_id: str | None = None):
        self.require(interaction, admin=True)
        await interaction.response.defer(ephemeral=True)
        if preview_id:
            result = await self.bot.backend.request('POST', '/api/animation/assets/' + quote(preview_id, safe='') + '/preview', body={})
            await self.reply(interaction, f"Preview queued on the desktop renderer: {result['action_id']}. This is not a Discord camera stream."); return
        result = await self.bot.backend.request('GET', '/api/animation')
        summary = {'pending': result.get('pending'), 'error': result.get('error'), 'current': result.get('current'),
            'entries': [{key: entry.get(key) for key in ('id','name','states')} for entry in result.get('entries', [])[:8]]}
        await self.reply(interaction, json.dumps(summary, ensure_ascii=False, indent=2))

    @app_commands.command(name='whiteboard', description='Show the board image in text chat, or move its desktop surface during a call.')
    async def whiteboard(self, interaction: discord.Interaction, x: int | None = None, y: int | None = None):
        self.require(interaction, admin=True)
        if (x is not None or y is not None) and not self.bot.voice_for(interaction.channel):
            raise ValueError('Whiteboard positioning is available during a call')
        await interaction.response.defer(ephemeral=True)
        self.bot.board_target = interaction.channel
        if self.bot.voice_for(interaction.channel):
            geometry = {key: value for key, value in {'x': x, 'y': y}.items() if value is not None}
            await self.bot.backend.request('PATCH', '/api/surfaces/whiteboard', body={'visible': True, **({'geometry': geometry} if geometry else {})})
            await self.reply(interaction, 'Whiteboard shown as a movable desktop surface. Drag/resize its window or use x/y here. Publishing it in Discord video requires a separate video bridge.')
        else:
            await self.bot.schedule_board_update()
            await self.reply(interaction, 'Whiteboard updates will be sent as images in this channel.')

    @app_commands.command(name='edit', description='Edit one bot-authored Discord message; model history is not rewritten.')
    async def edit(self, interaction: discord.Interaction, message_id: str, text: app_commands.Range[str, 1, 1900]):
        self.require(interaction, admin=True)
        if not message_id.isdigit(): raise ValueError('Use a Discord message ID')
        await interaction.response.defer(ephemeral=True)
        message = await interaction.channel.fetch_message(int(message_id))
        if message.author.id != self.bot.user.id: raise ValueError('Only bot-authored messages may be edited')
        await message.edit(content=text, allowed_mentions=discord.AllowedMentions.none())
        await self.reply(interaction, 'Discord message edited; durable companion history is unchanged.')

    @app_commands.command(name='messages', description='Confirm deletion of bot messages or a small channel purge; no memory reset.')
    @app_commands.choices(action=[app_commands.Choice(name='Delete bot messages', value='delete_bot'), app_commands.Choice(name='Purge channel messages', value='purge')])
    async def messages(self, interaction: discord.Interaction, action: str, count: app_commands.Range[int, 1, 100] = 10):
        self.require(interaction, admin=True)
        if action not in {'delete_bot', 'purge'}: raise ValueError('Unsupported action')
        def check_permissions():
            self.require(interaction, admin=True)
            if action == 'purge':
                if not interaction.guild: raise ValueError('Purging is only available in server channels')
                if not interaction.channel.permissions_for(interaction.user).manage_messages or not interaction.channel.permissions_for(interaction.guild.me).manage_messages:
                    raise ValueError('You and the bot both need Manage Messages permission')
        check_permissions()
        async def perform():
            check_permissions()
            deleted = 0
            async for message in interaction.channel.history(limit=count):
                if action == 'purge' or message.author.id == self.bot.user.id:
                    await message.delete(); deleted += 1
            return f'Deleted {deleted} Discord messages. Companion history/memory are unchanged.'
        view = ConfirmView(interaction.user.id, perform)
        await interaction.response.send_message(f'Delete {action} within the last {count} messages? This does not erase model history or memories.', view=view, ephemeral=True)

    @app_commands.command(name='camera', description='Show the paused video-calling status; no camera is captured.')
    async def camera(self, interaction: discord.Interaction):
        self.require(interaction, admin=True)
        text = 'Video calling is paused pending the custom Discord client. No camera or screen feed is captured. Chat and whiteboard images remain available.'
        await self.reply(interaction, text)
