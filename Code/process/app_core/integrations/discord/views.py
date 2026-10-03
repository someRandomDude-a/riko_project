"""Administrative confirmations and exact-call approval buttons."""
import time

import discord


class ConfirmView(discord.ui.View):
    def __init__(self, owner_id, action):
        super().__init__(timeout=30)
        self.owner_id, self.action, self.used = owner_id, action, False

    async def interaction_check(self, interaction):
        if interaction.user.id != self.owner_id or self.used:
            await interaction.response.send_message('This confirmation is not available to you.', ephemeral=True)
            return False
        return True

    @discord.ui.button(label='Confirm', style=discord.ButtonStyle.danger)
    async def confirm(self, interaction, button):
        self.used = True
        await interaction.response.defer(ephemeral=True)
        try:
            message = await self.action()
            await interaction.edit_original_response(content=message, view=None)
        except Exception:
            await interaction.edit_original_response(content='Operation failed; check bot permissions and backend availability.', view=None)
        self.stop()

    @discord.ui.button(label='Cancel', style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction, button):
        self.used = True
        await interaction.response.edit_message(content='Cancelled.', view=None)
        self.stop()


class ApprovalView(discord.ui.View):
    def __init__(self, bot, request, job):
        super().__init__(timeout=max(0, min(120, request['expires_at'] - time.time())))
        self.bot, self.request, self.job = bot, request, job
        self.message = None
        self.finished = False

    async def interaction_check(self, interaction):
        allowed = self.bot.settings.allows(interaction.user.id, interaction.channel_id,
            guild=interaction.guild is not None, admin=True)
        if not allowed or self.bot.active is not self.job or self.finished:
            await interaction.response.send_message('Only a configured admin can resolve this active tool call.', ephemeral=True)
            return False
        return True

    async def decide(self, interaction, approved):
        await interaction.response.defer(ephemeral=True)
        try:
            await self.bot.backend.request('POST', '/api/tools/approvals/' + self.request['id'], body={'approved': approved})
            await interaction.followup.send('Approved once.' if approved else 'Denied.', ephemeral=True)
        except Exception:
            await interaction.followup.send('Request expired, was cancelled or is no longer available.', ephemeral=True)
        await self.finish()

    async def finish(self):
        self.finished = True
        self.stop()
        for item in self.children: item.disabled = True
        if self.message:
            try: await self.message.edit(view=self)
            except discord.HTTPException: pass

    async def on_timeout(self): await self.finish()

    @discord.ui.button(label='Approve once', style=discord.ButtonStyle.success)
    async def approve(self, interaction, button): await self.decide(interaction, True)

    @discord.ui.button(label='Deny', style=discord.ButtonStyle.danger)
    async def deny(self, interaction, button): await self.decide(interaction, False)
