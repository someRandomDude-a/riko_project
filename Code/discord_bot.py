from __future__ import annotations

import os
import asyncio
import threading
from queue import Queue, Empty
from datetime import datetime

import discord
from discord.ext import commands
from dotenv import load_dotenv

from process.common.config import char_config
from process.llm_scripts.module import llm_response
from process.llm_scripts.MCP_Tools import call_tool


# ENV / config

time_offset = datetime.now().astimezone().utcoffset()
load_dotenv()

_TOKEN = os.getenv("Discord_bot_token", "").strip()
_channel_whitelist = [
    int(c.strip())
    for c in os.getenv("Discord_Channel_whitelist", "").split(",")
    if c.strip()
]
_admins = [c.strip() for c in os.getenv("Discord_admins", "").split(",") if c.strip()]

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

_queue_max = char_config.get("discord", {}).get("queue_max_size", 3)
_resource_dispatch = char_config.get("discord", {}).get("resource_dispatch", {
    "application/pdf": "pdf_extractor",
})

llm_response_queue: Queue = Queue(_queue_max)
_queue_mutex = threading.Lock()                    # FIX for v2 qsize race
discord_loop: asyncio.AbstractEventLoop | None = None


# Helpers

async def _safe_edit(msg: discord.Message, text: str):
    if msg.content == text:
        return
    try:
        await msg.edit(content=text)
    except discord.HTTPException:
        # Rate-limited or too long; silently ignore for now
        pass


def _schedule_edit(msg: discord.Message, text: str):
    if discord_loop is None:
        return
    asyncio.run_coroutine_threadsafe(_safe_edit(msg, text), discord_loop)


# Worker thread — sequential, single LLM in flight at a time

def worker():
    while True:
        try:
            message = llm_response_queue.get()
        except Exception:
            continue

        user_text = f"{message.author.display_name}: {message.content}"

        # RESOURCE-style attachments (PDFs, etc.)
        for attachment in message.attachments:
            tool_name = _resource_dispatch.get(attachment.content_type or "")
            if not tool_name:
                continue
            try:
                file_bytes = asyncio.run_coroutine_threadsafe(
                    attachment.read(), discord_loop
                ).result()
                result = call_tool(tool_name, file_bytes=file_bytes)
                user_text += "\n" + result
            except Exception as e:  # noqa: BLE001
                user_text += f"\n\n[{tool_name} failed: {e}]"

        print(f"[discord] Received: {user_text[:120]}…")

        timestamp = (message.created_at + time_offset) \
            .replace(tzinfo=None).isoformat(timespec='minutes')

        # Send a "thinking" placeholder we can edit as tokens arrive
        try:
            placeholder = asyncio.run_coroutine_threadsafe(
                message.reply("⏳ *thinking…*"), discord_loop
            ).result()
        except Exception:
            placeholder = None

        text_so_far = ""
        last_edit_len = [0]    # mutable container; nonlocal-ish

        def _on_token(delta: str):
            nonlocal text_so_far
            text_so_far += delta
            # Throttle edits — only edit every ~40 chars to stay under
            # the 5 edits / 5 s limit while keeping the user informed
            if placeholder and len(text_so_far) - last_edit_len[0] >= 40:
                last_edit_len[0] = len(text_so_far)
                _schedule_edit(placeholder, f"⏳ {text_so_far}")

        try:
            response, _ = llm_response(
                user_text,
                message.author.display_name,
                timestamp,
                on_token=_on_token,
            )
            final = response or "(no response)"
            if placeholder:
                _schedule_edit(placeholder, final)
            else:
                _schedule_edit(message, final)  # not ideal but fallback
        except Exception as e:  # noqa: BLE001
            err = f"⚠️ Error: {e}"
            if placeholder:
                _schedule_edit(placeholder, err)
        finally:
            llm_response_queue.task_done()


threading.Thread(target=worker, daemon=True).start()


# Admin / utility commands
@bot.command()
async def leave(ctx):
    if ctx.author.name in _admins:
        await ctx.leave_server(ctx.server)
    else:
        await ctx.send("❌ You don't have permission to make the bot leave.")


@bot.command()
async def clear_history(ctx):
    if ctx.channel.id not in _channel_whitelist:
        return
    if ctx.author.name not in _admins or ctx.author == bot.user:
        await ctx.send("❌ You don't have permission to clear the chat history.")
        return
    """Clears upto 100 messages in the channel"""

    await ctx.channel.purge(limit=10000)
 
    await ctx.send("✅ Chat history cleared.")

@bot.command()
async def cleardm(ctx, amount: int = 100):
    """
    Deletes the last X messages sent by the bot in this DM. 
    Usage: !cleardm 10
    """

    # Ensure it's a DM
    if not isinstance(ctx.channel, discord.DMChannel):
        await ctx.send("❌ This command only works in DMs.")
        return

    deleted = 0

    async for msg in ctx.channel.history(limit=200):
        if msg.author == bot.user:
            try:
                await msg.delete()
                deleted += 1
                await asyncio.sleep(0.6)
            except Exception:
                pass

            if deleted >= amount:
                break

    await ctx.send(f"✅ Deleted {deleted} of my messages.")


# Discord event handlers
@bot.event
async def on_ready():
    global discord_loop
    discord_loop = asyncio.get_running_loop()
    print(f"✅ Logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    print(f"{message.channel.id}\t{message.author.name}\t{message.content}")

    if message.channel.id not in _channel_whitelist:
        return

    if message.content and message.content[0] == '!':
        await bot.process_commands(message)
        return

    # Atomic check+put under a lock — prevents the qsize race that
    # previously let two messages through into a Queue(3) buffer.
    with _queue_mutex:
        if llm_response_queue.qsize() >= llm_response_queue.maxsize:
            await message.add_reaction("❌")
            await message.reply(
                "My input buffer is full, please wait until I finish with my queued responses!"
            )
            return
        llm_response_queue.put_nowait(message)
    await message.add_reaction("⏳")


bot.run(_TOKEN)