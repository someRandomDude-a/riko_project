import asyncio
import logging
import os
from datetime import datetime
from typing import Dict

import discord
from discord.ext import commands
from dotenv import load_dotenv

from process.llm_scripts.MCP_Tools import call_tool
from process.llm_scripts.core import llm_response

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
load_dotenv()

TOKEN = os.getenv("Discord_bot_token", "").strip()
if not TOKEN:
    logger.critical("Discord_bot_token not set in environment.")
    raise ValueError("Discord_bot_token is required")

WHITELIST_CHANNELS = [
    int(ch.strip()) for ch in os.getenv("Discord_Channel_whitelist", "").split(",")
    if ch.strip()
]
ADMIN_IDS = [
    int(id.strip()) for id in os.getenv("Discord_admins", "").split(",")
    if id.strip()
]
if not ADMIN_IDS:
    logger.warning("No admin IDs configured. Admin commands will be unavailable.")

# Discord Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Queue for incoming messages – maxsize prevents memory overload
message_queue: asyncio.Queue[discord.Message] = asyncio.Queue(maxsize=3)

# Map content types to tool names
RESOURCE_DISPATCH: Dict[str, str] = {
    "application/pdf": "pdf_extractor",
}

# Background queue processor
async def process_queue() -> None:
    """Background task that processes messages from the queue one by one."""
    while True:
        message = await message_queue.get()
        try:
            await handle_message(message)
        except Exception as e:
            logger.exception(f"Error processing message from {message.author}: {e}")
        finally:
            message_queue.task_done()

async def handle_message(message: discord.Message) -> None:
    # Basic context
    user_text = f"{message.author.display_name}: {message.content}"

    # Process attachments (read asynchronously, then call tool in executor)
    for attachment in message.attachments:
        tool_name = RESOURCE_DISPATCH.get(attachment.content_type or "")
        if not tool_name:
            continue

        # Read attachment bytes
        file_bytes = await attachment.read()

        # Call the potentially blocking tool in a thread
        try:
            result = await asyncio.to_thread(call_tool, tool_name, file_bytes=file_bytes)
            user_text += f"\n{result}"
        except Exception as e:
            user_text += f"\n\n[{tool_name} failed: {e}]"
            logger.error(f"Tool {tool_name} failed for {message.author}: {e}")

    logger.info(f"Processing message from {message.author}: {user_text[:100]}...")

    # Convert message timestamp to local timezone and format
    timestamp = (
        message.created_at.astimezone()
        .replace(tzinfo=None)
        .isoformat(timespec="minutes")
    )

    # Call the LLM in a thread
    try:
        response, _ = await asyncio.to_thread(
            llm_response,
            user_text,
            message.author.display_name,
            timestamp
        )
    except Exception as e:
        response = f"⚠️ Error: {e}"
        logger.error(f"LLM call failed for {message.author}: {e}")

    # Reply to the original message
    try:
        await message.reply(response)
    except discord.HTTPException as e:
        logger.error(f"Failed to reply to {message.author}: {e}")

# Commands
@bot.command(name="leave")
async def leave_guild(ctx: commands.Context) -> None:
    """Make the bot leave the current guild (server). Only admins can use this."""
    if ctx.author.id not in ADMIN_IDS:
        await ctx.send("❌ You don't have permission to make the bot leave.")
        return

    if not ctx.guild:
        await ctx.send("❌ This command must be used in a server.")
        return

    guild_name = ctx.guild.name
    try:
        await ctx.guild.leave()
        logger.info(f"Left guild '{guild_name}' on request of {ctx.author}")
    except Exception as e:
        logger.error(f"Failed to leave guild {guild_name}: {e}")
        await ctx.send(f"❌ Failed to leave: {e}")

@bot.command(name="clear_history")
async def clear_history(ctx: commands.Context) -> None:
    """
    Purge up to 10,000 messages in the current channel.
    Only works in whitelisted channels and for admins.
    """
    # Check permissions
    if ctx.author.id not in ADMIN_IDS:
        await ctx.send("❌ You don't have permission to clear chat history.")
        return
    if ctx.channel.id not in WHITELIST_CHANNELS:
        await ctx.send("❌ This channel is not whitelisted for history clearing.")
        return

    # Confirm action
    confirm_msg = await ctx.send(
        "⚠️ This will delete up to 10,000 messages. Continue? (reply with `yes`)"
    )
    try:
        reply = await bot.wait_for(
            "message",
            check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
            timeout=30.0
        )
        if reply.content.lower() != "yes":
            await ctx.send("❌ Cancelled.")
            return
    except asyncio.TimeoutError:
        await ctx.send("❌ Timed out. Cancelled.")
        return

    # Purge in chunks to avoid rate limits
    deleted = 0
    chunk_size = 100
    total_limit = 10000

    async for message in ctx.channel.history(limit=total_limit):
        try:
            await message.delete()
            deleted += 1
            if deleted % chunk_size == 0:
                await asyncio.sleep(1)  # pace deletions
        except discord.HTTPException as e:
            logger.warning(f"Could not delete message {message.id}: {e}")
            # Continue with next messages

    await ctx.send(f"✅ Deleted {deleted} messages.")

@bot.command(name="cleardm")
async def clear_dm(ctx: commands.Context, amount: int = 100) -> None:
    """
    Delete the last X messages sent by the bot in this DM.
    Only works in direct messages.
    """
    if not isinstance(ctx.channel, discord.DMChannel):
        await ctx.send("❌ This command only works in DMs.")
        return

    if amount <= 0:
        await ctx.send("❌ Amount must be positive.")
        return

    deleted = 0
    limit = min(amount, 200)  # Discord history limit per fetch
    async for message in ctx.channel.history(limit=limit):
        if message.author == bot.user:
            try:
                await message.delete()
                deleted += 1
                await asyncio.sleep(0.2)  # conservative rate limit
            except discord.HTTPException as e:
                logger.warning(f"Could not delete DM message {message.id}: {e}")
            if deleted >= amount:
                break

    await ctx.send(f"✅ Deleted {deleted} of my messages.")

# Event handlers
@bot.event
async def on_ready() -> None:
    """Called when the bot is connected and ready."""
    logger.info(f"✅ Logged in as {bot.user} (ID: {bot.user.id})") #type: ignore
    # Start the queue processor as a background task
    bot.loop.create_task(process_queue())
    logger.info("Queue processor started.")

@bot.event
async def on_message(message: discord.Message) -> None:
    """
    Main message handler:
      - Ignore bot's own messages.
      - Always process commands via bot.process_commands().
      - For whitelisted channels, queue non‑command messages for LLM processing.
    """
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    # If the message is a command don't queue.
    if message.content.startswith("!"):
        return

    # Only queue messages from whitelisted channels
    if WHITELIST_CHANNELS and message.channel.id not in WHITELIST_CHANNELS:
        return

    # Try to put the message into the queue
    try:
        message_queue.put_nowait(message)
        await message.add_reaction("⏳")
        logger.debug(f"Queued message from {message.author} in #{message.channel}")
    except asyncio.QueueFull:
        await message.add_reaction("❌")
        await message.reply(
            "My input buffer is full. Please wait until I finish my queued responses!"
        )
        logger.warning(f"Queue full - rejected message from {message.author}")

if __name__ == "__main__":
    try:
        bot.run(TOKEN)
    except discord.LoginFailure:
        logger.critical("Invalid bot token. Exiting.")
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    finally:
        # Cancel any remaining tasks
        asyncio.run(bot.close())