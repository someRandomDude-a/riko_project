from discord.ext import commands
import discord
import os
from dotenv import load_dotenv
from datetime import datetime
import threading
from queue import Queue
import asyncio

from process.llm_scripts.MCP_Tools import call_tool
from process.llm_scripts.module import llm_response

# Config
time_offset = datetime.now().astimezone().utcoffset()
load_dotenv()
_TOKEN = os.getenv("Discord_bot_token", "").strip()
_channel_whitelist = [
    int(ch.strip())
    for ch in os.getenv("Discord_Channel_whitelist", "").split(",")
    if ch.strip()
]

_admins = [
    str(ch.strip())
    for ch in os.getenv("Discord_admins", "").split(",")
    if ch.strip()
]


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------------------------------------
llm_response_queue = Queue(3) # The message queue
discord_loop = None

def worker():
    """
        Worker thread to process LLM responses sequentialy.
        To use, put messages in llm_response_queue.
    """
    if discord_loop is None:
        raise ValueError("variable `discord_loop` was not assigned correctly")
    while True:
        message = llm_response_queue.get()        

        user_text = f"{message.author.display_name}: {message.content}"

        # handle attachments
        for attachment in message.attachments:
            if attachment.content_type == "application/pdf":
                try:
                    pdf_byte = asyncio.run_coroutine_threadsafe(
                        attachment.read(),
                        discord_loop
                    ).result()
                    pdf_text = call_tool("pdf_extractor", file_bytes=pdf_byte)
                    user_text += ("\n" + pdf_text)
                    
                except Exception as e:
                    user_text += f"\n\nFailed to process PDF: {e}"
            
        print(f"Received: {user_text}")
        
        response = None
        try:
            timestamp = (message.created_at + time_offset ).replace(tzinfo=None).isoformat(timespec='minutes')
            
            response, _ = llm_response(user_text, message.author.display_name, timestamp)
            
        except Exception as e:
            response = f"⚠️ Error: {e}"
        finally:
            asyncio.run_coroutine_threadsafe(
                message.reply(response),
                discord_loop
            )
            llm_response_queue.task_done()        

threading.Thread(target=worker, daemon=True).start()


@bot.command()
async def leave(ctx):
    """Leave the server if you are the owner and give the leave command"""
    if ctx.author.name in _admins:
        await ctx.leave_server(ctx.server)
    else:
        await ctx.send("❌ You don't have permission to make the bot leave.")
    
@bot.command()
async def clear_history(ctx):
    if not ctx.channel.id in _channel_whitelist:
        return
    if not ctx.author.name in _admins or ctx.author == bot.user:
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

    async for message in ctx.channel.history(limit=200):
        if message.author == bot.user:
            try:
                await message.delete()
                deleted += 1
                await asyncio.sleep(0.6)  # Prevent rate limits
            except:
                pass

            if deleted >= amount:
                break

    await ctx.send(f"✅ Deleted {deleted} of my messages.")


@bot.event
async def on_ready():
    global discord_loop
    discord_loop = asyncio.get_running_loop()
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    # ignore messages from the bot itself
    if message.author == bot.user:
        return
    print(f"{message.channel.id}\t{message.author.name}\t{message.content}") 

   # only respond in the target channel
    if message.channel.id  in _channel_whitelist:
        
        if message.content[0] == '!':
            await bot.process_commands(message)
            return
        
        else:
            if llm_response_queue.qsize() >= llm_response_queue.maxsize:
                await message.add_reaction("❌")
                await message.reply("My input buffer is full, please wait until I finish with my queued responses!")
            llm_response_queue.put(message)
            await message.add_reaction("⏳")
            return
    # keep commands working

bot.run(_TOKEN)