from process.llm_funcs.llm_scr import Riko_Response
from discord.ext import commands
import discord
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import threading
from queue import Queue
import asyncio
from process.llm_funcs.pdf_processor import pdf_text_extract

# Config:
load_dotenv()
time_offset = datetime.now().astimezone().utcoffset()
TOKEN = os.getenv("Discord_bot_token").strip()
channel_whitelist = [
    int(ch.strip())
    for ch in os.getenv("Discord_Channel_whitelist", "").split(",")
    if ch.strip()
]

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------------------------------------
llm_response_queue = Queue() #The message queue
discord_loop = None

def worker():
    """
        Worker thread to process LLM responses sequentialy.
        To use, put messages in llm_response_queue.
    """
    while True:
        message = llm_response_queue.get()
        message.content = message.content.lstrip('&').strip()
        

        user_text = f"{message.author.display_name}: {message.content}"

        # handle attachments
        for attachment in message.attachments:
            if attachment.content_type == "application/pdf":
                try:
                    pdf_byte = asyncio.run_coroutine_threadsafe(
                        attachment.read(),
                        discord_loop
                    ).result()
                    pdf_text = pdf_text_extract(pdf_byte)
                    pdf_text = pdf_text[:100000] #limit max size
                    user_text += f"\n\n[PDF CONTENT START]\n{pdf_text}\n[PDF CONTENT END]\n"
                    
                except Exception as e:
                    user_text += f"\n\nFailed to process PDF: {e}"
            
        print(f"Received: {user_text}")
        try:
            timestamp = (message.created_at + time_offset ).replace(tzinfo=None).isoformat(timespec='minutes')
            
            response = Riko_Response(user_text, timestamp)
            
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
async def ping(ctx):
    """Responds with 'Pong!' when someone types !ping."""
    await ctx.send('Pong!')
    
@bot.command()
async def clear_history(ctx):
    if not ctx.channel.id in channel_whitelist:
        return
    if not ctx.author.name == "yuuta_togashi." or ctx.author == bot.user:
        await ctx.send("❌ You don't have permission to clear the chat history.")
        return
    """Clears upto 100 messages in the channel"""

    await ctx.channel.purge(limit=10000)
 
    await ctx.send("✅ Chat history cleared.")

@bot.command()
async def cleardm(ctx, amount: int):
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
    #process_message(message)
        # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    print(message.channel.id , message.author.name + "\t" + message.content +  "\t")
    # Only respond in the target channel
    if message.channel.id  in channel_whitelist:
        
        if message.content[0] == '!':
            await bot.process_commands(message)
            return
        
        else:
            llm_response_queue.put(message)
            await message.add_reaction("⏳")
            return
    # Keep commands working

bot.run(TOKEN)