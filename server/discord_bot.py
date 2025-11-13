from process.llm_funcs.llm_scr import llm_response
from discord.ext import commands
import discord
import os
from dotenv import load_dotenv
from queue import Queue
from datetime import datetime, timedelta
# Config:

load_dotenv()

time_offset = timedelta(hours=5, minutes=30)

TOKEN = os.getenv("Discord_bot_token") 
CHANNEL_ID = int(os.getenv("Discord_bot_channel_id"))
print("Channel ID: ",CHANNEL_ID)
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def process_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    print(message.channel.id , message.author.name + "\t" + message.content +  "\t")
    # Only respond in the target channel
    if message.channel.id  == CHANNEL_ID:
        
        if message.content[0] != '&':
            return
        message.content = message.content[1:]
        user_text = f"({message.author.display_name}): {message.content}"
        print(f"Received: {user_text}")

        try:
            timestamp = (message.created_at + time_offset ).replace(tzinfo=None).isoformat(timespec='seconds')
            response = llm_response(user_text,timestamp)
        except Exception as e:
            response = f"⚠️ Error: {e}"

        print(response)
        message.reply(response)


@bot.command()
async def ping(ctx):
    """Responds with 'Pong!' when someone types !ping."""
    await ctx.send('Pong!')


@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    #process_message(message)
        # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    print(message.channel.id , message.author.name + "\t" + message.content +  "\t")
    # Only respond in the target channel
    if message.channel.id  == CHANNEL_ID:
        
        if message.content[0] != '&':
            return
        message.content = message.content[1:]
        user_text = f"({message.author.display_name}): {message.content}"
        print(f"Received: {user_text}")

        try:
            timestamp = (message.created_at + time_offset ).replace(tzinfo=None).isoformat(timespec='seconds')
            response = llm_response(user_text,timestamp)
        except Exception as e:
            response = f"⚠️ Error: {e}"

        print(response)
        await message.reply(response)
    # Keep commands working
    #await bot.process_commands(message)

bot.run(TOKEN)