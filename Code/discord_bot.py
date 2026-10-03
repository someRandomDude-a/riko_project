"""Start the optional Discord client of the already-running Python backend."""
import logging
import os
from pathlib import Path


def main():
    from dotenv import load_dotenv
    from process.app_core.integrations.discord.config import BotSettings
    from process.app_core.integrations.discord.bot import CompanionBot
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env') # Never search parent/private directories for credentials.
    settings = BotSettings.from_env(root, os.environ)
    if not settings.token: raise ValueError('Discord_bot_token is required')
    if not settings.admins: raise ValueError('Configure Discord_admins before starting the bot')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    CompanionBot(settings).run(settings.token)


if __name__ == '__main__': main()
