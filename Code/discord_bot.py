"""Start the optional Discord client of the already-running Python backend."""
import asyncio
import logging
import os
import sys
import threading


def exit_with_backend(bot):
    """Started by the desktop backend: stop when its pipe closes, which happens however the backend exits."""
    try: sys.stdin.read()
    except (OSError, ValueError): pass
    try: asyncio.run_coroutine_threadsafe(bot.close(), bot.loop).result(10)
    except Exception: pass  # not logged in yet, or already closing
    os._exit(0)


def main():
    from dotenv import load_dotenv
    from process.app_core.configuration.paths import DataPaths, config_file, working_directory
    from process.app_core.integrations.discord.access import DiscordAccess
    from process.app_core.integrations.discord.bot import CompanionBot
    # The backend's data root: RIKO_CONFIG against RIKO_DATA_DIR (default: the checkout), as run_server resolves them; the
    # launcher passes both. The worker reads only fixed entries (access, .env, preferences), never a store the YAML moves.
    legacy = DataPaths.at(working_directory())  # where a worker started by hand looked before DataPaths (RIKO_DATA_DIR or the checkout)
    os.environ.setdefault('RIKO_DATA_DIR', str(working_directory()))  # Where the backend client finds the API token.
    paths = DataPaths.build(config_file())
    # A worker started by hand with RIKO_CONFIG elsewhere keeps the credentials and channel preferences it used before when
    # the config's folder has none, as paths.py keeps other stores where their data is.
    # (The access lists stay the backend's: it writes them beside its config.)
    kept = {name: getattr(legacy, name) for name in ('env_file', 'discord_preferences')
            if not getattr(paths, name).exists() and getattr(legacy, name).exists()}
    if kept:
        from dataclasses import replace
        logging.getLogger(__name__).warning('Using the Discord files from %s; move them beside %s', legacy.root, paths.config_file)
        paths = replace(paths, **kept)
    load_dotenv(paths.env_file) # Never search parent/private directories for credentials.
    settings = DiscordAccess(paths).settings()
    if '--dry-run' in sys.argv:  # run_server.release_check: the worker starts and builds its client, but never logs in
        CompanionBot(settings, preferences=paths.discord_preferences)
        print('RIKO_DISCORD_DRY_RUN_OK', flush=True)
        return
    if not settings.token: raise ValueError('Discord_bot_token is required')
    if not settings.admins: raise ValueError('Configure Discord_admins before starting the bot')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    bot = CompanionBot(settings, preferences=paths.discord_preferences)
    if os.environ.get('RIKO_EXIT_WITH_BACKEND') == '1':
        threading.Thread(target=exit_with_backend, args=(bot,), name='backend-watch', daemon=True).start()
    bot.run(settings.token)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        # The launcher consumes only these fixed codes, never raw logs or tokens.
        name = type(exc).__name__
        code = 'login' if name == 'LoginFailure' else 'intents' if name == 'PrivilegedIntentsRequired' else 'dependency' if isinstance(exc, ImportError) else 'backend' if isinstance(exc, TimeoutError) else 'startup'
        print('RIKO_DISCORD_ERROR:' + code, flush=True)
        if '--dry-run' in sys.argv: raise  # the release check shows the traceback; the launcher never passes --dry-run
        raise SystemExit(1) from None
