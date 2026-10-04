"""Backend entry point: standalone in development, Electron-owned in releases."""
from pathlib import Path
import os
import logging

import uvicorn

from process.app_core.configuration.config import load_config


def main():
    import sys
    if '--release-check' in sys.argv:
        import ctypes
        library = Path(sys.argv[sys.argv.index('--release-check') + 1]).resolve()
        directory = os.add_dll_directory(str(library.parent)) if os.name == 'nt' else None
        try:
            dll = ctypes.CDLL(str(library))
            for symbol in ('riko_create', 'riko_request', 'riko_set_interval', 'riko_stop', 'riko_destroy'): getattr(dll, symbol)
            import torch, faster_whisper, sounddevice, silero_vad, onnxruntime
        finally:
            if directory: directory.close()
        return
    if '--tool-worker' in sys.argv:
        import runpy
        runpy.run_module('process.app_core.tools.worker', run_name='__main__')
        return
    if '--discord-worker' in sys.argv:
        import runpy
        runpy.run_module('discord_bot', run_name='__main__')
        return
    os.chdir(Path(os.environ.get('RIKO_DATA_DIR', Path(__file__).resolve().parents[1])))
    config = load_config()
    from process.app_core.configuration.debug_logging import configure_logging
    configure_logging(config)
    # Debug our application, not WebSocket frames. Uvicorn DEBUG dumps every
    # voice.level/chat.delta packet and can dominate the capture/UI event loop.
    print("Riko AI server: http://127.0.0.1:8765 — Ctrl+C to stop", flush=True)
    import desktop_server
    server = uvicorn.Server(uvicorn.Config(desktop_server.app, host="127.0.0.1", port=8765,
                log_level="info", timeout_graceful_shutdown=10))
    if os.environ.get('RIKO_MANAGED') == '1':
        import threading
        def managed_shutdown():
            for line in sys.stdin:
                if line.strip() == 'shutdown': break
            server.should_exit = True
        threading.Thread(target=managed_shutdown, name='release-shutdown', daemon=True).start()
    server.run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
