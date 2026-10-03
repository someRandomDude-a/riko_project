"""Standalone backend entry point. Electron never owns this process."""
from pathlib import Path
import os
import logging

import uvicorn

from process.app_core.configuration.config import load_config


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    config = load_config()
    debug = config.raw.get("desktop", {}).get("debug", False) is True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Debug our application, not WebSocket frames. Uvicorn DEBUG dumps every
    # voice.level/chat.delta packet and can dominate the capture/UI event loop.
    for name in ('process.app_core', 'desktop_server'):
        logging.getLogger(name).setLevel(logging.DEBUG if debug else logging.INFO)
    print("Riko AI server: http://127.0.0.1:8765 — Ctrl+C to stop", flush=True)
    uvicorn.run("desktop_server:app", host="127.0.0.1", port=8765,
                log_level="info", timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
