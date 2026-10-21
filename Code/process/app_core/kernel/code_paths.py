"""Where Riko's own code is: the checkout in development, the frozen backend in a release. Data never lives here
(configuration/paths.py DataPaths), so the Discord and tool workers are found from this file's location, never under the
data root: a data folder outside the checkout (RIKO_DATA_DIR, RIKO_CONFIG) still finds Code/discord_bot.py."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


@dataclass(frozen=True)
class CodePaths:
    root: Path  # the checkout (Code/ beside electron/), or the frozen backend's bundle (sys._MEIPASS)
    frozen: bool  # a PyInstaller build: the backend binary is its own workers (run_server --tool-worker, --discord-worker)
    bundle: Path | None = None  # RIKO_BUNDLE_ROOT: the installed app's resources (native/<backend>/), set only by packaged Electron

    @classmethod
    def current(cls, environ=None):
        environ = os.environ if environ is None else environ
        frozen = bool(getattr(sys, 'frozen', False))
        root = Path(getattr(sys, '_MEIPASS', None) or Path(sys.executable).parent) if frozen else Path(__file__).resolve().parents[4]
        return cls(root, frozen, Path(environ['RIKO_BUNDLE_ROOT']) if environ.get('RIKO_BUNDLE_ROOT') else None)

    @property
    def discord_bot(self): return self.root / 'Code' / 'discord_bot.py'

    @property
    def tool_worker(self): return self.root / 'Code' / 'process' / 'app_core' / 'tools' / 'worker.py'

    def discord_command(self): return [sys.executable, '--discord-worker'] if self.frozen else [sys.executable, str(self.discord_bot)]

    def tool_command(self): return [sys.executable, '--tool-worker'] if self.frozen else [sys.executable, str(self.tool_worker)]

    def server_command(self, *arguments):
        """run_server with arguments: the frozen binary itself, or Code/run_server.py (--setup-config, --validate-config)."""
        return [sys.executable, *arguments] if self.frozen else [sys.executable, str(self.root / 'Code' / 'run_server.py'), *arguments]
