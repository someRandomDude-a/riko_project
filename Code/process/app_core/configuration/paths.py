"""Where Riko keeps its data. DataPaths names every store under the data root, the directory holding character_config.yaml:
the checkout in development, the folder packaged Electron passes as RIKO_DATA_DIR (holding RIKO_CONFIG) in a release.
load_config builds it once (AppConfig.paths), honouring memory.history_file (or a top-level history_file), memory.store_file
and tasks.store_file, and the stores, the logs, the Discord worker and Code/task_mcp_server.py take their files from it.
tests/test_data_paths.py pins every entry to the location earlier releases used, in both layouts, so no data moves. Where
the code is lives in kernel/code_paths.py (CodePaths), which the tool and Discord launchers can import."""
from __future__ import annotations

from dataclasses import dataclass, fields
import os
from pathlib import Path
import sys

from ..kernel.code_paths import CodePaths

MEMORIES = 'persistent_memories'
TODO_LIST = (MEMORIES, 'mcp_modules', 'todo_list')


def resolve(root, value):
    """A configured path: ~ expanded, a relative one under root; None stays None."""
    if value is None: return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path


def working_directory(environ=None):
    """The backend's working directory, which a relative RIKO_CONFIG is resolved against: RIKO_DATA_DIR, else the checkout
    (frozen: the backend binary's folder)."""
    environ = os.environ if environ is None else environ
    if environ.get('RIKO_DATA_DIR'): return Path(environ['RIKO_DATA_DIR'])
    code = CodePaths.current(environ)
    return Path(sys.executable).resolve().parent if code.frozen else code.root


def config_file(environ=None):
    """The YAML the backend loads, for a process that starts beside it (the Discord worker, the task MCP server)."""
    environ = os.environ if environ is None else environ
    path = Path(environ.get('RIKO_CONFIG') or 'character_config.yaml').expanduser()
    return path if path.is_absolute() else working_directory(environ) / path


def _kept(path, legacy):
    """path, unless only legacy exists: a store an earlier release wrote elsewhere stays where its data is."""
    return legacy if legacy is not None and legacy != path and not path.exists() and legacy.exists() else path


@dataclass(frozen=True)
class DataPaths:
    root: Path  # the data root: the directory holding character_config.yaml
    config_file: Path
    memories: Path  # persistent_memories/: user data, never deleted by cleanup
    chat_history: Path  # memory.history_file (or a top-level history_file): the model's context
    memory_store: Path  # memory.store_file
    memory_index: Path  # memory.index_file: the FAISS index of earlier releases, neither read nor written now
    tasks: Path  # tasks.store_file, shared with Code/task_mcp_server.py
    conversations: Path  # the UI's archive
    whiteboard: Path
    desktop_settings: Path  # the avatar's geometry
    tool_approvals: Path
    initiative_settings: Path  # live initiative preferences: they override the YAML's initiative section
    discord_access: Path
    discord_preferences: Path
    wake_words: Path  # enrolled wake-word profiles (a folder)
    todo_list: Path  # the built-in todo_list tool's folder
    emotion_probes: Path  # probe datasets of earlier releases (a folder), only read
    api_token: Path  # development only: the backend writes both (desktop/api_guard.py), electron/main.cjs reads them
    confirm_key: Path
    env_file: Path  # .env: the Discord credentials, and secrets the debug log redacts
    logs: Path  # logs/debug.log; packaged Electron also writes logs/backend-launch.log
    models: Path  # emotion-probe weights and datasets; packaged Electron also puts the Hugging Face and torch caches here

    @classmethod
    def build(cls, config_file, raw=None, *, working=None):
        """Every location for the YAML at config_file (absolute), whose mapping is raw (None: the overridable stores at their
        defaults, enough for a process that reads only fixed entries, such as the Discord worker). working: the backend's
        working directory (RIKO_DATA_DIR), where earlier releases kept the to-do list (relative to the cwd)."""
        config_file = Path(config_file)
        root, raw = config_file.parent, raw if isinstance(raw, dict) else {}
        memory, tasks = (raw.get(name) if isinstance(raw.get(name), dict) else {} for name in ('memory', 'tasks'))
        memories, task_file = root / MEMORIES, tasks.get('store_file')
        # Earlier releases joined tasks.store_file to the root without expanding ~, so `~/tasks.sqlite3` made a folder named ~.
        literal = root / task_file if isinstance(task_file, str) and task_file.startswith('~') else None
        return cls(root=root, config_file=config_file, memories=memories,
            chat_history=resolve(root, raw.get('history_file', memory.get('history_file'))) or memories / 'chat_history.json',
            memory_store=resolve(root, memory.get('store_file')) or memories / 'memory_store.json',
            memory_index=resolve(root, memory.get('index_file')) or memories / 'faiss_index.index',
            tasks=_kept(resolve(root, task_file) or memories / 'tasks.sqlite3', literal),
            conversations=memories / 'conversations.sqlite3', whiteboard=memories / 'whiteboard.json',
            desktop_settings=memories / 'desktop_settings.json', tool_approvals=memories / 'tool_approvals.json',
            initiative_settings=memories / 'initiative_settings.json', discord_access=memories / 'discord_access.json',
            discord_preferences=memories / 'discord_preferences.json', wake_words=memories / 'wake_words',
            todo_list=_kept(root.joinpath(*TODO_LIST), Path(working).joinpath(*TODO_LIST) if working else None),
            emotion_probes=memories / 'emotion_probes', api_token=memories / 'api_token', confirm_key=memories / 'confirm_key',
            env_file=root / '.env', logs=root / 'logs', models=root / 'models')

    @classmethod
    def at(cls, root):
        """The default layout under root (tests, tools)."""
        return cls.build(Path(root) / 'character_config.yaml')

    def entries(self): return {item.name: getattr(self, item.name) for item in fields(self)}


def from_environment(environ=None):
    """DataPaths for the YAML the backend loads (config_file), with its overrides, for a process that runs no load_config, such
    as the task MCP server. A YAML that cannot be read fails rather than leaving a store at a default the backend may not use."""
    environ = os.environ if environ is None else environ
    path, raw = config_file(environ), {}
    if path.exists():
        import yaml  # PyYAML, as load_config reads it
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return DataPaths.build(path, raw, working=environ.get('RIKO_DATA_DIR'))
