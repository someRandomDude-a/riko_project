"""DataPaths (configuration/paths.py) and CodePaths (kernel/code_paths.py): every store resolves exactly where earlier releases
kept it, in the development layout (the checkout is the data root) and the packaged one (Electron's RIKO_DATA_DIR, RIKO_CONFIG
and RIKO_BUNDLE_ROOT), so no data moves; and the processes that start beside the backend (the Discord worker, the task MCP
server) find the backend's files, and the workers are found where the code is, never under the data root (V106)."""
from pathlib import Path
import re
import sys

import pytest

from process.app_core.configuration.config import AppConfig, MemoryConfig, load_config
from process.app_core.configuration.native_backends import backends_for, library_name
from process.app_core.configuration.paths import DataPaths, config_file, from_environment, working_directory
from process.app_core.desktop.api_guard import client_root, secret_path
from process.app_core.kernel.code_paths import CodePaths
from process.app_core.tools import isolation

REPO = Path(__file__).resolve().parents[1]


def earlier(root, working):
    """Where each store was before DataPaths, joined as each module joined it then."""
    memories = root / 'persistent_memories'
    return {'root': root, 'config_file': root / 'character_config.yaml', 'memories': memories,
        'chat_history': memories / 'chat_history.json', 'memory_store': memories / 'memory_store.json',  # config.py _path(root, ...)
        'memory_index': memories / 'faiss_index.index',
        'tasks': root / 'persistent_memories/tasks.sqlite3',  # factory.py: config.root / tasks.store_file
        'conversations': memories / 'conversations.sqlite3', 'whiteboard': memories / 'whiteboard.json',  # desktop_server.py
        'desktop_settings': memories / 'desktop_settings.json', 'tool_approvals': memories / 'tool_approvals.json',  # registry.py
        'initiative_settings': memories / 'initiative_settings.json',  # config.py, initiative.py and settings_store.py
        'discord_access': memories / 'discord_access.json',  # discord/access.py
        'discord_preferences': root / 'persistent_memories/discord_preferences.json',  # discord/bot.py: settings.root / ...
        'wake_words': memories / 'wake_words',  # audio/wake_word.py
        'todo_list': working / 'persistent_memories' / 'mcp_modules' / 'todo_list',  # todo_list.py: ./persistent_memories/..., the backend's cwd
        'emotion_probes': memories / 'emotion_probes',  # factory.py legacy_directory, probe_storage.corpus_files
        'api_token': secret_path(root, 'api_token'), 'confirm_key': secret_path(root, 'confirm_key'),  # desktop/api_guard.py
        'env_file': root / '.env',  # discord_bot.py, discord/access.py, debug_logging.py
        'logs': root / 'logs', 'models': root / 'models'}  # debug_logging.py; emotion/probe_storage.py


def test_every_store_stays_where_it_was_in_the_development_layout(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / 'character_config.yaml').write_text('runtime:\n  provider: openai\nmemory:\n  token_budget: 900\n', encoding='utf-8')
    # run_server: no RIKO_* set, so the cwd and RIKO_DATA_DIR become the checkout and the YAML is ./character_config.yaml.
    assert working_directory({}) == CodePaths.current({}).root == REPO
    monkeypatch.setenv('RIKO_DATA_DIR', str(checkout))
    monkeypatch.delenv('RIKO_CONFIG', raising=False)
    monkeypatch.chdir(checkout)
    paths = load_config().paths
    assert paths.entries() == earlier(checkout, checkout)
    assert paths == DataPaths.at(checkout) == from_environment({'RIKO_DATA_DIR': str(checkout)})


def test_every_store_stays_where_it_was_in_the_packaged_layout(tmp_path, monkeypatch):
    data, resources = tmp_path / 'Riko data', tmp_path / 'resources'
    data.mkdir()
    backend = backends_for(sys.platform)[0]
    (data / 'character_config.yaml').write_text(f'runtime:\n  native_library: "bundled:{backend}"\n', encoding='utf-8')
    # electron/release.cjs startBackend: these three variables, and the data folder as the cwd.
    environ = {'RIKO_DATA_DIR': str(data), 'RIKO_CONFIG': str(data / 'character_config.yaml'), 'RIKO_BUNDLE_ROOT': str(resources)}
    for name, value in environ.items(): monkeypatch.setenv(name, value)
    monkeypatch.chdir(data)
    config = load_config()
    assert config.paths.entries() == earlier(data, data)
    assert config.paths == from_environment(environ) and config_file(environ) == data / 'character_config.yaml'
    assert config.runtime.native_library == resources / 'native' / backend / library_name(sys.platform)
    assert CodePaths.current(environ).bundle == resources and CodePaths.current({}).bundle is None


def test_electron_creates_and_reads_the_same_folders():
    release, main = ((REPO / 'electron' / name).read_text(encoding='utf-8') for name in ('release.cjs', 'main.cjs'))
    paths = DataPaths.at(Path('data'))
    folders = re.search(r"for\(const folder of \[([^\]]*)\]\)fs\.mkdirSync", release).group(1)
    assert {name.strip("'") for name in folders.split(',')} == {paths.models.name, paths.memories.name, paths.logs.name}
    for variable in ('HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME'):  # the packaged model caches, under DataPaths.models
        assert re.search(variable + r":path\.join\(directory,'models',", release)
    assert "path.join(directory,'logs','backend-launch.log')" in release
    # Development: main reads the token and confirmation key from persistent_memories beside the config.
    assert "backend.watchSecrets(path.join(path.dirname(configPath),'persistent_memories'));" in main
    assert paths.api_token.parent == paths.confirm_key.parent == paths.root / 'persistent_memories'


def test_per_file_overrides_resolve_as_before_and_tasks_now_expand_home(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    for name in ('HOME', 'USERPROFILE'): monkeypatch.setenv(name, str(home))
    root, elsewhere = tmp_path / 'data', tmp_path / 'elsewhere'
    raw = {'memory': {'history_file': 'chats/history.json', 'store_file': str(elsewhere / 'memories.json'), 'index_file': '~/index'},
        'tasks': {'store_file': 'shared/tasks.sqlite3'}}
    paths = DataPaths.build(root / 'character_config.yaml', raw)
    assert (paths.chat_history, paths.memory_store, paths.memory_index, paths.tasks) == (
        root / 'chats/history.json', elsewhere / 'memories.json', home / 'index', root / 'shared/tasks.sqlite3')
    assert DataPaths.build(root / 'character_config.yaml', {'tasks': {'store_file': str(elsewhere / 't.db')}}).tasks == elsewhere / 't.db'
    # A top-level history_file (older YAML) still wins over memory.history_file.
    assert DataPaths.build(root / 'character_config.yaml', {**raw, 'history_file': '~/h.json'}).chat_history == home / 'h.json'
    # tasks.store_file was joined without expanding ~: a store saved in that literal folder stays there; otherwise ~ is home.
    tilde = {'tasks': {'store_file': '~/tasks.sqlite3'}}
    assert DataPaths.build(root / 'character_config.yaml', tilde).tasks == home / 'tasks.sqlite3'
    (root / '~').mkdir(parents=True)
    (root / '~' / 'tasks.sqlite3').write_bytes(b'')
    assert DataPaths.build(root / 'character_config.yaml', tilde).tasks == root / '~' / 'tasks.sqlite3'
    (root / 'character_config.yaml').write_text('tasks:\n  store_file: ~/tasks.sqlite3\n', encoding='utf-8')
    assert load_config(root / 'character_config.yaml').paths.tasks == root / '~' / 'tasks.sqlite3'
    (home / 'tasks.sqlite3').parent.mkdir(parents=True)
    (home / 'tasks.sqlite3').write_bytes(b'')
    assert DataPaths.build(root / 'character_config.yaml', tilde).tasks == home / 'tasks.sqlite3'  # once both exist, the configured one


def test_a_to_do_list_in_the_old_working_directory_stays_there(tmp_path):
    root, working = tmp_path / 'data', tmp_path / 'cwd'  # RIKO_CONFIG outside RIKO_DATA_DIR: the list followed the cwd
    config = root / 'character_config.yaml'
    assert DataPaths.build(config, working=working).todo_list == DataPaths.build(config).todo_list == root / 'persistent_memories/mcp_modules/todo_list'
    (working / 'persistent_memories/mcp_modules/todo_list').mkdir(parents=True)
    assert DataPaths.build(config, working=working).todo_list == working / 'persistent_memories/mcp_modules/todo_list'
    (root / 'persistent_memories/mcp_modules/todo_list').mkdir(parents=True)
    assert DataPaths.build(config, working=working).todo_list == root / 'persistent_memories/mcp_modules/todo_list'


def test_a_directly_built_config_keeps_every_store_under_its_root_never_the_cwd(tmp_path):
    config = AppConfig(root=tmp_path, memory=MemoryConfig(store_file=Path('custom/memories.json')))
    assert config.paths == DataPaths.build(tmp_path / 'character_config.yaml', {'memory': {'store_file': 'custom/memories.json'}})
    assert (config.memory.history_file, config.memory.store_file) == (config.paths.chat_history, tmp_path / 'custom/memories.json')
    assert config.paths.chat_history == tmp_path / 'persistent_memories' / 'chat_history.json'


def test_a_broken_yaml_fails_the_task_server_rather_than_guessing_a_store(tmp_path):
    (tmp_path / 'character_config.yaml').write_text('tasks: [unclosed\n', encoding='utf-8')
    with pytest.raises(Exception): from_environment({'RIKO_DATA_DIR': str(tmp_path)})


def test_clients_resolve_the_backends_config_as_run_server_does(tmp_path):
    data, other = tmp_path / 'data', tmp_path / 'other'
    assert config_file({}) == REPO / 'character_config.yaml'
    assert config_file({'RIKO_DATA_DIR': str(data)}) == data / 'character_config.yaml'
    assert config_file({'RIKO_CONFIG': 'profiles/riko.yaml'}) == REPO / 'profiles' / 'riko.yaml'
    assert config_file({'RIKO_DATA_DIR': str(data), 'RIKO_CONFIG': 'riko.yaml'}) == data / 'riko.yaml'
    assert config_file({'RIKO_DATA_DIR': str(data), 'RIKO_CONFIG': str(other / 'riko.yaml')}) == other / 'riko.yaml'


def test_client_token_reads_where_the_backend_writes(tmp_path, monkeypatch):
    data, other = tmp_path / 'data', tmp_path / 'other'
    for environ in ({'RIKO_DATA_DIR': str(data)}, {'RIKO_DATA_DIR': str(data), 'RIKO_CONFIG': str(other / 'riko.yaml')}):
        monkeypatch.delenv('RIKO_CONFIG', raising=False)
        for name, value in environ.items(): monkeypatch.setenv(name, value)
        backend = DataPaths.build(config_file(environ))
        assert secret_path(client_root(), 'api_token') == backend.api_token


def test_the_discord_worker_uses_the_backends_data_root_when_the_variables_disagree(tmp_path, monkeypatch, capsys):
    pytest.importorskip('discord')
    import discord_bot
    from process.app_core.integrations.discord import bot
    data, other = tmp_path / 'data', tmp_path / 'other'  # V106: RIKO_DATA_DIR and RIKO_CONFIG name different folders
    monkeypatch.setenv('RIKO_DATA_DIR', str(other))
    monkeypatch.setenv('RIKO_CONFIG', str(data / 'character_config.yaml'))
    monkeypatch.setattr(sys, 'argv', ['discord_bot.py', '--dry-run'])
    seen = {}
    monkeypatch.setattr('dotenv.load_dotenv', lambda path: seen.setdefault('dotenv', Path(path)))
    monkeypatch.setattr('dotenv.dotenv_values', lambda path: seen.setdefault('values', Path(path)) and {'Discord_admins': '1'})
    monkeypatch.setattr(bot, 'CompanionBot', lambda settings, *, preferences: seen.update(root=settings.root, preferences=preferences))
    discord_bot.main()
    paths = DataPaths.at(data)
    assert seen == {'dotenv': paths.env_file, 'values': paths.env_file, 'root': data, 'preferences': paths.discord_preferences}
    assert 'RIKO_DISCORD_DRY_RUN_OK' in capsys.readouterr().out
    legacy = DataPaths.at(other)  # a worker started by hand before DataPaths read these from RIKO_DATA_DIR: they stay in use
    legacy.memories.mkdir(parents=True); legacy.env_file.write_text('Discord_bot_token=x\n'); legacy.discord_preferences.write_text('{}')
    seen.clear()
    discord_bot.main()
    assert seen == {'dotenv': legacy.env_file, 'values': legacy.env_file, 'root': data, 'preferences': legacy.discord_preferences}


def test_the_task_mcp_server_defaults_to_the_backends_store(tmp_path, monkeypatch):
    from process.app_core.tools.registry import StdioMCPClient
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'character_config.yaml').write_text('tasks:\n  store_file: shared/tasks.sqlite3\n', encoding='utf-8')
    monkeypatch.delenv('RIKO_CONFIG', raising=False)
    monkeypatch.chdir(tmp_path)  # not the checkout: the default never followed the cwd
    client = StdioMCPClient(sys.executable, [str(REPO / 'Code' / 'task_mcp_server.py')], env={'RIKO_DATA_DIR': str(data)})
    try:
        task = client.call('task_create', {'title': 'Shared with the app'})['structuredContent']
    finally:
        client.close()
        client.process.wait(timeout=5)
    store = load_config(data / 'character_config.yaml').paths.tasks
    assert store == data / 'shared' / 'tasks.sqlite3'
    from process.app_core.persistence.tasks import TaskStore
    assert TaskStore(store).get(task['id'])['title'] == 'Shared with the app'
    assert not (REPO / 'shared').exists()


def test_workers_are_found_where_the_code_is(monkeypatch):
    code = CodePaths.current({})
    assert not code.frozen and code.root == REPO and code.discord_bot == REPO / 'Code' / 'discord_bot.py' and code.discord_bot.is_file()
    assert code.tool_worker == Path(isolation.__file__).resolve().with_name('worker.py') and code.tool_worker.is_file()
    assert isolation.worker_command() == code.tool_command() == [sys.executable, str(code.tool_worker)]
    assert code.discord_command() == [sys.executable, str(code.discord_bot)]
    monkeypatch.setattr(sys, 'frozen', True, raising=False)  # PyInstaller: the backend binary is its own workers
    monkeypatch.setattr(sys, '_MEIPASS', str(REPO / 'bundle'), raising=False)
    frozen = CodePaths.current({})
    assert frozen.frozen and frozen.root == REPO / 'bundle'
    assert isolation.worker_command() == [sys.executable, '--tool-worker'] and frozen.discord_command() == [sys.executable, '--discord-worker']
