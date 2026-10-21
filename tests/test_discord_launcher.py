from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from process.app_core.integrations.discord import launcher
from process.app_core.configuration.paths import DataPaths


def test_launcher_is_explicit_idempotent_and_uses_only_the_transport(tmp_path, monkeypatch):
    # The data root is outside the checkout and holds no Code/: the worker is code, found where this code is (V106).
    monkeypatch.setattr(launcher.os, 'environ', {'Discord_bot_token': 'fake-token', 'Discord_admins': '1', 'RIKO_PORT': '9123'})
    monkeypatch.setattr('dotenv.dotenv_values', lambda path: {})
    monkeypatch.setattr(launcher.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda: None))
    calls = []
    class Process:
        code = None
        def poll(self): return self.code
        def terminate(self): self.code = 0
        def wait(self, timeout=None): return self.code
    process = Process()
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda command, **kwargs: calls.append((command, kwargs)) or process)
    paths = DataPaths.at(tmp_path / 'data')
    service = launcher.DiscordLauncher(paths)
    service.token = lambda: 'x' * 43  # desktop_server hands the launcher this backend start's token
    assert not service.status()['running']
    assert calls == []
    assert service.start()['running']
    assert service.start()['running']
    assert len(calls) == 1
    script = Path(launcher.__file__).resolve().parents[4] / 'discord_bot.py'
    assert calls[0][0] == [sys.executable, str(script)] and script.is_file() and not script.is_relative_to(tmp_path)
    assert 'fake-token' not in str(calls[0][0])  # credentials never on the command line
    # The child inherits the environment as before, plus only the backend API token, data root, config, exit rule and
    # address: it loads this backend's .env and access file whatever RIKO_DATA_DIR and RIKO_CONFIG held here, and reaches
    # the port this backend bound (RIKO_PORT) whatever Discord_backend_url its .env names.
    env = calls[0][1]['env']
    assert set(env) - set(launcher.os.environ) == {'RIKO_API_TOKEN', 'RIKO_DATA_DIR', 'RIKO_CONFIG', 'RIKO_EXIT_WITH_BACKEND', 'Discord_backend_url'}
    assert env['Discord_backend_url'] == 'http://127.0.0.1:9123'
    assert calls[0][1]['stdin'] == launcher.subprocess.PIPE  # closes when this backend exits
    assert env['RIKO_API_TOKEN'] == 'x' * 43 and env['RIKO_DATA_DIR'] == str(paths.root) == str(calls[0][1]['cwd'])
    assert env['RIKO_CONFIG'] == str(tmp_path / 'data' / 'character_config.yaml')
    assert not calls[0][1].get('shell')
    service.stop()
    assert process.code == 0
    assert not service.status()['running']


def test_launcher_configuration_failures_never_spawn_or_expose_values(tmp_path, monkeypatch):
    monkeypatch.setattr('dotenv.dotenv_values', lambda path: {})
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Unexpected process launch'))
    monkeypatch.setattr(launcher.os, 'environ', {})
    service = launcher.DiscordLauncher(DataPaths.at(tmp_path))
    with pytest.raises(ValueError, match='Discord_bot_token'): service.start()
    monkeypatch.setattr(launcher.os, 'environ', {'Discord_bot_token': 'private-test-token', 'Discord_admins': 'invalid-private-id'})
    with pytest.raises(ValueError, match='configuration is invalid'): service.start()
    assert 'private' not in str(service.status())


def test_launcher_reports_exit_without_exposing_child_output(tmp_path):
    service = launcher.DiscordLauncher(DataPaths.at(tmp_path))
    process = SimpleNamespace(wait=lambda: 1)
    service.process = process
    service._watch(process)
    assert not service.status()['running']
    assert 'exited' in service.status()['error']


def test_worker_started_by_the_backend_exits_when_the_backend_goes_away():
    import subprocess, sys
    code = "import sys, types; sys.path.insert(0, 'Code'); import discord_bot; discord_bot.exit_with_backend(types.SimpleNamespace())"
    child = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE)
    child.stdin.close()  # what the OS does when the backend process ends
    assert child.wait(timeout=20) == 0
