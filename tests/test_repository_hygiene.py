"""Guard the Git index: ignore rules alone do not untrack committed artifacts."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT=Path(__file__).resolve().parents[1]
GENERATED=('electron/node_modules/', 'electron/dist/', 'electron/public/models/',
           'electron/out/', 'electron/release/', 'electron/.vite/', 'electron/.cache/',
           'electron/coverage/', 'electron/test-results/', 'electron/playwright-report/')


def git(*args):
    if not shutil.which('git'): pytest.skip('Git unavailable')
    result=subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,text=True)
    if result.returncode not in {0,1}: pytest.skip('Source checkout has no Git metadata')
    return result


def test_git_index_excludes_generated_electron_artifacts():
    paths=git('ls-files','-z','--','electron').stdout.split('\0')
    forbidden=[path for path in paths if path.lower().startswith(GENERATED)
               or (path.count('/') == 1 and path.lower().endswith(('.log','.tsbuildinfo')))]
    assert not forbidden, 'Generated Electron artifacts are staged/tracked: ' + ', '.join(forbidden[:10])


def test_ignore_covers_artifacts_but_keeps_source_assets_and_lockfile():
    for directory in GENERATED:
        assert git('check-ignore','--no-index','--',directory+'example.file').returncode == 0, directory
    for path in ('electron/runtime.log','electron/cache.tsbuildinfo'):
        assert git('check-ignore','--no-index','--',path).returncode == 0, path
    for path in ('electron/package-lock.json','electron/public/assets/example.svg','electron/src/ui/icons.jsx'):
        assert git('check-ignore','--no-index','--',path).returncode == 1, path


def test_no_tracked_file_is_ignored_and_the_data_roots_user_files_are():
    """An ignore rule that covers a tracked file hides its later changes; persistent_memories/ and the data root's other
    user files (credentials, MCP config, models, approvals, the to-do list) must never be staged by `git add .`."""
    assert git('ls-files', '-ci', '--exclude-standard').stdout == ''  # character_files/main_sample.wav stays tracked
    for path in ('persistent_memories/tool_approvals.json', 'persistent_memories/mcp_modules/todo_list/todo.json', 'persistent_memories/new_store.json',
            'mcp.json', '.env', '.env.local', '.env.discord', 'models/huggingface/hub/x.bin', '.resource-estimate-abc.yaml', 'wake_words/x.onnx',
            'stray.tmp', 'character_files/models/user.vrm', 'character_files/voice.wav', 'character_config.yaml.previous', 'logs/debug.log',
            '~/tasks.sqlite3',
            # A data root inside the checkout: copies of the YAML (it may hold API keys), user data and logs, at any depth.
            'data/character_config.yaml', 'data/character_config.yaml.previous', 'data/.character_config.yaml.previous.x.tmp', 'data/.settings-save-x',
            'data/.settings-validation-x.yaml', 'data/.resource-estimate-x.yaml', 'data/persistent_memories/tool_approvals.json',
            'data/persistent_memories/mcp_modules/todo_list/todo.json', 'data/mcp.json', 'data/logs/debug.log', 'data/logs/backend-launch.log.1'):
        assert git('check-ignore', '--no-index', '--', path).returncode == 0, path
    for path in ('persistent_memories/README.md', 'character_files/main_sample.wav', '.env.discord.example', 'mcp.json.example',
            'Code/run_server.py', 'tests/test_safe_paths.py', 'tools/release/build.py', 'electron/src/content_security.test.mjs'):
        assert git('check-ignore', '--no-index', '--', path).returncode == 1, path
