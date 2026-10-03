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
