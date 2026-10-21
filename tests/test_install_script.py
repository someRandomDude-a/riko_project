import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'install_reqs.sh'
pytestmark = pytest.mark.skipif(os.name == 'nt' or not shutil.which('bash'), reason='POSIX install script')


def install(tmp_path, system, tools=None, fail=None, **env):
    """Run install_reqs.sh against stub uname/uv/python (and optional GPU tools); return its exit code and uv calls."""
    stubs, log = tmp_path / 'bin', tmp_path / 'uv.log'
    stubs.mkdir(parents=True, exist_ok=True)
    for name, body in {'uname': f'echo {system}', 'python': 'exit 0', 'nvidia-smi': 'exit 9',  # a failing nvidia-smi is no GPU
            'uv': f'echo "$*" >> "{log}"' + (f'; case "$*" in *{fail}*) exit 1;; esac' if fail else ''), **(tools or {})}.items():
        (stubs / name).write_text('#!/bin/sh\n' + body + '\n'); (stubs / name).chmod(0o755)
    result = subprocess.run(['bash', str(SCRIPT)], capture_output=True, text=True, timeout=30,
        env={'PATH': f'{stubs}:/usr/bin:/bin', 'HOME': str(tmp_path), 'PYTHON': str(stubs / 'python'), **env})
    calls = log.read_text().splitlines() if log.exists() else []
    return result.returncode, [call.split(' --python ')[-1].split(' ', 1)[-1] for call in calls]


@pytest.mark.parametrize('system,tools,torch', [
    ('Darwin', {}, 'torch'),
    ('Linux', {'nvidia-smi': 'case "$*" in *driver_version*) echo 580.65.06;; esac'}, 'torch --index-url https://download.pytorch.org/whl/cu130'),
    ('Linux', {'nvidia-smi': 'case "$*" in *driver_version*) echo 550.54.15;; esac'}, 'torch --index-url https://download.pytorch.org/whl/cu126'),
    ('Linux', {'nvidia-smi': 'case "$*" in *driver_version*) echo 580.65.06;; *compute_cap*) echo 6.1;; esac'}, 'torch --index-url https://download.pytorch.org/whl/cu126'),  # Pascal: no CUDA 13 kernels
    ('Linux', {'nvidia-smi': 'case "$*" in *driver_version*) echo 580.65.06;; *compute_cap*) echo 7.5;; esac'}, 'torch --index-url https://download.pytorch.org/whl/cu130'),
    ('Linux', {'rocminfo': 'exit 0'}, 'torch --index-url https://download.pytorch.org/whl/rocm7.2'),
    ('Linux', {}, 'torch --index-url https://download.pytorch.org/whl/cpu'),
    ('MINGW64_NT-10.0', {'rocminfo': 'exit 0'}, 'torch --index-url https://download.pytorch.org/whl/cpu'),
])
def test_install_script_picks_torch_wheels_for_the_platform(tmp_path, system, tools, torch):
    if system == 'Linux' and not tools and os.path.exists('/opt/rocm/bin/rocminfo'): pytest.skip('host has ROCm')
    code, calls = install(tmp_path, system, tools)
    assert code == 0 and calls == [torch, '-r requirements-runtime.txt', '--no-deps EfficientWord-Net']


def test_install_script_override_and_failures_stop_it(tmp_path):
    assert install(tmp_path / 'a', 'Darwin', RIKO_TORCH_INDEX='cu126') == (0, ['torch --index-url https://download.pytorch.org/whl/cu126', '-r requirements-runtime.txt', '--no-deps EfficientWord-Net'])
    assert install(tmp_path / 'b', 'Darwin', RIKO_TORCH_INDEX='https://example.invalid') == (1, [])
    # A failed step ends the script instead of continuing into a half-installed environment.
    assert install(tmp_path / 'c', 'Darwin', fail='torch') == (1, ['torch'])
    assert install(tmp_path / 'd', 'Darwin', tools={'python': 'exit 1'})[0] == 1
    # Blackwell on a pre-580 driver: CUDA 12.6 has no kernels for it, so stop instead of installing a wheel that cannot run.
    assert install(tmp_path / 'e', 'Linux', {'nvidia-smi': 'case "$*" in *driver_version*) echo 575.51;; *compute_cap*) echo 12.0;; esac'}) == (1, [])


def test_install_script_is_strict_and_installs_nothing_unused():
    text = SCRIPT.read_text(encoding='utf-8')
    assert text.startswith('#!/usr/bin/env bash\n') and 'set -euo pipefail' in text and os.access(SCRIPT, os.X_OK)
    assert 'torchaudio' not in text and 'nltk' not in text
