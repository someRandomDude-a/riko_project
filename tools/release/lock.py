"""Regenerates the release CI's hash-locked Python requirements with uv (https://docs.astral.sh/uv/).

    python tools/release/lock.py                          # keep the pins, follow edits to the inputs
    python tools/release/lock.py --upgrade                # newest allowed versions of everything
    python tools/release/lock.py --upgrade-package numpy  # one package

One universal file serves CPython 3.11 on Windows, Linux and macOS. torch is the CPU build from
download.pytorch.org (macOS gets the regular wheel), so pip needs that index as an extra index (release.yml).
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = ['--python-version', '3.11', '--universal', '--fork-strategy', 'fewest', '--torch-backend', 'cpu',
    '--generate-hashes', '--custom-compile-command', 'python tools/release/lock.py']


def compile_lock(*args, **kwargs):
    subprocess.run(['uv', 'pip', 'compile', '--quiet', *args, *OPTIONS, *sys.argv[1:]], cwd=ROOT, check=True, **kwargs)


def main():
    compile_lock('requirements-runtime.txt', 'tools/release/requirements-build.in', '-o', 'tools/release/requirements-lock.txt')
    # Wake enrollment uses only EfficientWord-Net's ONNX module; its declared dependencies (PyAudio, twine, typer)
    # are not installed (see install_reqs.sh), so it has its own --no-deps lock.
    compile_lock('-', '--no-deps', '-o', 'tools/release/requirements-lock-no-deps.txt', input=b'EfficientWord-Net\n')


if __name__ == '__main__': main()
