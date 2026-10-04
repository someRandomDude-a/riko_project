"""CI-only build; never reads personal YAML, memories, model weights or .env."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'release-stage'
PIN = 'b92761a515ea31e852e7fbc1fad5f874b46f3718'


def run(*args, cwd=ROOT):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def cmake_file_definition(name, path):
    # CMake substitutes this value into generated source lists. Windows
    # backslashes (for example D:\a\...) can otherwise become invalid escapes.
    return f'-D{name}:FILEPATH={path.as_posix()}'


def apply_patch(checkout, patch):
    # .patch files may have CRLF in a Windows checkout. Keep the patched native
    # checkout and patch stream consistently LF, independent of global Git settings.
    content = patch.read_bytes().replace(b'\r\n', b'\n')
    for options in (['--check'], []):
        subprocess.run(['git', 'apply', *options, '-'], input=content, cwd=checkout, check=True)


def main():
    checkout = ROOT / '.native/llama.cpp-release'
    if checkout.exists(): raise RuntimeError('Use a clean release workspace; refusing to overwrite a checkout')
    checkout.parent.mkdir(exist_ok=True)
    run('git', '-c', 'core.autocrlf=false', 'clone', 'https://github.com/ggml-org/llama.cpp', checkout)
    run('git', '-c', 'core.autocrlf=false', 'checkout', '--detach', PIN, cwd=checkout)
    apply_patch(checkout, ROOT / 'tools/llama_cpp/emotion-probe.patch')
    for backend in ('cuda', 'vulkan'):
        build = checkout / ('build-' + backend)
        flags = ['-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_OPENSSL=OFF',
            '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_SERVER=ON',
            cmake_file_definition('RIKO_NATIVE_BRIDGE_SOURCE', ROOT / 'tools/llama_cpp/riko-native.cpp'),
            f'-DGGML_CUDA={"ON" if backend == "cuda" else "OFF"}',
            f'-DGGML_VULKAN={"ON" if backend == "vulkan" else "OFF"}']
        if backend == 'cuda':
            # Upstream uses -INFINITY as a max-reduction sentinel. MSVC's
            # expansion triggers NVCC #221 repeatedly; retain the upstream code.
            flags += ['-DCMAKE_CUDA_FLAGS=--diag-suppress=221']
        if sys.platform != 'win32': flags += ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', '-DCMAKE_INSTALL_RPATH=$ORIGIN']
        run('cmake', '-S', checkout, '-B', build, *flags)
        run('cmake', '--build', build, '--config', 'Release', '--target', 'riko-native', '-j', '2')
        destination = STAGE / 'native' / backend
        destination.mkdir(parents=True, exist_ok=True)
        for file in build.rglob('*'):
            if file.is_file() and (file.suffix == '.dll' or '.so' in file.name): shutil.copy2(file, destination / file.name)
        if backend == 'cuda':
            cuda = Path(os.environ.get('CUDA_PATH', '/usr/local/cuda'))
            patterns = ('**/cudart64*.dll', '**/cublas64*.dll', '**/cublasLt64*.dll') if sys.platform == 'win32' else ('**/libcudart.so*', '**/libcublas.so*', '**/libcublasLt.so*')
            for pattern in patterns:
                files = list(cuda.glob(pattern))
                if not files: raise RuntimeError(f'CUDA redistributable missing: {pattern}')
                for file in files:
                    if file.is_file(): shutil.copy2(file, destination / file.name)
        library = destination / ('riko-native.dll' if sys.platform == 'win32' else 'libriko-native.so')
        if not library.exists(): raise RuntimeError('Native library was not produced')
    notices = STAGE / 'notices'
    notices.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkout / 'LICENSE', notices / 'llama.cpp-LICENSE')
    shutil.copy2(ROOT / 'tools/release/README.md', notices / 'distribution-notes.md')
    # Freeze the backend and dynamic imports; no source config or private assets.
    run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
        '--name', 'riko-backend', '--paths', ROOT / 'Code', '--distpath', STAGE,
        '--workpath', ROOT / 'release-build', '--specpath', ROOT / 'release-build',
        '--collect-submodules', 'process', '--hidden-import', 'desktop_server', '--hidden-import', 'discord_bot',
        *[item for package in ('torch', 'transformers', 'sentence_transformers', 'faster_whisper',
            'ctranslate2', 'silero_vad', 'onnxruntime', 'faiss', 'sounddevice', 'soundfile',
            'scipy', 'cv2', 'ruamel.yaml', 'eff_word_net', 'discord', 'uvicorn')
            for item in ('--collect-all', package)], ROOT / 'Code/run_server.py')
    shutil.copytree(STAGE / 'riko-backend', STAGE / 'backend', dirs_exist_ok=True)
    executable = STAGE / 'backend' / ('riko-backend.exe' if sys.platform == 'win32' else 'riko-backend')
    for backend in ('cuda', 'vulkan'):
        library = STAGE / 'native' / backend / ('riko-native.dll' if sys.platform == 'win32' else 'libriko-native.so')
        run(executable, '--release-check', library)
    manifest = {str(file.relative_to(STAGE)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in STAGE.rglob('*') if file.is_file()}
    (notices / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
