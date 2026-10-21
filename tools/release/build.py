"""CI-only build; never reads personal YAML, memories, model weights or .env."""
from pathlib import Path
import fnmatch
import os
import shutil
import subprocess
import sys
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[2]
STAGE, WORK = ROOT / 'release-stage', ROOT / 'release-build'  # what electron-builder packages; PyInstaller's work and dist
PIN = 'b92761a515ea31e852e7fbc1fad5f874b46f3718'


def load_native_backends():
    """Shipped backends, library names and CMake flags (electron/native_backends.json), read through the backend's own
    module, which imports nothing else from the app."""
    source = ROOT / 'Code/process/app_core/configuration/native_backends.py'
    spec = importlib.util.spec_from_file_location('native_backends', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load_native_backends()


def run(*args, cwd=ROOT, env=None):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True, env=env)


def release_check_environment(environ):
    """Users have no CUDA toolkit or Vulkan SDK, so the release check runs without the build's library paths, toolkit
    variables and PATH entries: a bundled library that is missing must fail to load, not resolve from the toolchain."""
    removed = {'LD_LIBRARY_PATH', 'LD_PRELOAD', 'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES',
        'PYTHONPATH', 'PYTHONHOME', 'VULKAN_SDK', 'VK_SDK_PATH'}
    environment = {key: value for key, value in environ.items()
        if key.upper() not in removed and not key.upper().startswith(('CUDA', 'RIKO_'))}
    for key in [key for key in environment if key.upper() == 'PATH']:
        environment[key] = os.pathsep.join(entry for entry in environment[key].split(os.pathsep)
            if entry and not any(name in entry.lower() for name in ('cuda', 'vulkan', 'nvidia')))
    return {**environment, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}


def cmake_file_definition(name, path):
    # CMake substitutes this value into generated source lists. Windows
    # backslashes (for example D:\a\...) can otherwise become invalid escapes.
    return f'-D{name}:FILEPATH={path.as_posix()}'


def native_flags(backend, platform=sys.platform):
    """CMake configure flags for one bundle: this backend's switch ON and the others OFF, then its own flags (the manifest's
    notes say why), and an rpath that finds the bundle's other libraries beside each one, then the backend's system folder."""
    flags = ['-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_OPENSSL=OFF',
        '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_SERVER=ON',
        cmake_file_definition('RIKO_NATIVE_BRIDGE_SOURCE', ROOT / 'tools/llama_cpp/riko-native.cpp'),
        *native.cmake_flags(backend)]
    if platform != 'win32': flags.append('-DCMAKE_BUILD_TYPE=Release')  # Visual Studio is multi-config: --config Release
    rpath = native.rpath(backend, platform)
    if rpath:
        # Without it CMake records the build directory, which the release check on CI resolves from and users lack. dyld
        # does not expand $ORIGIN, so macOS gets @loader_path, for build-tree and install-tree binaries alike.
        flags += ['-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', f'-DCMAKE_INSTALL_RPATH={rpath}']
        if platform == 'darwin': flags += ['-DCMAKE_MACOSX_RPATH=ON', f'-DCMAKE_BUILD_RPATH={rpath}']
    return flags


def shared_libraries(build, platform=sys.platform):
    """The build's shared libraries by the manifest's patterns (*.dll; *.so, *.so.*; *.dylib), symlinks included."""
    patterns = native.platform_build(platform)['shared_libraries']
    return sorted(file for file in build.rglob('*') if (file.is_symlink() or file.is_file())
        and any(fnmatch.fnmatch(file.name, pattern) for pattern in patterns))


def copy_shared_library(file, destination):
    """Copies one library into a bundle without following symlinks: a versioned chain (libggml.dylib -> libggml.0.dylib ->
    libggml.0.9.4.dylib, or libcublas.so -> .so.12 -> .so.12.4.5.8) keeps one copy of the code under every name the
    loader asks for. A link stays a link only to a file beside it; any other link is copied as the file it points to."""
    target = destination / file.name
    if target.is_symlink() or target.exists(): target.unlink()
    link = os.readlink(file) if file.is_symlink() else None
    if link and Path(link).name == link and (file.parent / link).exists(): os.symlink(link, target)
    else: shutil.copy2(file, target)
    return target


def dangling_links(destination):
    """Links in a bundle whose file was not copied beside them; the loader would fail on a user's machine."""
    return sorted(file.name for file in destination.iterdir() if file.is_symlink() and not file.exists())


def copy_cuda_redistributables(destination, platform=sys.platform, cuda=None):
    """cudart and cuBLAS from the toolkit, for the cuda bundle only; never the driver (nvcuda.dll, libcuda.so), nor the
    link-time stubs (lib64/stubs/libcublas.so), which share the real libraries' names."""
    cuda = Path(cuda or os.environ.get('CUDA_PATH', '/usr/local/cuda'))
    patterns = ('**/cudart64*.dll', '**/cublas64*.dll', '**/cublasLt64*.dll') if platform == 'win32' else ('**/libcudart.so*', '**/libcublas.so*', '**/libcublasLt.so*')
    for pattern in patterns:
        files = [file for file in cuda.glob(pattern) if (file.is_symlink() or file.is_file()) and 'stubs' not in file.relative_to(cuda).parts]
        if not files: raise RuntimeError(f'CUDA redistributable missing: {pattern}')
        for file in files: copy_shared_library(file, destination)


def mac_icon(svg, icns):
    """The app icon (electron-builder.yml mac.icon) from the SVG logo, with sips, which renders SVG, and iconutil, both part
    of macOS. The artwork fills 880 of 1024 pixels, so its rounded square sits on Apple's 824-pixel icon grid."""
    iconset = icns.with_suffix('.iconset')
    iconset.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels, png = size * scale, iconset / f'icon_{size}x{size}{"@2x" if scale == 2 else ""}.png'
            run('sips', '-s', 'format', 'png', '-z', round(pixels * 880 / 1024), round(pixels * 880 / 1024), svg, '--out', png)
            run('sips', '-p', pixels, pixels, png, '--out', png)  # transparent margin
    run('iconutil', '-c', 'icns', iconset, '-o', icns)
    return icns


def freeze_environment(environ):
    """PyInstaller's environment: Code/ on PYTHONPATH. The generated spec runs collect_submodules('process') before
    Analysis adds --paths, so without it 'process' is not importable there, the call quietly returns just 'process', and
    every module the backend imports only by name is left out: --tool-worker's process.app_core.tools.worker (runpy) and
    the built-in tools its worker imports."""
    return {**environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(ROOT / 'Code'), environ.get('PYTHONPATH')]))}


def manifest_data():
    """PyInstaller --add-data value: the frozen backend resolves bundled:<backend> from its own copy of the manifest."""
    return f'{native.manifest_path()}{os.pathsep}{native.FROZEN_DIRECTORY}'


def release_check(executable, bundles):
    """The frozen backend's --release-check against each staged native bundle (the OS's default ones, and an optional one
    the stage holds), one process each (the bundles name their libraries alike), without the build's toolchain paths."""
    environment = release_check_environment(os.environ)
    for backend in staged_backends(bundles):
        run(executable, '--release-check', bundles / backend / native.library_name(sys.platform), env=environment)


def check_app(app):
    """`build.py --check-app <Riko.app>`: the release check again inside the packaged app, where the backend runs signed,
    under the hardened runtime and with electron/build/entitlements.mac.plist, as users get it."""
    resources = Path(app).resolve() / 'Contents' / 'Resources'
    release_check(resources / 'backend' / 'riko-backend', resources / 'native')


def apply_patch(checkout, patch):
    # .patch files may have CRLF in a Windows checkout. Keep the patched native
    # checkout and patch stream consistently LF, independent of global Git settings.
    content = patch.read_bytes().replace(b'\r\n', b'\n')
    for options in (['--check'], []):
        subprocess.run(['git', 'apply', *options, '-'], input=content, cwd=checkout, check=True)


def build_jobs(environ=os.environ):
    """Parallel compile jobs: 2, or RIKO_BUILD_JOBS (the hip build compiles ggml-hip for twelve GPU targets on a 4-core runner)."""
    jobs = environ.get('RIKO_BUILD_JOBS') or '2'
    if not jobs.isdigit() or int(jobs) < 1: raise RuntimeError(f'RIKO_BUILD_JOBS must be a positive whole number, not {jobs!r}')
    return jobs


def build_bundle(checkout, backend, destination):
    """One native bundle from a patched checkout: riko-native and every library it loads, side by side in `destination`."""
    build = checkout / ('build-' + backend)
    run('cmake', '-S', checkout, '-B', build, *native_flags(backend))
    run('cmake', '--build', build, '--config', 'Release', '--target', 'riko-native', '-j', build_jobs())
    destination.mkdir(parents=True, exist_ok=True)
    for file in shared_libraries(build): copy_shared_library(file, destination)
    if backend == 'cuda': copy_cuda_redistributables(destination)
    return finish_bundle(destination, backend)


def finish_bundle(destination, backend, platform=sys.platform):
    """The bundle's library, once its folder holds every link's file, the library and (GGML_BACKEND_DL) its modules."""
    if dangling_links(destination): raise RuntimeError(f'Bundle links without their files: {dangling_links(destination)}')
    library = destination / native.library_name(platform)
    if not library.exists(): raise RuntimeError('Native library was not produced')
    if missing_modules(destination, backend, platform): raise RuntimeError(f'GGML_BACKEND_DL modules missing from the bundle: {missing_modules(destination, backend, platform)}')
    return library


def optional_bundles(folder, platform=sys.platform):
    """RIKO_OPTIONAL_BUNDLES: a folder of prebuilt optional bundles by backend id (release.yml unpacks its hip job's there),
    {backend: folder}; {} when unset. Only optional backends this OS ships: the default ones are built here."""
    if not folder: return {}
    folder = Path(folder)
    if not folder.is_dir(): raise RuntimeError(f'RIKO_OPTIONAL_BUNDLES is not a folder: {folder}')
    optional = sorted(set(native.backends_for(platform)) - set(native.release_backends(platform)))
    bundles = {path.name: path for path in sorted(folder.iterdir()) if path.is_dir()}
    unknown = sorted(set(bundles) - set(optional))
    if unknown or not bundles:
        raise RuntimeError(f'RIKO_OPTIONAL_BUNDLES must hold bundle folders named {" or ".join(optional) or "(no optional backend ships for " + platform + ")"}; found {", ".join(unknown) or "none"}')
    return bundles


def stage_bundle(source, backend, destination, platform=sys.platform):
    """A prebuilt optional bundle (release.yml builds hip in the ROCm container) into the stage, checked as build_bundle's."""
    destination.mkdir(parents=True)
    for file in shared_libraries(Path(source), platform): copy_shared_library(file, destination)
    return finish_bundle(destination, backend, platform)


def staged_backends(bundles, platform=sys.platform):
    """The bundles a release checks: every default backend of the OS, and an optional one only when the stage holds it."""
    released = native.release_backends(platform)
    return [backend for backend in native.backends_for(platform) if backend in released or (Path(bundles) / backend).is_dir()]


def missing_modules(destination, backend, platform=sys.platform):
    """With GGML_BACKEND_DL (the manifest's cmake_flags) the bundle needs its GPU module and at least one CPU module, which
    riko-native loads from its own folder; the release check then proves they load from there."""
    modules, names = native.backend_modules(backend, platform), [file.name for file in destination.iterdir()]
    if not modules: return []
    return [name for name in [modules['gpu']] if name and name not in names] + ([] if any(fnmatch.fnmatch(name, pattern)
        for name in names for pattern in modules['cpu']) else [' or '.join(modules['cpu'])])


def smoke_bundle(backend, checkout, destination):
    """`build.py --smoke-bundle <backend> <checkout> <destination>`: one bundle as a release builds it, with the same patch,
    flags and library copying, from an unpatched checkout of PIN: native-smoke.yml's Metal bundle, and the hip bundle that
    .github/actions/hip-bundle builds for rocm-compile.yml's compile job and for release.yml's opt-in hip job."""
    checkout, destination = Path(checkout).resolve(), Path(destination).resolve()
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=checkout, capture_output=True, text=True, check=True).stdout.strip()
    if head != PIN: raise RuntimeError(f'{checkout} is at {head}, not the pinned llama.cpp {PIN}')
    if backend not in native.backends_for(sys.platform): raise RuntimeError(f'{sys.platform} does not ship a {backend} bundle')
    apply_patch(checkout, ROOT / 'tools/llama_cpp/emotion-probe.patch')
    return build_bundle(checkout, backend, destination)


def check_bundle(library):
    """`build.py --check-bundle <native/<backend>/library>`: the release check's native half (run_server.native_release_problems)
    on a bundle, without the frozen app, where the backend's system libraries are installed (ROCm in the hip container), so
    they must load too: the exports (riko_devices and its device list included), a CPU module from the bundle, every module
    with its dependencies, no foreign copy."""
    library, shipped, name = Path(library).resolve(), native.backends_for(sys.platform), native.library_name(sys.platform)
    backend = library.parent.name
    if backend not in shipped or library.name != name: raise SystemExit(f'{library} is not a native/<{"|".join(shipped)}>/{name} bundle')
    spec = importlib.util.spec_from_file_location('run_server', ROOT / 'Code/run_server.py')  # stdlib-only at import
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    problems = server.native_release_problems(library, native.backend_modules(backend))
    if problems: raise SystemExit('The native bundle cannot ship:\n  ' + '\n  '.join(problems))
    print('RIKO_BUNDLE_CHECK_OK', flush=True)


def main():
    checkout = ROOT / '.native/llama.cpp-release'
    # Every output is new: copies over an earlier run's would ship its stale libraries and modules and hash them as this one's.
    stale = [path.relative_to(ROOT).as_posix() for path in (checkout, STAGE, WORK) if path.exists() or path.is_symlink()]
    if stale: raise RuntimeError(f'Use a clean release workspace; refusing to build over {", ".join(stale)}')
    optional = optional_bundles(os.environ.get('RIKO_OPTIONAL_BUNDLES'))  # before hours of building
    checkout.parent.mkdir(exist_ok=True)
    run('git', '-c', 'core.autocrlf=false', 'clone', 'https://github.com/ggml-org/llama.cpp', checkout)
    run('git', '-c', 'core.autocrlf=false', 'checkout', '--detach', PIN, cwd=checkout)
    apply_patch(checkout, ROOT / 'tools/llama_cpp/emotion-probe.patch')
    for backend in native.release_backends(sys.platform):  # cuda and vulkan on Windows and Linux, metal on macOS
        build_bundle(checkout, backend, STAGE / 'native' / backend)
    for backend, folder in optional.items(): stage_bundle(folder, backend, STAGE / 'native' / backend)  # hip, when asked
    notices = STAGE / 'notices'
    notices.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkout / 'LICENSE', notices / 'llama.cpp-LICENSE')
    shutil.copy2(ROOT / 'tools/release/NOTICE.md', notices / 'NOTICE.md')  # for users; tools/release/README.md is for maintainers
    if sys.platform == 'darwin': mac_icon(ROOT / 'assets/logo.svg', STAGE / 'mac' / 'icon.icns')
    # Freeze the backend and dynamic imports; no source config or private assets.
    # UTF-8 mode for the backend, --tool-worker and --discord-worker: frozen apps ignore PYTHONUTF8, and on Windows
    # stdio and open() would otherwise use the ANSI code page (logs/backend-launch.log, tool results).
    run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--python-option', 'X utf8',
        '--name', 'riko-backend', '--paths', ROOT / 'Code', '--distpath', WORK / 'dist',
        '--workpath', WORK, '--specpath', WORK,
        '--collect-submodules', 'process', '--hidden-import', 'desktop_server', '--hidden-import', 'discord_bot',
        '--add-data', manifest_data(),
        # numpy >= 2.3 imports this only from C; PyInstaller's own numpy hook lists it from 6.14.1 on.
        '--hidden-import', 'numpy._core._exceptions',
        *[item for package in ('torch', 'transformers', 'sentence_transformers', 'faster_whisper',
            'ctranslate2', 'silero_vad', 'onnxruntime', 'sounddevice', 'soundfile',
            'scipy', 'ruamel.yaml', 'eff_word_net', 'discord', 'uvicorn')
            for item in ('--collect-all', package)], ROOT / 'Code/run_server.py', env=freeze_environment(os.environ))
    # Moved, not copied: PyInstaller links libraries into _internal on Linux and macOS (libtorch_cpu.so -> torch/lib/...), and
    # a framework Python's Versions/Current must stay a link for codesign; a copy that followed them would duplicate or
    # break them. electron-builder's extraResources copy recreates the links (builder-util copyDir).
    os.replace(WORK / 'dist' / 'riko-backend', STAGE / 'backend')
    executable = STAGE / 'backend' / ('riko-backend.exe' if sys.platform == 'win32' else 'riko-backend')
    release_check(executable, STAGE / 'native')
    shipped = ('backend', 'native', 'notices')  # mac/ (the icon source) is not packaged as a file
    manifest = {str(file.relative_to(STAGE)): hashlib.sha256(file.read_bytes()).hexdigest()  # each file once, not its links
        for file in STAGE.rglob('*') if file.is_file() and not file.is_symlink() and file.relative_to(STAGE).parts[0] in shipped}
    if sys.platform == 'darwin':
        manifest = {'_note': 'Hashes of the files before packaging; signing the macOS app changes every binary.', **manifest}
    (notices / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--check-app']:
        if len(sys.argv) < 3: raise SystemExit('usage: build.py --check-app <path to Riko.app>')
        check_app(sys.argv[2])
    elif sys.argv[1:2] == ['--smoke-bundle']:
        if len(sys.argv) != 5: raise SystemExit('usage: build.py --smoke-bundle <backend> <llama.cpp checkout at PIN> <destination>')
        print(smoke_bundle(*sys.argv[2:5]), flush=True)
    elif sys.argv[1:2] == ['--check-bundle']:
        if len(sys.argv) != 3: raise SystemExit('usage: build.py --check-bundle <bundle folder>/<native library>')
        check_bundle(sys.argv[2])
    else: main()
