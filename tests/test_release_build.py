import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys

import pytest
import yaml

import run_server
from process.app_core.configuration.native_backends import backends_for, library_name, release_backends


def release_build():
    source = Path(__file__).resolve().parents[1] / 'tools/release/build.py'
    spec = importlib.util.spec_from_file_location('release_build', source)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


CPU_DEVICE = {'index': 0, 'name': 'CPU', 'description': 'Test CPU', 'backend': 'CPU', 'type': 'cpu', 'device_id': None, 'memory_free': 1 << 30, 'memory_total': 1 << 30}


def riko_devices(text=None, result=None):
    """A fake riko_devices: writes `text` (ggml's device list as the bridge prints it) into the caller's buffer."""
    data = (text if text is not None else json.dumps({'version': 1, 'devices': [CPU_DEVICE]})).encode()
    def devices(buffer, capacity):
        if len(data) < capacity: buffer.value = data
        return len(data) if result is None else result
    return devices


def test_cmake_source_path_uses_forward_slashes_on_windows():
    source = Path(__file__).resolve().parents[1] / 'tools/release/build.py'
    spec = importlib.util.spec_from_file_location('release_build', source)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    argument = build.cmake_file_definition('RIKO_NATIVE_BRIDGE_SOURCE',
        PureWindowsPath(r'D:\a\riko_project\riko_project\tools\llama_cpp\riko-native.cpp'))
    assert argument == '-DRIKO_NATIVE_BRIDGE_SOURCE:FILEPATH=D:/a/riko_project/riko_project/tools/llama_cpp/riko-native.cpp'
    assert '\\' not in argument


def test_release_patch_normalizes_windows_line_endings(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'tools/release/build.py'
    spec = importlib.util.spec_from_file_location('release_build', source)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    patch = tmp_path / 'change.patch'
    patch.write_bytes(b'patch line\r\nnext line\r\n')
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: calls.append((args, kwargs)))
    build.apply_patch(tmp_path, patch)
    assert calls[0][0][0] == ['git', 'apply', '--check', '-']
    assert calls[1][0][0] == ['git', 'apply', '-']
    assert all(call[1]['input'] == b'patch line\nnext line\n' for call in calls)
    assert all(call[1]['check'] for call in calls)


def test_upstream_patch_is_limited_to_probe_and_build_glue():
    root = Path(__file__).resolve().parents[1]
    patch = (root / 'tools/llama_cpp/emotion-probe.patch').read_text()
    files = {line.split(' b/', 1)[1] for line in patch.splitlines() if line.startswith('diff --git ')}
    assert files == {'tools/server/CMakeLists.txt', 'tools/server/server-context.cpp',
        'tools/server/server-context.h', 'tools/server/server-task.cpp', 'tools/server/server-task.h'}
    assert not (root / 'tools/llama_cpp/cuda-infinity.patch').exists()


def test_bridge_loads_backend_modules_from_its_own_folder_and_builds_where_it_is_installed():
    # GGML_BACKEND_DL: llama_backend_init loads modules from beside the executable (python, riko-backend) and the working
    # directory, never from native/<backend>/, so the bridge loads them from its own folder into the still empty registry.
    root = Path(__file__).resolve().parents[1]
    bridge = (root / 'tools/llama_cpp/riko-native.cpp').read_text(encoding='utf-8')
    assert 'llama_log_set(riko_log_callback, nullptr); riko_load_backends(); llama_backend_init();' in bridge
    assert 'if (ggml_backend_reg_count() > 0) return;' in bridge and 'ggml_backend_load_all_from_path(' in bridge
    assert 'dladdr(' in bridge and 'GetModuleHandleExW(' in bridge  # the library's own path, not the process's
    # Windows resolves a module's dependencies (cudart, cuBLAS) from its own folder only when asked to.
    assert 'LoadLibraryExW(module.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)' in bridge
    assert 'line.rfind("riko-native: ", 0) == 0' in bridge  # the no-GPU-device warning reaches riko_create's notes
    patch = (root / 'tools/llama_cpp/emotion-probe.patch').read_text(encoding='utf-8')
    # server-context alone as PIC did not link a static Linux build (llama-common's objects are position-dependent too).
    assert 'NOT BUILD_SHARED_LIBS AND NOT CMAKE_POSITION_INDEPENDENT_CODE AND NOT WIN32 AND NOT APPLE' in patch
    assert '+            set_target_properties(riko-native PROPERTIES INSTALL_RPATH "@loader_path")' in patch
    assert '+            set_target_properties(riko-native PROPERTIES INSTALL_RPATH "$ORIGIN")' in patch


def test_frozen_backend_runs_in_utf8_mode():
    # PyInstaller apps ignore PYTHONUTF8, so the backend and its frozen --tool-worker/--discord-worker children get
    # UTF-8 stdio only from this build option (Windows otherwise uses the ANSI code page).
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "'--python-option', 'X utf8'" in source


def test_cuda_bundle_does_not_link_the_nvidia_driver():
    # ggml-cuda otherwise links nvcuda.dll/libcuda.so.1 directly, and the release check could not even load it on a
    # runner without an NVIDIA driver (run_server.DRIVER_LIBRARIES fails the check if it is loaded anyway).
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert '-DGGML_CUDA_NO_VMM=ON' in release_build().native.cmake_flags('cuda')
    assert '*native.cmake_flags(backend)]' in source
    assert 'env=environment' in source and 'release_check_environment(os.environ)' in source


def test_build_names_bundles_from_the_manifest_and_ships_it_to_the_frozen_backend():
    # electron/native_backends.json is the only place that names libraries; the frozen backend resolves bundled:<backend>
    # from the copy PyInstaller puts where native_backends.manifest_path() looks under sys._MEIPASS.
    root = Path(__file__).resolve().parents[1]
    source = (root / 'tools/release/build.py').read_text(encoding='utf-8')
    assert not any(name in source for name in ('riko-native.dll', 'libriko-native', 'GGML_CUDA=', 'GGML_VULKAN='))
    assert source.count('native.library_name(sys.platform)') == 2 and "'--add-data', manifest_data()" in source
    assert release_build().manifest_data() == f'{root / "electron" / "native_backends.json"}{os.pathsep}electron'


def test_release_check_refuses_a_library_bundled_resolution_would_not_find(tmp_path):
    for library in (tmp_path / 'native' / backends_for()[0] / 'riko-native.bogus', tmp_path / 'native' / 'other' / library_name()):
        with pytest.raises(SystemExit, match='bundled:<backend>'): run_server.release_check(library)


def test_release_check_environment_drops_the_build_toolchain():
    toolkit = os.pathsep.join(['/usr/local/cuda-12.4/bin', '/tmp/vulkan-sdk/x86_64/bin', '/opt/NVIDIA GPU Computing Toolkit/v12.4/bin', '/usr/bin', ''])
    environment = release_build().release_check_environment({'PATH': toolkit, 'HOME': '/home/runner', 'LD_LIBRARY_PATH': '/usr/local/cuda/lib64',
        'CUDA_PATH': '/usr/local/cuda', 'CUDA_PATH_V12_4': 'x', 'VULKAN_SDK': '/tmp/vulkan-sdk', 'PYTHONPATH': 'Code',
        'RIKO_CONFIG': 'character_config.yaml', 'RIKO_DATA_DIR': '.'})
    assert environment == {'PATH': '/usr/bin', 'HOME': '/home/runner', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    assert release_build().release_check_environment({'Path': '/opt/Cuda/bin' + os.pathsep + '/windows'})['Path'] == '/windows'


def test_native_bundle_problems_name_shadowed_and_driver_libraries(tmp_path):
    bundle = tmp_path / 'native' / 'cuda'
    bundle.mkdir(parents=True)
    library = bundle / 'libriko-native.so'
    loaded = {'libriko-native.so': library, 'libcublas.so.12': bundle / 'libcublas.so.12'}
    assert run_server.native_bundle_problems(library, loaded) == []
    toolkit = tmp_path / 'cuda' / 'lib64' / 'libcublasLt.so.12'
    problems = run_server.native_bundle_problems(library, {**loaded, 'libcublasLt.so.12': toolkit, 'libcuda.so.1': bundle / 'libcuda.so.1'})
    assert problems == [f'libcublasLt.so.12 was loaded from {toolkit}', f'libcuda.so.1 was loaded from {bundle / "libcuda.so.1"}']


def test_native_release_check_proves_the_bridge_loads_backend_modules_from_its_own_folder(tmp_path, monkeypatch):
    # GGML_BACKEND_DL: ggml looks for modules beside the executable and in the working directory only, so the CPU module
    # loaded from the bundle is the bridge's doing; a module it did not keep must still load from the bundle by itself.
    import ctypes
    bundle = tmp_path / 'native' / 'cuda'
    bundle.mkdir(parents=True)
    names = ['libriko-native.so', 'libggml-base.so.0', 'libggml-cuda.so', 'libggml-cpu-x64.so', 'libggml-cpu-haswell.so', 'libcublas.so.12']
    for name in names: (bundle / name).write_bytes(b'x')
    library, modules, toolkit = bundle / names[0], {'gpu': 'libggml-cuda.so', 'cpu': ['libggml-cpu.so', 'libggml-cpu-*.so']}, tmp_path / 'cuda'

    class Bridge:  # riko_create fails on empty arguments once it has loaded the backends
        riko_request = riko_set_interval = riko_stop = riko_destroy = object()
        riko_devices = staticmethod(riko_devices())
        def __init__(self, created):
            def riko_create(arguments, interval, error, capacity):
                assert arguments == b'[]' and interval == 0 and capacity == len(error)
                loaded.update(created)
            self.riko_create = riko_create
    def run(created, cpu_loads=True):
        loaded.clear(); loaded.update({name: bundle / name for name in names[:2]})
        attempts.clear()
        def load(path):
            if Path(path) == library: return Bridge(created)
            attempts.append(Path(path).name)
            if not cpu_loads and 'cpu' in Path(path).name: raise OSError('undefined symbol')
            loaded[Path(path).name] = Path(path)
        monkeypatch.setattr(ctypes, 'CDLL', load)
        return run_server.native_release_problems(library, modules)
    loaded, attempts = {}, []
    monkeypatch.setattr(run_server, 'loaded_libraries', lambda wanted: {name: path for name, path in loaded.items() if name in wanted or run_server.is_driver(name)})
    # cudart may load the driver once CUDA starts: only a driver that loads with the bundle itself counts.
    assert run({'libggml-cuda.so': bundle / 'libggml-cuda.so', 'libggml-cpu-haswell.so': bundle / 'libggml-cpu-haswell.so',
        'libcublas.so.12': bundle / 'libcublas.so.12', 'libcuda.so.1': Path('/usr/lib/libcuda.so.1')}) == []
    assert attempts == ['libggml-cpu-x64.so']  # the variant ggml did not pick
    assert run({}) == [f'riko_create loaded no CPU backend module (libggml-cpu.so or libggml-cpu-*.so) from {bundle}']
    assert attempts == ['libggml-cpu-haswell.so', 'libggml-cpu-x64.so', 'libggml-cuda.so']
    problems = run({'libcublas.so.12': toolkit / 'libcublas.so.12'}, cpu_loads=False)
    assert problems[1:] == ['libggml-cpu-haswell.so does not load: undefined symbol', 'libggml-cpu-x64.so does not load: undefined symbol',
        f'libcublas.so.12 was loaded from {toolkit / "libcublas.so.12"}']
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: Bridge({}))
    loaded.clear(); loaded['libriko-native.so'] = library
    assert run_server.native_release_problems(library, None) == []  # linked backends (Metal): no module expectations


def test_native_release_check_requires_riko_devices_and_a_cpu_device(capsys):
    # The app treats riko_devices as optional (an older bridge lacks it), but a bundle built from this checkout must have it.
    import ctypes
    class Bridge: pass
    assert run_server.native_device_problems(Bridge()) == ['riko_devices missing: an older riko-native; rebuild it from this checkout (tools/llama_cpp/README.md)']
    def check(function):
        bridge = Bridge(); bridge.riko_devices = function
        return run_server.native_device_problems(bridge)
    gpu = {**CPU_DEVICE, 'index': 1, 'name': 'MTL0', 'description': 'Apple M5 Pro', 'backend': 'MTL', 'type': 'gpu', 'memory_total': 38339 * 1024 ** 2}
    assert check(riko_devices(json.dumps({'version': 1, 'devices': [gpu, CPU_DEVICE]}))) == []
    assert 'ggml devices: MTL0 (Apple M5 Pro, gpu, 38339 MiB), CPU (Test CPU, cpu, 1024 MiB)' in capsys.readouterr().out
    assert check(riko_devices(json.dumps({'version': 1, 'devices': [gpu]}))) == ['riko_devices lists no CPU device']
    assert check(riko_devices('not json'))[0].startswith('riko_devices returned no device list')
    assert check(riko_devices(json.dumps({'version': 2, 'devices': []})))[0].startswith('riko_devices returned no device list (not a version 1')
    assert check(riko_devices('backends failed', -1)) == ['riko_devices returned -1: backends failed']
    assert check(riko_devices('', -2)) == ['riko_devices returned -2: ']  # riko_create had not initialised the backends


def test_loaded_libraries_finds_this_process_system_library():
    name = 'kernel32.dll' if os.name == 'nt' else 'libSystem.B.dylib' if sys.platform == 'darwin' else 'libc.so.6'
    found = run_server.loaded_libraries({name, 'not-a-loaded-library.so'})
    assert list(found) == [name] and found[name].is_absolute()


def test_release_check_passes_on_this_tree_without_a_native_library():
    # The frozen backend runs this same check (tools/release/build.py), so a dynamic import or data file the build
    # would miss, or a broken --tool-worker or --discord-worker path, fails here before a release does.
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'Code/run_server.py'), '--release-check'], cwd=root,
        env=release_build().release_check_environment(os.environ), capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'RIKO_RELEASE_CHECK_OK' in result.stdout


def test_frozen_worker_entry_points_route_through_run_server(tmp_path):
    # Frozen, the backend is its own tool and Discord worker (ToolRegistry, DiscordLauncher): the same argv here.
    root = Path(__file__).resolve().parents[1]
    environment = {**release_build().release_check_environment(os.environ), 'RIKO_DATA_DIR': str(tmp_path)}
    server = [sys.executable, str(root / 'Code/run_server.py')]
    request = {'module': 'process.app_core.tools.builtin.scientific_calculator', 'class': 'Tool', 'arguments': {'expression': '6*7'}}
    tool = subprocess.run([*server, '--tool-worker'], input=json.dumps(request), capture_output=True, text=True, encoding='utf-8',
        env=environment, cwd=tmp_path, timeout=300)
    assert json.loads(tool.stdout.splitlines()[-1]) == {'result': '42'}, tool.stdout + tool.stderr
    discord = subprocess.run([*server, '--discord-worker', '--dry-run'], capture_output=True, text=True, encoding='utf-8',
        env=environment, cwd=tmp_path, timeout=300)
    assert discord.returncode == 0 and 'RIKO_DISCORD_DRY_RUN_OK' in discord.stdout, discord.stdout + discord.stderr
    assert not list(tmp_path.iterdir())  # a dry run writes nothing into the data root


def test_frozen_numpy_and_locked_release_dependencies():
    # numpy >= 2.3 imports numpy._core._exceptions only from C, so PyInstaller cannot see it.
    root = Path(__file__).resolve().parents[1]
    source = (root / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "'--hidden-import', 'numpy._core._exceptions'" in source
    workflow = (root / '.github/workflows/release.yml').read_text(encoding='utf-8')
    assert 'pip install --require-hashes -r tools/release/requirements-lock.txt' in workflow
    assert 'pip install --require-hashes --no-deps -r tools/release/requirements-lock-no-deps.txt' in workflow
    assert '-r requirements-runtime.txt' not in workflow and 'pyinstaller==' not in workflow
    # electron-builder comes from electron/package-lock.json through npm ci, never an ad-hoc install.
    assert 'npm install' not in workflow and 'npx --no-install electron-builder' in workflow


def test_driver_library_is_recognised_under_its_versioned_linux_name_and_ci_steps_stop_on_failure():
    import run_server
    assert all(run_server.is_driver(name) for name in ('nvcuda.dll', 'NVCUDA.DLL', 'libcuda.so', 'libcuda.so.1', 'libcuda.so.550.54.15', 'libcuda.dylib'))
    assert not any(run_server.is_driver(name) for name in ('libcudart.so.12', 'libcuda_helper.so', 'cudart64_12.dll'))
    library = Path('/bundle/libriko-native.so')
    assert run_server.native_bundle_problems(library, {'libcuda.so.550.54.15': Path('/usr/lib/x86_64-linux-gnu/libcuda.so.550.54.15')})
    workflow = (Path(__file__).resolve().parents[1] / '.github/workflows/release.yml').read_text(encoding='utf-8')
    assert 'defaults:\n      run:\n' in workflow and 'shell: bash' in workflow


def test_each_os_builds_its_own_bundles_with_an_rpath_its_loader_expands():
    # dyld ignores $ORIGIN, so a macOS bundle with it would resolve its own libraries only from the build directory.
    build = release_build()
    metal = build.native_flags('metal', 'darwin')
    assert {'-DGGML_METAL=ON', '-DGGML_METAL_EMBED_LIBRARY=ON', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_MACOSX_RPATH=ON', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', '-DCMAKE_INSTALL_RPATH=@loader_path',
        '-DCMAKE_BUILD_RPATH=@loader_path', '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0'} <= set(metal)
    assert not any('$ORIGIN' in flag or 'CUDA_FLAGS' in flag for flag in metal)
    for backend in ('cuda', 'vulkan'):  # unchanged for Linux: Release, then the install rpath for build-tree binaries too
        linux = build.native_flags(backend, 'linux')
        assert linux[-3:] == ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', '-DCMAKE_INSTALL_RPATH=$ORIGIN']
        assert not any('MACOSX' in flag or '@loader_path' in flag or 'DEPLOYMENT_TARGET' in flag for flag in linux)
        windows = build.native_flags(backend, 'win32')  # multi-config Visual Studio: --config Release, and no rpath
        assert f'-DGGML_{backend.upper()}=ON' in windows and not any('RPATH' in flag or 'BUILD_TYPE' in flag for flag in windows)
    hip = build.native_flags('hip', 'linux')  # ROCm's /opt/rocm/lib after the bundle's own folder (the manifest's rpath)
    assert hip[-1] == '-DCMAKE_INSTALL_RPATH=$ORIGIN;/opt/rocm/lib' and '-DGGML_HIP=ON' in hip and '-DGGML_CUDA=OFF' in hip
    assert any(flag.startswith('-DGPU_TARGETS=gfx') for flag in hip)
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    # A default release builds the OS's default bundles (no hip), and release-checks those plus any optional one it staged.
    assert 'for backend in native.release_backends(sys.platform):' in source and 'for backend in staged_backends(bundles):' in source
    assert "('cuda', 'vulkan')" not in source and 'RPATH=$ORIGIN' not in source and '/opt/rocm' not in source


def symlink_or_skip(target, link):
    try: os.symlink(target, link)
    except OSError: pytest.skip('creating symlinks needs a privilege this host lacks')


def test_bundles_keep_versioned_library_links_as_links(tmp_path):
    # libriko-native links @rpath/libggml.0.dylib (libggml.so.0 on Linux): following the links copied every library three times.
    build, release = tmp_path / 'build', release_build()
    destination, outside = tmp_path / 'native' / 'metal', tmp_path / 'elsewhere'
    for folder in (build / 'bin', build / 'CMakeFiles', destination, outside): folder.mkdir(parents=True)
    (build / 'bin' / 'libggml.0.25.3.dylib').write_bytes(b'ggml')
    symlink_or_skip('libggml.0.25.3.dylib', build / 'bin' / 'libggml.0.dylib')
    symlink_or_skip('libggml.0.dylib', build / 'bin' / 'libggml.dylib')
    (outside / 'libextra.dylib').write_bytes(b'extra')
    symlink_or_skip(outside / 'libextra.dylib', build / 'bin' / 'libextra.dylib')
    for name in ('bin/libriko-native.dylib', 'bin/ggml.dll', 'bin/libggml.so.0', 'bin/libggml.so', 'CMakeFiles/notes.dylib.txt', 'bin/version.sorted'):
        (build / name).write_bytes(b'x')
    found = release.shared_libraries(build, 'darwin')
    assert [file.name for file in found] == ['libextra.dylib', 'libggml.0.25.3.dylib', 'libggml.0.dylib', 'libggml.dylib', 'libriko-native.dylib']
    for _ in range(2):  # a rerun replaces what an earlier run copied
        for file in found: release.copy_shared_library(file, destination)
    assert os.readlink(destination / 'libggml.dylib') == 'libggml.0.dylib' and os.readlink(destination / 'libggml.0.dylib') == 'libggml.0.25.3.dylib'
    assert not (destination / 'libggml.0.25.3.dylib').is_symlink() and (destination / 'libggml.dylib').read_bytes() == b'ggml'
    assert not (destination / 'libextra.dylib').is_symlink() and (destination / 'libextra.dylib').read_bytes() == b'extra'
    assert release.dangling_links(destination) == []
    (destination / 'libggml.0.25.3.dylib').unlink()
    assert release.dangling_links(destination) == ['libggml.0.dylib', 'libggml.dylib']
    assert [file.name for file in release.shared_libraries(build, 'linux')] == ['libggml.so', 'libggml.so.0']  # not version.sorted
    assert [file.name for file in release.shared_libraries(build, 'win32')] == ['ggml.dll']


def test_cuda_redistributables_go_only_into_the_cuda_bundle_with_their_links(tmp_path):
    release = release_build()
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert source.count('copy_cuda_redistributables(destination)') == 1 and "if backend == 'cuda': copy_cuda_redistributables(destination)" in source
    toolkit, destination = tmp_path / 'cuda' / 'targets' / 'x86_64-linux' / 'lib', tmp_path / 'native' / 'cuda'
    for folder in (toolkit, destination): folder.mkdir(parents=True)
    for library, version in (('libcudart', '12.4.127'), ('libcublas', '12.4.5.8'), ('libcublasLt', '12.4.5.8')):
        (toolkit / f'{library}.so.{version}').write_bytes(library.encode())
        symlink_or_skip(f'{library}.so.{version}', toolkit / f'{library}.so.12')
        symlink_or_skip(f'{library}.so.12', toolkit / f'{library}.so')
    (toolkit / 'libcuda.so').write_bytes(b'driver stub')
    (toolkit / 'stubs').mkdir()  # link-time stubs share the real names; copied after the links, one replaced libcublas.so
    for name in ('libcuda.so', 'libcublas.so', 'libcublasLt.so'): (toolkit / 'stubs' / name).write_bytes(b'stub')
    release.copy_cuda_redistributables(destination, 'linux', tmp_path / 'cuda')
    assert sorted(file.name for file in destination.iterdir() if not file.is_symlink()) == ['libcublas.so.12.4.5.8', 'libcublasLt.so.12.4.5.8', 'libcudart.so.12.4.127']
    assert os.readlink(destination / 'libcublasLt.so.12') == 'libcublasLt.so.12.4.5.8' and release.dangling_links(destination) == []
    assert os.readlink(destination / 'libcublas.so') == 'libcublas.so.12'
    assert not (destination / 'libcuda.so').exists()
    with pytest.raises(RuntimeError, match='CUDA redistributable missing'): release.copy_cuda_redistributables(destination, 'linux', tmp_path / 'none')


def test_mac_icon_is_rendered_from_the_logo_where_electron_builder_packages_it(tmp_path, monkeypatch):
    root, release, calls = Path(__file__).resolve().parents[1], release_build(), []
    monkeypatch.setattr(release, 'run', lambda *args, **kwargs: calls.append([str(arg) for arg in args]))
    icns = release.mac_icon(root / 'assets/logo.svg', tmp_path / 'mac' / 'icon.icns')
    assert calls[-1] == ['iconutil', '-c', 'icns', str(tmp_path / 'mac' / 'icon.iconset'), '-o', str(icns)]
    renders = [call for call in calls if call[1:3] == ['-s', 'format']]
    assert len(renders) == 10 and all(call[7:9] == [str(root / 'assets/logo.svg'), '--out'] for call in renders)
    assert renders[-1][4:7] == ['-z', '880', '880'] and calls[-2][1:4] == ['-p', '1024', '1024']  # 512@2x: 880 px of art, padded
    assert sorted({int(call[2]) for call in calls if call[1] == '-p'}) == [16, 32, 64, 128, 256, 512, 1024]
    source = (root / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "if sys.platform == 'darwin': mac_icon(ROOT / 'assets/logo.svg', STAGE / 'mac' / 'icon.icns')" in source
    builder = yaml.safe_load((root / 'electron/electron-builder.yml').read_text(encoding='utf-8'))
    assert (root / 'electron' / builder['mac']['icon']).resolve() == (release.STAGE / 'mac' / 'icon.icns').resolve()


@pytest.mark.skipif(sys.platform != 'darwin', reason='sips and iconutil are part of macOS')
def test_mac_icon_renders_with_the_system_tools(tmp_path):
    icns = release_build().mac_icon(Path(__file__).resolve().parents[1] / 'assets/logo.svg', tmp_path / 'icon.icns')
    assert icns.read_bytes()[:4] == b'icns' and icns.stat().st_size > 10_000


def test_frozen_backend_is_moved_into_the_stage_with_its_symlinks():
    # A copy that followed PyInstaller's links stored libtorch and friends twice; the stage holds PyInstaller's output once.
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "'--distpath', WORK / 'dist'" in source and "os.replace(WORK / 'dist' / 'riko-backend', STAGE / 'backend')" in source
    assert 'copytree' not in source and "'--distpath', STAGE" not in source


def test_release_build_refuses_to_build_over_an_earlier_run(tmp_path, monkeypatch):
    # Merging into release-stage/ shipped an earlier run's libraries and Python modules and listed them in SHA256SUMS.json.
    release, calls = release_build(), []
    monkeypatch.setattr(release, 'run', lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    monkeypatch.setattr(release, 'STAGE', tmp_path / 'release-stage')
    monkeypatch.setattr(release, 'WORK', tmp_path / 'release-build')
    for stale in ('release-stage/native', 'release-build', '.native/llama.cpp-release'):
        (tmp_path / stale).mkdir(parents=True)
        with pytest.raises(RuntimeError, match='refusing to build over'): release.main()
    with pytest.raises(RuntimeError, match='release-stage, release-build'): release.main()
    assert calls == []
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "'tools/release/NOTICE.md', notices / 'NOTICE.md'" in source and "'tools/release/README.md'" not in source  # not maintainer notes


def test_dynamic_backend_bundles_must_hold_their_gpu_and_a_cpu_module(tmp_path):
    release = release_build()
    for platform, files in (('linux', ['libriko-native.so', 'libggml-base.so.0', 'libggml-cuda.so', 'libggml-cpu-x64.so', 'libggml-cpu-haswell.so']),
            ('win32', ['riko-native.dll', 'ggml-base.dll', 'ggml-cuda.dll', 'ggml-cpu-sse42.dll'])):
        bundle = tmp_path / platform / 'cuda'
        bundle.mkdir(parents=True)
        for name in files: (bundle / name).write_bytes(b'x')
        assert release.missing_modules(bundle, 'cuda', platform) == []
        for name in files[2:]: (bundle / name).unlink()
        assert release.missing_modules(bundle, 'cuda', platform) == [files[2], ' or '.join(release.native.backend_modules('cuda', platform)['cpu'])]
    assert release.missing_modules(tmp_path / 'linux' / 'cuda', 'metal', 'darwin') == []  # linked backends: nothing to look for
    flags = release.native_flags('vulkan', 'linux')
    assert {'-DGGML_BACKEND_DL=ON', '-DGGML_CPU_ALL_VARIANTS=ON', '-DGGML_NATIVE=OFF', '-DBUILD_SHARED_LIBS=ON'} <= set(flags)
    assert '-DGGML_BACKEND_DL=ON' not in release.native_flags('metal', 'darwin')


def test_release_workflow_packages_a_macos_dmg_without_the_cuda_and_vulkan_steps():
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / '.github/workflows/release.yml').read_text(encoding='utf-8'))
    job = workflow['jobs']['package']
    assert job['strategy']['matrix']['os'] == ['windows-2022', 'ubuntu-22.04', 'macos-15']  # macos-15: Apple silicon
    steps = {step.get('name'): step for step in job['steps']}
    assert steps['CUDA build toolkit']['if'] == "runner.os != 'macOS'"
    assert steps['Linux build prerequisites']['if'] == steps['Linux Vulkan SDK shader compiler']['if'] == "runner.os == 'Linux'"
    assert steps['Windows Vulkan SDK']['if'] == "runner.os == 'Windows'"
    assert steps['Build the native backends and freeze Python']['run'] == 'python tools/release/build.py'
    package = steps['Package installers']
    assert package['env']['CSC_IDENTITY_AUTO_DISCOVERY'] == 'false'
    secrets = {key: value for key, value in package['env'].items() if key.startswith('MAC_')}
    assert len(secrets) == 5 and all(value.startswith("${{ runner.os == 'macOS' && secrets.") for value in secrets.values())
    # An empty CSC_LINK makes electron-builder import an empty certificate: export it only when the secret is set.
    script = package['run']
    assert 'CSC_LINK' not in package['env'] and script.index('if [ -n "$MAC_CSC_LINK" ]; then') < script.index('export CSC_LINK=')
    assert script.index('export CSC_LINK=') < script.index('export APPLE_API_KEY=') < script.index('npx --no-install electron-builder')
    upload = job['steps'][-1]['with']
    assert 'electron/release/*.dmg' in upload['path']
    assert upload['name'] == "Riko-${{ runner.os }}-${{ runner.os == 'macOS' && 'arm64' || 'x64' }}${{ runner.os == 'Linux' && needs.hip.result == 'success' && '-hip' || '' }}"


def test_pyinstaller_sees_the_app_package_when_its_spec_collects_submodules():
    # The spec's collect_submodules('process') runs before Analysis applies --paths: without Code/ on PYTHONPATH it returned
    # nothing, and the frozen --tool-worker failed the release check with "No module named process.app_core.tools.worker".
    root, release = Path(__file__).resolve().parents[1], release_build()
    source = (root / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "'--collect-submodules', 'process'" in source and "ROOT / 'Code/run_server.py', env=freeze_environment(os.environ))" in source
    assert release.freeze_environment({'PATH': '/usr/bin'}) == {'PATH': '/usr/bin', 'PYTHONPATH': str(root / 'Code')}
    assert release.freeze_environment({'PYTHONPATH': 'extra'})['PYTHONPATH'] == str(root / 'Code') + os.pathsep + 'extra'
    pytest.importorskip('PyInstaller', reason='release CI installs PyInstaller from the hash lock')  # what the spec then sees:
    found = subprocess.run([sys.executable, '-c', "from PyInstaller.utils.hooks import collect_submodules; print('\\n'.join(collect_submodules('process')))"],
        cwd=root, env=release.freeze_environment({key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}),
        capture_output=True, text=True, timeout=300).stdout.split()
    assert {'process.app_core.tools.worker', 'process.app_core.tools.builtin.scientific_calculator', 'process.app_core.tools.builtin.todo_list',
        *run_server.KERNEL_MODULES} <= set(found)


def test_release_check_covers_every_kernel_module_and_the_lazily_imported_server():
    # Code imports some kernel/ modules only inside functions, and run_server imports uvicorn only after dispatching the
    # worker modes, so the frozen check imports them by name.
    kernel = Path(__file__).resolve().parents[1] / 'Code/process/app_core/kernel'
    assert set(run_server.KERNEL_MODULES) == {f'process.app_core.kernel.{path.stem}' for path in kernel.glob('*.py') if path.stem != '__init__'}
    assert 'uvicorn' in run_server.RELEASE_MODULES
    # What each frozen worker call no longer pays for: neither the import nor main()'s dispatch loads them (workers stubbed).
    for flag in ('', '--tool-worker', '--discord-worker'):
        probe = ("import runpy, sys, run_server\n" + (f"runpy.run_module = lambda *a, **k: None\nsys.argv = ['run_server.py', '{flag}']\nrun_server.main()\n" if flag else '')
            + "print(sorted({'uvicorn', 'process.app_core.configuration.config'} & set(sys.modules)))")
        loaded = subprocess.run([sys.executable, '-c', probe], cwd=kernel.parents[2], capture_output=True, text=True, timeout=120)
        assert loaded.stdout.strip() == '[]', (flag, loaded.stdout + loaded.stderr)


def test_release_check_runs_per_shipped_bundle_on_the_stage_and_inside_the_signed_mac_app(tmp_path, monkeypatch):
    release, calls = release_build(), []
    monkeypatch.setattr(release, 'run', lambda *args, **kwargs: calls.append(([str(arg) for arg in args], kwargs['env'])))
    app = tmp_path / 'Riko.app'
    release.check_app(app)
    resources = app / 'Contents' / 'Resources'
    assert [call[0] for call in calls] == [[str(resources / 'backend' / 'riko-backend'), '--release-check', str(resources / 'native' / backend / library_name())]
        for backend in release_backends()]  # an optional bundle (hip) only when the stage holds one:
    assert all(call[1] == release.release_check_environment(os.environ) for call in calls)
    bundles = tmp_path / 'native'
    assert release.staged_backends(bundles, 'linux') == ['cuda', 'vulkan'] and release.staged_backends(bundles, 'darwin') == ['metal']
    (bundles / 'hip').mkdir(parents=True)
    assert release.staged_backends(bundles, 'linux') == ['cuda', 'hip', 'vulkan'] and release.staged_backends(bundles, 'win32') == ['cuda', 'vulkan']
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "    release_check(executable, STAGE / 'native')\n" in source
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / '.github/workflows/release.yml').read_text(encoding='utf-8'))
    steps = {step.get('name'): step for step in workflow['jobs']['package']['steps']}
    check = steps['Release check inside the signed macOS app']
    assert check['if'] == "runner.os == 'macOS'" and check['run'] == 'python tools/release/build.py --check-app electron/release/mac-arm64/Riko.app'
    names = [step.get('name') for step in workflow['jobs']['package']['steps']]
    assert names.index('Package installers') < names.index('Release check inside the signed macOS app') < len(names) - 1  # before the upload


def test_probe_feature_version_matches_the_native_patch():
    # The bridge reports this string in /props and tags every sample with it; on a mismatch the probe does not start (the
    # model still loads). It reports no width, so the pooled width, the event name and the capture slot are checked here.
    from process.app_core.emotion.probe_hook import CAPTURE_SLOT, FEATURE_VERSION, FEATURE_WIDTH, SAMPLE_EVENT
    patch = (Path(__file__).resolve().parents[1] / 'tools/llama_cpp/emotion-probe.patch').read_text(encoding='utf-8')
    assert f'"{FEATURE_VERSION}"' in patch and f'"{SAMPLE_EVENT}"' in patch and f'token.id_slot == {CAPTURE_SLOT} ' in patch
    assert f'std::vector<float> features({FEATURE_WIDTH});' in patch and f'riko_probe_features.size() == {FEATURE_WIDTH})' in patch


HIP_IMAGE = 'rocm/dev-ubuntu-22.04:7.2.1@sha256:42851dac319afce41cf993e25f95005b7f2cd0a0f6abd32ad8f25cd876ec56df'


def test_hip_bundle_is_opt_in_and_built_by_the_compile_checked_recipe():
    # Tag pushes and default dispatches build what they always did; the hip input adds a job in the ROCm container whose
    # bundle the ubuntu leg stages. rocm-compile.yml's compile job runs the same action in the same image (by digest).
    root = Path(__file__).resolve().parents[1]
    release = yaml.safe_load((root / '.github/workflows/release.yml').read_text(encoding='utf-8'))
    smoke = yaml.safe_load((root / '.github/workflows/native-smoke.yml').read_text(encoding='utf-8'))
    compile_workflow = yaml.safe_load((root / '.github/workflows/rocm-compile.yml').read_text(encoding='utf-8'))
    triggers = release[True]  # PyYAML reads the key `on` as True
    assert triggers['push'] == {'tags': ['v*']} and triggers['workflow_dispatch']['inputs']['hip'] == {
        'description': triggers['workflow_dispatch']['inputs']['hip']['description'], 'type': 'boolean', 'default': False}
    hip, compile_job = release['jobs']['hip'], compile_workflow['jobs']['hip']
    assert hip['if'] == '${{ inputs.hip }}' and hip['container'] == compile_job['container'] == HIP_IMAGE
    action = {'uses': './.github/actions/hip-bundle', 'with': {'destination': 'native-hip/hip'}}
    assert hip['steps'][1] == compile_job['steps'][1] == action and compile_job['steps'][0]['uses'] == hip['steps'][0]['uses']
    assert re.fullmatch(r'actions/checkout@[0-9a-f]{40}', hip['steps'][0]['uses'])  # tests/test_ci_workflows.py: every action by commit
    assert hip['steps'][-2]['run'] == 'tar -C native-hip -cf native-hip.tar hip'  # an artifact upload would copy every link
    assert hip['steps'][-1]['with']['name'] == 'native-hip' and hip['steps'][-1]['with']['path'] == 'native-hip.tar'
    package = release['jobs']['package']
    # Windows and macOS never wait on hip; a requested hip that did not build (failed or timed out) stops only Linux.
    assert package['needs'] == 'hip' and package['if'] == '${{ !cancelled() }}'
    steps = {step.get('name'): step for step in package['steps']}
    assert package['steps'][0]['name'] == 'Requested HIP bundle was not built' and steps['Requested HIP bundle was not built']['if'] == (
        "runner.os == 'Linux' && inputs.hip && needs.hip.result != 'success'") and 'exit 1' in steps['Requested HIP bundle was not built']['run']
    assert steps['Opt-in HIP bundle']['if'] == steps['Unpack the HIP bundle']['if'] == "runner.os == 'Linux' && needs.hip.result == 'success'"
    assert steps['Build the native backends and freeze Python']['env']['RIKO_OPTIONAL_BUNDLES'] == (
        "${{ runner.os == 'Linux' && needs.hip.result == 'success' && format('{0}/optional', runner.temp) || '' }}")
    assert '"$RUNNER_TEMP/optional"' in steps['Unpack the HIP bundle']['run']
    names = [step.get('name') for step in package['steps']]
    assert names.index('Unpack the HIP bundle') < names.index('Build the native backends and freeze Python')
    # Twelve GPU targets take hours: the compile job runs only when asked, and a later push never cancels it.
    assert compile_workflow[True] == {'workflow_dispatch': None} and compile_workflow['concurrency']['cancel-in-progress'] is False
    assert 'hip' not in smoke['jobs'] and not any('hip' in path for event in ('push', 'pull_request') for path in smoke[True][event]['paths'])
    recipe = yaml.safe_load((root / '.github/actions/hip-bundle/action.yml').read_text(encoding='utf-8'))
    script = '\n'.join(step['run'] for step in recipe['runs']['steps'])
    assert 'apt-get install -y --no-install-recommends git cmake hipblas-dev rocblas-dev' in script
    assert 'python3 tools/release/build.py --smoke-bundle hip "$RUNNER_TEMP/llama.cpp-hip" "${{ inputs.destination }}"' in script
    assert 'python3 tools/release/build.py --check-bundle "${{ inputs.destination }}/libriko-native.so"' in script
    assert '$ORIGIN:/opt/rocm/lib' in script and 'RIKO_BUILD_JOBS="$(nproc)"' in script and 'HIPCXX="$(/opt/rocm/bin/hipconfig -l)/clang"' in script


def test_optional_bundles_are_staged_only_for_this_os_and_checked_like_built_ones(tmp_path):
    release = release_build()
    assert release.optional_bundles(None, 'linux') == {} and release.optional_bundles('', 'linux') == {}
    prebuilt = tmp_path / 'optional' / 'hip'
    prebuilt.mkdir(parents=True)
    for name in ('libriko-native.so', 'libggml-base.so.0.9.4', 'libggml-hip.so', 'libggml-cpu-haswell.so', 'libggml-cpu-x64.so'): (prebuilt / name).write_bytes(b'x')
    symlink_or_skip('libggml-base.so.0.9.4', prebuilt / 'libggml-base.so.0')
    assert release.optional_bundles(tmp_path / 'optional', 'linux') == {'hip': prebuilt}
    for platform in ('win32', 'darwin'):  # hip ships only for Linux
        with pytest.raises(RuntimeError, match='no optional backend ships for ' + platform): release.optional_bundles(tmp_path / 'optional', platform)
    with pytest.raises(RuntimeError, match='not a folder'): release.optional_bundles(tmp_path / 'missing', 'linux')
    (tmp_path / 'optional' / 'cuda').mkdir()  # default bundles are built, never taken from elsewhere
    with pytest.raises(RuntimeError, match='named hip; found cuda'): release.optional_bundles(tmp_path / 'optional', 'linux')
    (tmp_path / 'optional' / 'cuda').rmdir()
    staged = tmp_path / 'stage' / 'native' / 'hip'
    assert release.stage_bundle(prebuilt, 'hip', staged, 'linux') == staged / 'libriko-native.so'
    assert os.readlink(staged / 'libggml-base.so.0') == 'libggml-base.so.0.9.4'  # links stay links
    (prebuilt / 'libggml-hip.so').unlink()
    with pytest.raises(RuntimeError, match=r"modules missing from the bundle: \['libggml-hip.so'\]"):
        release.stage_bundle(prebuilt, 'hip', tmp_path / 'stage2' / 'hip', 'linux')
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "optional = optional_bundles(os.environ.get('RIKO_OPTIONAL_BUNDLES'))" in source  # checked before the hours of building
    assert "for backend, folder in optional.items(): stage_bundle(folder, backend, STAGE / 'native' / backend)" in source


def test_build_jobs_default_to_two_and_follow_riko_build_jobs():
    release = release_build()
    assert release.build_jobs({}) == '2' and release.build_jobs({'RIKO_BUILD_JOBS': '4'}) == '4' and release.build_jobs({'RIKO_BUILD_JOBS': ''}) == '2'
    for value in ('0', '-1', 'all', '2 3'):
        with pytest.raises(RuntimeError, match='RIKO_BUILD_JOBS'): release.build_jobs({'RIKO_BUILD_JOBS': value})


def test_release_check_accepts_a_missing_system_library_only_where_the_manifest_names_it(tmp_path, monkeypatch):
    # The hip bundle links the user's ROCm, which the release runner lacks: ggml-hip cannot load there, and that is expected.
    import ctypes
    from process.app_core.configuration.native_backends import backend_modules, system_libraries
    bundle = tmp_path / 'native' / 'hip'
    bundle.mkdir(parents=True)
    names = ['libriko-native.so', 'libggml-base.so.0', 'libggml-hip.so', 'libggml-cpu-haswell.so']
    for name in names: (bundle / name).write_bytes(b'x')
    library, loaded = bundle / names[0], {}

    class Bridge:  # riko_create loads the CPU module ggml picked; ggml-hip's ROCm is missing
        riko_request = riko_set_interval = riko_stop = riko_destroy = object()
        riko_devices = staticmethod(riko_devices())
        def __init__(self): self.riko_create = lambda arguments, interval, error, capacity: loaded.update({'libggml-cpu-haswell.so': bundle / 'libggml-cpu-haswell.so'})
    def load(path, missing='libamdhip64.so.7'):
        if Path(path) == library: return Bridge()
        raise OSError(f'{missing}: cannot open shared object file: No such file or directory')
    monkeypatch.setattr(ctypes, 'CDLL', load)
    monkeypatch.setattr(run_server, 'loaded_libraries', lambda wanted: {name: path for name, path in {'libriko-native.so': library, **loaded}.items() if name in wanted})
    modules = backend_modules('hip', 'linux')
    assert run_server.native_release_problems(library, modules, system_libraries('hip')) == []
    assert run_server.native_release_problems(library, modules) == ['libggml-hip.so does not load: libamdhip64.so.7: cannot open shared object file: No such file or directory']
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: load(path, 'libggml-base.so.0'))  # a bundle library is never the system's
    assert run_server.native_release_problems(library, modules, system_libraries('hip')) == [
        'libggml-hip.so does not load: libggml-base.so.0: cannot open shared object file: No such file or directory']
    source = (Path(__file__).resolve().parents[1] / 'Code/run_server.py').read_text(encoding='utf-8')
    assert 'native_release_problems(library, backend_modules(backend), system_libraries(backend))' in source


def test_check_bundle_is_the_strict_native_check_for_a_build_machine_with_the_system_libraries(tmp_path):
    # In the ROCm container ROCm is installed, so build.py --check-bundle passes no system patterns: ggml-hip must load.
    release = release_build()
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert 'problems = server.native_release_problems(library, native.backend_modules(backend))\n' in source
    for library in (tmp_path / 'native' / 'other' / library_name(), tmp_path / 'native' / backends_for()[0] / 'riko-native.bogus'):
        with pytest.raises(SystemExit, match='is not a native/'): release.check_bundle(library)
