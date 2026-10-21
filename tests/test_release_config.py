from fnmatch import fnmatch
import json
from pathlib import Path
import re
import sys

import pytest

from process.app_core.configuration import native_backends
from process.app_core.configuration.config import load_config
from process.app_core.configuration.native_backends import (backend_modules, backends_for, bundled_library, cmake_flags, library_name,
    platform_build, release_backends, rpath, system_libraries)

# Written out, not derived: each OS is checked on every host (electron/src/native_backends.test.mjs holds the same tables).
LIBRARIES = {'win32': 'riko-native.dll', 'linux': 'libriko-native.so', 'darwin': 'libriko-native.dylib'}
SHIPPED = {'win32': ['cuda', 'vulkan'], 'linux': ['cuda', 'hip', 'vulkan'], 'darwin': ['metal']}  # what bundled:<backend> resolves
RELEASED = {'win32': ['cuda', 'vulkan'], 'linux': ['cuda', 'vulkan'], 'darwin': ['metal']}  # what a default release builds: hip is opt-in


def test_bundled_backend_resolves_current_install_location(tmp_path, monkeypatch):
    backend = backends_for()[0]  # cuda on Windows and Linux, metal on macOS
    config = tmp_path / 'data' / 'character_config.yaml'
    config.parent.mkdir()
    config.write_text(f'runtime:\n  provider: llama_cpp\n  native_library: bundled:{backend}\n  hf_repo_id: owner/model\n  hf_filename: model.gguf\n')
    resources = tmp_path / 'application'
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(resources))
    loaded = load_config(config)
    assert loaded.root == config.parent
    assert loaded.runtime.native_library == resources / 'native' / backend / LIBRARIES[sys.platform]
    assert loaded.memory.store_file.is_relative_to(config.parent)
    moved = tmp_path / 'upgraded-app'
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(moved))
    assert load_config(config).runtime.native_library.is_relative_to(moved)


def test_bundled_backend_rejects_untrusted_names(tmp_path, monkeypatch):
    config = tmp_path / 'character_config.yaml'
    config.write_text('runtime:\n  native_library: bundled:../../other\n')
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='Bundled native'): load_config(config)


@pytest.mark.parametrize('platform', sorted(LIBRARIES))
def test_bundled_library_names_and_backends_per_os(platform, tmp_path):
    assert library_name(platform) == LIBRARIES[platform]
    assert backends_for(platform) == SHIPPED[platform] and release_backends(platform) == RELEASED[platform]
    for backend in SHIPPED[platform]:  # existing bundled:cuda and bundled:vulkan configs keep resolving, and bundled:hip on Linux
        assert bundled_library(backend, tmp_path, platform) == tmp_path / 'native' / backend / LIBRARIES[platform]
    for backend in {'cuda', 'hip', 'vulkan', 'metal'} - set(SHIPPED[platform]):
        with pytest.raises(ValueError, match=f'not shipped for {platform}; use bundled:{SHIPPED[platform][0]}'):
            bundled_library(backend, tmp_path, platform)
    for backend in ('../../other', '', 'CUDA', 'cuda/../metal'):
        with pytest.raises(ValueError, match='Bundled native'): bundled_library(backend, tmp_path, platform)
    with pytest.raises(ValueError, match='packaged app'): bundled_library(SHIPPED[platform][0], None, platform)


def test_unsupported_platform_has_no_bundled_backend(tmp_path):
    with pytest.raises(ValueError, match='no bundled backend ships for freebsd'): bundled_library('cuda', tmp_path, 'freebsd')
    with pytest.raises(ValueError, match='freebsd'): library_name('freebsd')


def test_manifest_carries_what_the_release_build_needs():
    assert native_backends.validate(native_backends.manifest()) is native_backends.manifest()
    assert cmake_flags('cuda') == ['-DGGML_CUDA=ON', '-DGGML_HIP=OFF', '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
        '-DCMAKE_CUDA_FLAGS=--diag-suppress=221', '-DGGML_CUDA_NO_VMM=ON', '-DGGML_BACKEND_DL=ON', '-DGGML_CPU_ALL_VARIANTS=ON']
    assert cmake_flags('vulkan') == ['-DGGML_CUDA=OFF', '-DGGML_HIP=OFF', '-DGGML_VULKAN=ON', '-DGGML_METAL=OFF', '-DGGML_BACKEND_DL=ON', '-DGGML_CPU_ALL_VARIANTS=ON']
    assert cmake_flags('metal') == ['-DGGML_CUDA=OFF', '-DGGML_HIP=OFF', '-DGGML_VULKAN=OFF', '-DGGML_METAL=ON', '-DGGML_METAL_EMBED_LIBRARY=ON',
        '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0']
    assert cmake_flags('hip')[:4] == ['-DGGML_CUDA=OFF', '-DGGML_HIP=ON', '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF']
    with pytest.raises(ValueError, match='Unknown native backend'): cmake_flags('sycl')
    # macOS dyld ignores $ORIGIN; Windows finds a DLL's neighbours without an rpath.
    assert {platform: platform_build(platform).get('rpath') for platform in LIBRARIES} == {'win32': None, 'linux': '$ORIGIN', 'darwin': '@loader_path'}
    bundles = {'win32': ['riko-native.dll', 'ggml-cuda.dll', 'cublas64_12.dll'], 'linux': ['libriko-native.so', 'libggml-base.so', 'libcublas.so.12'],
        'darwin': ['libriko-native.dylib', 'libggml-metal.dylib', 'libllama.dylib']}
    for platform, files in bundles.items():
        patterns = platform_build(platform)['shared_libraries']
        assert all(any(fnmatch(file, pattern) for pattern in patterns) for file in files), platform
        assert not any(fnmatch(file, pattern) for pattern in patterns for other in set(bundles) - {platform} for file in bundles[other][:1])


def test_x86_bundles_load_ggml_backends_as_modules_with_a_cpu_build_per_isa_level():
    # GGML_NATIVE=OFF alone builds one CPU backend that assumes AVX2/FMA/F16C/BMI2 (SIGILL on older x86); the variants let
    # ggml pick what this CPU runs. Metal stays linked: every Apple silicon Mac runs the arm64 build.
    assert backend_modules('cuda', 'linux') == {'gpu': 'libggml-cuda.so', 'cpu': ['libggml-cpu.so', 'libggml-cpu-*.so']}
    assert backend_modules('vulkan', 'win32') == {'gpu': 'ggml-vulkan.dll', 'cpu': ['ggml-cpu.dll', 'ggml-cpu-*.dll']}
    assert backend_modules('metal', 'darwin') is None and '-DGGML_BACKEND_DL=ON' not in cmake_flags('metal')
    for backend in ('cuda', 'vulkan'):
        for platform in ('win32', 'linux'):  # build.py copies a bundle's files by these patterns: the modules among them
            modules = backend_modules(backend, platform)
            assert all(any(fnmatch(name, pattern) for pattern in platform_build(platform)['shared_libraries'])
                for name in (modules['gpu'], modules['cpu'][1].replace('*', 'haswell')))
    assert backend_modules('hip', 'linux') == {'gpu': 'libggml-hip.so', 'cpu': ['libggml-cpu.so', 'libggml-cpu-*.so']}
    data = json.loads(native_backends.manifest_path().read_text(encoding='utf-8'))
    data['backends'].append({'id': 'cpu', 'label': 'CPU', 'description': 'no GPU', 'platforms': ['linux'], 'cmake_switch': None,
        'cmake_flags': ['-DGGML_BACKEND_DL=ON', '-DGGML_CPU_ALL_VARIANTS=ON']})
    native_backends.validate(data)
    assert backend_modules('cpu', 'linux', data)['gpu'] is None


def test_manifest_leaves_room_for_a_cpu_bundle():
    data = native_backends.validate(json.loads(native_backends.manifest_path().read_text(encoding='utf-8')))
    data['backends'].append({'id': 'cpu', 'label': 'CPU', 'description': 'no GPU', 'platforms': ['win32', 'linux', 'darwin'], 'cmake_switch': None})
    native_backends.validate(data)
    assert backends_for('linux', data) == ['cuda', 'hip', 'vulkan', 'cpu'] and backends_for('darwin', data) == ['metal', 'cpu']
    assert release_backends('linux', data) == ['cuda', 'vulkan', 'cpu']
    assert cmake_flags('cpu', data) == ['-DGGML_CUDA=OFF', '-DGGML_HIP=OFF', '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF']
    assert bundled_library('cpu', '/app', 'darwin', data) == Path('/app/native/cpu/libriko-native.dylib')


def test_hip_is_an_opt_in_linux_bundle_for_amd_gpus_with_rocm():
    # The release workflow ships it only when asked (its hip input); bundled:hip resolves on Linux like any shipped bundle.
    hip = next(backend for backend in native_backends.manifest()['backends'] if backend['id'] == 'hip')
    assert hip['optional'] is True and hip['platforms'] == ['linux'] and hip['cmake_switch'] == 'GGML_HIP'
    assert bundled_library('hip', '/app', 'linux') == Path('/app/native/hip/libriko-native.so')
    with pytest.raises(ValueError, match='not shipped for win32'): bundled_library('hip', '/app', 'win32')
    flags = cmake_flags('hip')
    assert {'-DGGML_BACKEND_DL=ON', '-DGGML_CPU_ALL_VARIANTS=ON'} <= set(flags)  # modules and CPU variants as for cuda and vulkan
    targets = next(flag for flag in flags if flag.startswith('-DGPU_TARGETS=')).split('=', 1)[1].split(';')
    # Written out: CDNA (MI100, MI200, MI300, MI350), RDNA2, RDNA3, RDNA3.5 and RDNA4, as the manifest's notes document.
    assert targets == ['gfx908', 'gfx90a', 'gfx942', 'gfx950', 'gfx1030', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1150', 'gfx1151', 'gfx1200', 'gfx1201']
    # The user's ROCm, never bundled: found beside the bundle, then where AMD's packages put it.
    assert rpath('hip', 'linux') == '$ORIGIN;/opt/rocm/lib' and rpath('cuda', 'linux') == '$ORIGIN' and rpath('hip', 'win32') is None
    assert rpath('metal', 'darwin') == '@loader_path'
    patterns = system_libraries('hip')
    for name in ('libamdhip64.so.7', 'libhipblas.so.3', 'librocblas.so.5'): assert any(fnmatch(name, pattern) for pattern in patterns), name
    assert not any(fnmatch(name, pattern) for pattern in patterns for name in ('libggml-hip.so', 'libggml-base.so.0', 'libcublas.so.12'))
    assert system_libraries('cuda') == system_libraries('vulkan') == system_libraries('metal') == []
    # Detection: only GPU agent names count (rocminfo also lists the CPU and each agent's ISA), each must be a built target
    # (detect.targets names the flag), and ROCm 7's runtime must be installed. amd-smi is not proof: it reports a GPU
    # without the /dev/kfd access HIP needs.
    detect, rocminfo = hip['detect'], ('  Name:                    AMD Ryzen 9 7950X\n  Name:                    gfx1100\n  Marketing Name:          Radeon RX 7900 XTX\n'
        '      Name:                    amdgcn-amd-amdhsa--gfx1100')
    assert detect['required'] is True and detect['command'] == 'rocminfo' and detect['fallbacks'] == [['/opt/rocm/bin/rocminfo']]
    assert [line.strip() for line in rocminfo.splitlines() if re.search(detect['pattern'], line.strip())] == ['Name:                    gfx1100']
    assert detect['targets'] == 'GPU_TARGETS' and '/opt/rocm/lib/libamdhip64.so.7' in detect['runtime']


def entry(data, backend): return next(item for item in data['backends'] if item['id'] == backend)


@pytest.mark.parametrize('change, message', [
    (lambda data: data['backends'].append({**data['backends'][0], 'id': '../x'}), 'backend id'),
    (lambda data: data['backends'].append(dict(data['backends'][0])), 'listed twice'),
    (lambda data: data['backends'].append({**data['backends'][0], 'id': 'sycl', 'platforms': ['linux-arm64']}), 'platforms'),
    (lambda data: data['backends'].append({**data['backends'][0], 'id': 'sycl'}), 'cmake_switch'),
    (lambda data: data['backends'].append({**data['backends'][0], 'id': 'sycl', 'cmake_switch': 'HIP'}), 'cmake_switch'),
    (lambda data: entry(data, 'hip').update(optional='yes'), 'optional must be a boolean'),
    (lambda data: entry(data, 'hip').update(rpath='opt/rocm/lib'), 'rpath must be an absolute'),
    (lambda data: entry(data, 'hip').update(system_libraries='libamdhip64.so*'), 'system_libraries'),
    (lambda data: entry(data, 'hip')['detect'].update(fallbacks=['/opt/rocm/bin/rocminfo']), 'fallbacks'),
    (lambda data: entry(data, 'hip')['detect'].update(pattern='gfx[0-9'), 'detect.pattern'),
    (lambda data: entry(data, 'hip')['detect'].update(targets='AMDGPU_TARGETS'), 'detect.targets'),
    (lambda data: entry(data, 'hip')['detect'].update(runtime=['libamdhip64.so.7']), 'runtime: absolute paths'),
    (lambda data: [backend.update(optional=True) for backend in data['backends'] if 'darwin' in backend['platforms']], 'darwin ships only optional'),
    (lambda data: data['backends'][0].update(cmake_flags=['--diag-suppress=221']), 'cmake_flags'),
    (lambda data: data['backends'][0]['detect'].update(required='yes'), 'detect'),
    (lambda data: data['platforms']['win32'].update(library='native/riko-native.dll'), 'bare file name'),
    (lambda data: data['platforms']['win32'].update(library='..\\riko-native.dll'), 'bare file name'),
    (lambda data: data['platforms'].update(freebsd=data['platforms']['linux']), 'unknown platform'),
    (lambda data: data['platforms']['darwin'].pop('shared_libraries'), 'shared_libraries'),
    (lambda data: entry(data, 'vulkan').update(cmake_flags=['-DGGML_CPU_ALL_VARIANTS=ON']), 'GGML_CPU_ALL_VARIANTS needs'),
    # ggml names modules .so on macOS too, which build.py's *.dylib would leave out of the bundle
    (lambda data: entry(data, 'metal')['cmake_flags'].append('-DGGML_BACKEND_DL=ON'), r'darwin.shared_libraries must match metal'),
])
def test_manifest_validation_rejects_entries_that_would_misresolve(change, message):
    data = json.loads(native_backends.manifest_path().read_text(encoding='utf-8'))
    change(data)
    with pytest.raises(ValueError, match=message): native_backends.validate(data)


def test_frozen_backend_reads_its_bundled_copy_of_the_manifest(tmp_path, monkeypatch):
    assert native_backends.manifest_path() == Path(__file__).resolve().parents[1] / 'electron' / 'native_backends.json'
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
    assert native_backends.manifest_path() == tmp_path / native_backends.FROZEN_DIRECTORY / 'native_backends.json'


def test_macos_bundle_and_app_declare_the_same_oldest_macos():
    # The Metal library targets the macOS that Info.plist (LSMinimumSystemVersion) admits, not the build runner's.
    import yaml
    builder = yaml.safe_load((Path(__file__).resolve().parents[1] / 'electron/electron-builder.yml').read_text(encoding='utf-8'))
    assert backends_for('darwin') == ['metal'] and library_name('darwin') == 'libriko-native.dylib'
    assert f"-DCMAKE_OSX_DEPLOYMENT_TARGET={builder['mac']['minimumSystemVersion']}" in cmake_flags('metal')
    assert builder['mac']['target'] == [{'target': 'dmg', 'arch': ['arm64']}]
