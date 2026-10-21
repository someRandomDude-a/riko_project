"""The native-backend manifest, electron/native_backends.json: which bundled llama.cpp builds ship for which OS, their
library file names and their build flags. Electron's first-run setup (electron/native_backends.cjs), the
`runtime.native_library: bundled:<backend>` resolution here and tools/release/build.py all read that one file; the
frozen backend carries a copy (build.py adds it with --add-data). Every function takes the platform (sys.platform
values), so tests check each OS on every host. An optional backend (hip) resolves like the others but is left out of a
default release (release_backends). Dependency-free: build.py loads this file by path."""
from __future__ import annotations

import fnmatch
import functools
import json
import re
import sys
from pathlib import Path

NAME = 'native_backends.json'
FROZEN_DIRECTORY = 'electron'  # the manifest's folder inside the frozen backend, as in the repository
PLATFORMS = ('win32', 'linux', 'darwin')
_ID = re.compile(r'[a-z][a-z0-9_]*')  # also the bundle's folder name: nothing that could leave resources/native
_SWITCH = re.compile(r'GGML_[A-Z0-9_]+')
# In a backend's cmake_flags: ggml builds its backends as modules that the bridge loads from its own folder at run time.
DYNAMIC = '-DGGML_BACKEND_DL=ON'


def manifest_path() -> Path:
    frozen = getattr(sys, '_MEIPASS', None)
    return (Path(frozen) / FROZEN_DIRECTORY if frozen else Path(__file__).resolve().parents[4] / 'electron') / NAME


def validate(data: dict) -> dict:
    """The manifest, or ValueError naming the first entry that would mis-resolve a bundle or a build."""
    def fail(message): raise ValueError(f'{NAME}: {message}')
    platforms, backends = data.get('platforms'), data.get('backends')
    if not isinstance(platforms, dict) or not platforms or not isinstance(backends, list) or not backends:
        fail('needs a "platforms" object and a "backends" list')
    for platform, entry in platforms.items():
        if platform not in PLATFORMS: fail(f'unknown platform {platform!r}')
        library = entry.get('library') if isinstance(entry, dict) else None
        if not isinstance(library, str) or not library or Path(library).name != library or '\\' in library:
            fail(f'{platform}.library must be a bare file name')
        patterns = entry.get('shared_libraries')
        if not isinstance(patterns, list) or not patterns or not all(isinstance(item, str) and item for item in patterns):
            fail(f'{platform}.shared_libraries must list file patterns')
        if 'rpath' in entry and not isinstance(entry['rpath'], str): fail(f'{platform}.rpath must be a string')
    seen, switches = set(), set()
    for backend in backends:
        name = backend.get('id') if isinstance(backend, dict) else None
        if not isinstance(name, str) or not _ID.fullmatch(name): fail(f'backend id {name!r} must be lowercase letters, digits or _')
        if name in seen: fail(f'backend {name!r} is listed twice')
        seen.add(name)
        if not all(isinstance(backend.get(key), str) and backend[key] for key in ('label', 'description')):
            fail(f'{name} needs a label and a description')
        shipped = backend.get('platforms')
        if not isinstance(shipped, list) or not shipped or any(item not in platforms for item in shipped) or len(set(shipped)) != len(shipped):
            fail(f'{name}.platforms must list platforms defined above')
        switch = backend.get('cmake_switch')
        if switch is not None and (not isinstance(switch, str) or not _SWITCH.fullmatch(switch) or switch in switches):
            fail(f'{name}.cmake_switch must be a GGML_ option no other backend uses, or null for a CPU-only build')
        switches.add(switch)
        if type(backend.get('optional', False)) is not bool: fail(f'{name}.optional must be a boolean')
        if 'rpath' in backend and not (isinstance(backend['rpath'], str) and backend['rpath'].startswith('/')):
            fail(f'{name}.rpath must be an absolute directory searched after the platform\'s')
        system = backend.get('system_libraries', [])
        if not isinstance(system, list) or not all(isinstance(item, str) and item for item in system): fail(f'{name}.system_libraries must list file patterns')
        flags = backend.get('cmake_flags', [])
        if not isinstance(flags, list) or not all(isinstance(flag, str) and flag.startswith('-D') for flag in flags):
            fail(f'{name}.cmake_flags must be -D definitions')
        if '-DGGML_CPU_ALL_VARIANTS=ON' in flags and DYNAMIC not in flags: fail(f'{name}: GGML_CPU_ALL_VARIANTS needs {DYNAMIC}')
        for platform in shipped if DYNAMIC in flags else []:  # build.py stages a bundle's files by its platform's patterns
            module = _modules(platform, switch)['cpu'][0]
            if not any(fnmatch.fnmatch(module, pattern) for pattern in platforms[platform]['shared_libraries']):
                fail(f'{platform}.shared_libraries must match {name}\'s GGML_BACKEND_DL modules ({module})')
        detect = backend.get('detect')
        if detect is not None and not (isinstance(detect, dict) and all(isinstance(detect.get(key), str) and detect[key] for key in ('command', 'label', 'missing'))
                and isinstance(detect.get('args', []), list) and all(isinstance(item, str) for item in detect.get('args', []))
                and isinstance(detect.get('fallbacks', []), list) and all(isinstance(argv, list) and argv and all(isinstance(item, str) and item for item in argv)
                    for argv in detect.get('fallbacks', []))
                and isinstance(detect.get('pattern', ''), str) and type(detect.get('required', False)) is bool
                and isinstance(detect.get('targets', ''), str) and isinstance(detect.get('runtime', []), list)
                and all(isinstance(item, str) and item.startswith('/') for item in detect.get('runtime', []))):
            fail(f'{name}.detect needs a command, args, label, missing text and a boolean required (fallbacks: argv lists; pattern, targets: text; runtime: absolute paths)')
        if detect and detect.get('targets') and not any(flag.startswith(f"-D{detect['targets']}=") for flag in backend.get('cmake_flags', [])):
            fail(f"{name}.detect.targets names -D{detect['targets']}=, which its cmake_flags do not set")
        if detect and detect.get('pattern'):
            try: re.compile(detect['pattern'])
            except re.error as error: fail(f'{name}.detect.pattern: {error}')
    for platform in platforms:  # the release builds these, and setup falls back to them
        if backends_for(platform, data) and not release_backends(platform, data): fail(f'{platform} ships only optional backends')
    return data


@functools.cache
def manifest() -> dict:
    return validate(json.loads(manifest_path().read_text(encoding='utf-8')))


def _platform(platform):
    return sys.platform if platform is None else platform


def platform_build(platform: str | None = None, data: dict | None = None) -> dict:
    """{library, shared_libraries, rpath?} for a platform's bundles."""
    platform, platforms = _platform(platform), (data or manifest())['platforms']
    if platform not in platforms: raise ValueError(f'No native library is defined for platform {platform!r}')
    return platforms[platform]


def library_name(platform: str | None = None, data: dict | None = None) -> str:
    return platform_build(platform, data)['library']


def backends_for(platform: str | None = None, data: dict | None = None) -> list[str]:
    """Backend ids shipped for a platform, in preference order."""
    platform = _platform(platform)
    return [backend['id'] for backend in (data or manifest())['backends'] if platform in backend['platforms']]


def release_backends(platform: str | None = None, data: dict | None = None) -> list[str]:
    """What a release builds and checks for the platform by default: its backends but the optional ones (hip), which
    tools/release/build.py stages only when asked."""
    platform = _platform(platform)
    return [backend['id'] for backend in (data or manifest())['backends'] if platform in backend['platforms'] and not backend.get('optional')]


def bundled_library(backend: str, bundle: str | Path | None, platform: str | None = None, data: dict | None = None) -> Path:
    """`bundled:<backend>` as a path inside the installed app (RIKO_BUNDLE_ROOT), so the configuration survives moving
    or upgrading the app; only backends shipped for this platform, whose ids are safe folder names, resolve."""
    platform = _platform(platform)
    shipped = backends_for(platform, data)
    if not shipped: raise ValueError(f'Bundled native library: no bundled backend ships for {platform}')
    if backend not in shipped:
        raise ValueError(f'Bundled native library {backend!r} is not shipped for {platform}; use '
            + ' or '.join(f'bundled:{name}' for name in shipped))
    if not bundle: raise ValueError('Bundled native library requires the packaged app (RIKO_BUNDLE_ROOT is not set)')
    return Path(bundle) / 'native' / backend / library_name(platform, data)


def _entry(backend, data):
    entry = next((item for item in data['backends'] if item['id'] == backend), None)
    if entry is None: raise ValueError(f'Unknown native backend {backend!r}')
    return entry


def rpath(backend: str, platform: str | None = None, data: dict | None = None) -> str | None:
    """The bundle's install rpath as a CMake list: the platform's (each library finds the others beside it), then the
    backend's own system directory (hip: /opt/rocm/lib, where AMD's ROCm packages put libamdhip64). None on Windows."""
    data = data or manifest()
    own = platform_build(platform, data).get('rpath')
    return ';'.join(filter(None, [own, _entry(backend, data).get('rpath')])) if own else None


def system_libraries(backend: str, data: dict | None = None) -> list[str]:
    """Patterns of the libraries a bundle loads from the user's system rather than carrying (hip: ROCm's HIP runtime,
    hipBLAS and rocBLAS); the release check accepts that they are missing on the build machine."""
    return list(_entry(backend, data or manifest()).get('system_libraries', []))


def cmake_flags(backend: str, data: dict | None = None) -> list[str]:
    """The backend's own switch ON, every other backend's OFF (macOS turns Metal on by default), then its flags."""
    data = data or manifest()
    entry = _entry(backend, data)
    switches = [item['cmake_switch'] for item in data['backends'] if item.get('cmake_switch')]
    return [f'-D{switch}={"ON" if switch == entry.get("cmake_switch") else "OFF"}' for switch in switches] + list(entry.get('cmake_flags', []))


def _modules(platform, switch):
    prefix, extension = ('ggml-', '.dll') if platform == 'win32' else ('libggml-', '.so')  # ggml names modules .so on macOS too
    return {'gpu': f'{prefix}{switch[5:].lower()}{extension}' if switch else None, 'cpu': [f'{prefix}cpu{extension}', f'{prefix}cpu-*{extension}']}


def backend_modules(backend: str, platform: str | None = None, data: dict | None = None) -> dict | None:
    """For a bundle built with GGML_BACKEND_DL, the module files ggml_backend_load_all_from_path loads from it, by name:
    {'gpu': 'libggml-cuda.so' (from the cmake_switch; None for a CPU bundle), 'cpu': ['libggml-cpu.so', 'libggml-cpu-*.so']}
    (ggml-*.dll on Windows), the CPU patterns covering GGML_CPU_ALL_VARIANTS' per-ISA builds. None when the backends are linked."""
    entry = _entry(backend, data or manifest())
    return _modules(_platform(platform), entry.get('cmake_switch')) if DYNAMIC in entry.get('cmake_flags', []) else None
