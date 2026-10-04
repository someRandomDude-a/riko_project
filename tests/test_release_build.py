import importlib.util
from pathlib import Path, PureWindowsPath
import subprocess


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
