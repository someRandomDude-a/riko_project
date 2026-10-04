import importlib.util
from pathlib import Path, PureWindowsPath


def test_cmake_source_path_uses_forward_slashes_on_windows():
    source = Path(__file__).resolve().parents[1] / 'tools/release/build.py'
    spec = importlib.util.spec_from_file_location('release_build', source)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    argument = build.cmake_file_definition('RIKO_NATIVE_BRIDGE_SOURCE',
        PureWindowsPath(r'D:\a\riko_project\riko_project\tools\llama_cpp\riko-native.cpp'))
    assert argument == '-DRIKO_NATIVE_BRIDGE_SOURCE:FILEPATH=D:/a/riko_project/riko_project/tools/llama_cpp/riko-native.cpp'
    assert '\\' not in argument
