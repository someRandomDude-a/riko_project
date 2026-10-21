"""Backend entry point: standalone in development, Electron-owned in releases."""
from pathlib import Path
import os
import logging
import sys

# Imported lazily or dynamically, so a module or data file the frozen build misses would otherwise fail only on a
# user's machine (tools/release/build.py collects them).
RELEASE_MODULES = ('numpy', 'torch', 'transformers', 'sentence_transformers', 'faster_whisper', 'ctranslate2', 'silero_vad',
    'onnxruntime', 'sounddevice', 'soundfile', 'scipy.signal', 'ruamel.yaml', 'pypdf', 'openai', 'huggingface_hub', 'discord',
    'discord_bot', 'uvicorn')
# kernel/, which code imports partly inside functions (chat's metrics, the background budgets): the release check imports
# each by name, as it loads every name of the lazy process.app_core facade. tests/test_release_build.py keeps it equal to kernel/.
KERNEL_MODULES = tuple(f'process.app_core.kernel.{name}' for name in ('audio_config', 'background_budget', 'cancellation', 'code_paths', 'lifecycle',
    'messages', 'metrics', 'model_pins', 'output_filter', 'safe_paths', 'schema', 'streaming', 'torch_device', 'turns', 'validation', 'workers'))
# Installed only with an NVIDIA display driver: a bundle that loads it at load time fails everywhere else.
DRIVER_LIBRARIES = {'nvcuda.dll', 'libcuda.so', 'libcuda.so.1', 'libcuda.dylib'}


def is_driver(name):
    """The NVIDIA driver library under any name it loads by; Linux maps list libcuda.so.<driver version>."""
    import re
    return name.lower() == 'nvcuda.dll' or bool(re.fullmatch(r'libcuda\.(dylib|so(\.\d+)*)', name))


def server_class():
    import asyncio
    import uvicorn

    class Server(uvicorn.Server):
        """uvicorn, except that the active turn ends before requests drain: a reply still generating would hold its
        request for the whole drain, and Electron allows the entire shutdown 15 s (electron/release.cjs)."""
        def __init__(self, config, stop_turn):
            super().__init__(config)
            self.stop_turn = stop_turn

        async def shutdown(self, sockets=None):
            await asyncio.to_thread(self.stop_turn)
            await super().shutdown(sockets=sockets)
    return Server


def __getattr__(name):
    """uvicorn, Server and load_config load on first use and stay: frozen, every built-in tool call and the Discord client
    re-execute this binary (--tool-worker, --discord-worker), and main() dispatches those before any of them loads."""
    if name == 'uvicorn': import uvicorn as value
    elif name == 'Server': value = server_class()
    elif name == 'load_config': from process.app_core.configuration.config import load_config as value
    else: raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    globals()[name] = value
    return value


def loaded_libraries(names):
    """{name: path} for the shared libraries among `names` that this process has loaded."""
    import ctypes
    if os.name == 'nt':
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetModuleHandleW.argtypes, kernel32.GetModuleHandleW.restype = [wintypes.LPCWSTR], wintypes.HMODULE
        kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        kernel32.GetModuleFileNameW.restype = wintypes.DWORD
        found = {}
        for name in names:
            buffer = ctypes.create_unicode_buffer(32768)
            module = kernel32.GetModuleHandleW(name)
            if module and kernel32.GetModuleFileNameW(module, buffer, len(buffer)): found[name] = Path(buffer.value)
        return found
    if sys.platform == 'darwin':
        dyld = ctypes.CDLL(None)
        dyld._dyld_get_image_name.argtypes, dyld._dyld_get_image_name.restype = [ctypes.c_uint32], ctypes.c_char_p
        paths = [os.fsdecode(name) for name in map(dyld._dyld_get_image_name, range(dyld._dyld_image_count())) if name]
    else:
        with open('/proc/self/maps', encoding='utf-8', errors='surrogateescape') as maps:
            paths = [fields[5].rstrip('\n') for fields in (line.split(None, 5) for line in maps) if len(fields) == 6 and fields[5].startswith('/')]
    return {Path(path).name: Path(path) for path in paths if Path(path).name in names or is_driver(Path(path).name)}


def native_bundle_problems(library, loaded):
    """Why a loaded native bundle cannot ship: one of its libraries resolved from elsewhere (a toolkit, SDK or
    LD_LIBRARY_PATH that users lack was shadowing it), or the NVIDIA driver library loaded with it."""
    bundle = library.resolve().parent
    return [f'{name} was loaded from {path}' for name, path in sorted(loaded.items())
            if is_driver(name) or path.resolve().parent != bundle]


def native_device_problems(dll):
    """riko_devices, after riko_create initialised the backends: version 1 JSON listing ggml's devices, the CPU among them
    (printed with their memory). [] when it does; a bridge without the export is an older riko-native."""
    import ctypes
    import json
    try: devices = dll.riko_devices
    except AttributeError: return ['riko_devices missing: an older riko-native; rebuild it from this checkout (tools/llama_cpp/README.md)']
    devices.argtypes, devices.restype = [ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int
    buffer = ctypes.create_string_buffer(65536)
    length = devices(buffer, len(buffer))
    text = buffer.value.decode('utf-8', errors='replace')
    if not 0 <= length < len(buffer): return [f'riko_devices returned {length}: {text}']
    try:
        data = json.loads(text)
        listed = data['devices'] if isinstance(data, dict) and data.get('version') == 1 and isinstance(data.get('devices'), list) else None
        if listed is None or not all(isinstance(device, dict) for device in listed): raise ValueError('not a version 1 device list')
    except ValueError as exc: return [f'riko_devices returned no device list ({exc}): {text[:300]}']
    print('ggml devices: ' + ', '.join(f'{device.get("name")} ({device.get("description")}, {device.get("type")}, '
        f'{int(device.get("memory_total") or 0) // 1024 ** 2} MiB)' for device in listed), flush=True)
    return [] if any(device.get('type') == 'cpu' for device in listed) else ['riko_devices lists no CPU device']


def native_release_problems(library, modules=None, system=()):
    """Why a native bundle cannot ship ([] when it can), loading it as the app does: its six exports (the app treats
    riko_devices as optional, for older bridges, but a new bundle must have it); no library it links resolves from outside
    the bundle (a toolkit, SDK or LD_LIBRARY_PATH users lack) and no NVIDIA driver library loads with it. riko_create, given
    no arguments, initialises ggml's backends and fails before it reads a model; riko_devices then lists ggml's devices,
    which must include the CPU. With modules
    (native_backends.backend_modules: GGML_BACKEND_DL), ggml looks for them only beside the executable and in the working
    directory, so a CPU module loaded from the bundle proves that the bridge loads from its own folder; every module there
    must then load with its dependencies from the bundle too, except a dependency matching `system`
    (native_backends.system_libraries: hip's ROCm), which users install and the build machine may lack. Initialised, cudart
    may load the driver (on a machine that has one), so only bundle libraries count from then on."""
    import ctypes
    import fnmatch
    import re
    library = Path(library).resolve()
    names = {file.name for file in library.parent.iterdir()}
    directory = os.add_dll_directory(str(library.parent)) if os.name == 'nt' else None
    try:
        dll = ctypes.CDLL(str(library))
        for symbol in ('riko_create', 'riko_request', 'riko_set_interval', 'riko_stop', 'riko_destroy'): getattr(dll, symbol)
        problems = native_bundle_problems(library, loaded_libraries(names | DRIVER_LIBRARIES))
        dll.riko_create.argtypes, dll.riko_create.restype = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_void_p
        error = ctypes.create_string_buffer(8192)
        if dll.riko_create(b'[]', 0, error, len(error)): problems.append('riko_create accepted empty arguments')
        print(f'riko_create without arguments: {error.value.decode("utf-8", errors="replace")}', flush=True)
        problems += native_device_problems(dll)
        if modules:
            def module(name): return name == modules['gpu'] or any(fnmatch.fnmatch(name, pattern) for pattern in modules['cpu'])
            loaded = {name for name in loaded_libraries(names) if name in names}
            print(f'Backend modules riko_create loaded: {", ".join(sorted(filter(module, loaded))) or "none"}', flush=True)
            if not any(fnmatch.fnmatch(name, pattern) for name in loaded for pattern in modules['cpu']):
                problems.append(f'riko_create loaded no CPU backend module ({" or ".join(modules["cpu"])}) from {library.parent}')
            for name in sorted(filter(module, names - loaded)):  # those it found no device for or did not pick, by name
                try: ctypes.CDLL(str(library.parent / name))
                except OSError as failure:
                    lacking = re.match(r'([^:\s]+): cannot open shared object file', str(failure))  # glibc names the library it lacks
                    if lacking and any(fnmatch.fnmatch(lacking[1], pattern) for pattern in system):
                        print(f'{name} needs {lacking[1]} from the system, which this machine lacks (not bundled)', flush=True)
                    else: problems.append(f'{name} does not load: {failure}')
        problems += native_bundle_problems(library, {name: path for name, path in loaded_libraries(names).items() if name in names})
    finally:
        if directory: directory.close()
    return list(dict.fromkeys(problems))


def release_check(library=None):
    """`--release-check [library]`: can this build do what an installed app needs? tools/release/build.py runs it on the
    frozen backend once per native backend, without the build's toolkit paths; tests/test_release_build.py runs it on this
    tree without a library. It needs no GPU, audio device, credentials, network or model, and touches no user data."""
    import importlib
    import json
    import subprocess
    import tempfile
    import yaml
    from process.app_core.configuration.native_backends import backend_modules, backends_for, bundled_library, library_name, system_libraries
    shipped, name = backends_for(sys.platform), library_name(sys.platform)  # the frozen copy bundled:<backend> resolves with
    if library and (Path(library).name != name or Path(library).absolute().parent.name not in shipped):
        raise SystemExit(f'{library} is not where bundled:<backend> looks on {sys.platform}: native/<{"|".join(shipped)}>/{name}')
    if library:
        backend, library = Path(library).absolute().parent.name, Path(library).resolve()
        problems = native_release_problems(library, backend_modules(backend), system_libraries(backend))
        if problems: raise SystemExit('The native bundle cannot ship:\n  ' + '\n  '.join(problems))
    for module in RELEASE_MODULES + KERNEL_MODULES: importlib.import_module(module)
    core = importlib.import_module('process.app_core')
    for name in core.__all__: getattr(core, name)
    from eff_word_net.audio_processing import Resnet50_Arc_loss
    from silero_vad import load_silero_vad
    Resnet50_Arc_loss(); load_silero_vad()  # wake words and voice activity: both models are package data files
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as data:
        # The app over the defaults, in an empty data root, never a developer's files.
        os.environ.update(RIKO_DATA_DIR=data, RIKO_CONFIG=str(Path(data) / 'character_config.yaml'))
        os.chdir(data)
        try:
            import desktop_server  # the FastAPI app, every router and the whole app_core import graph
            from process.app_core.configuration.config import load_config
            desktop_server.create_app(load_config(recover='setup'))  # builds the routes and the Backend; starts nothing
            from process.app_core.tools.builtin.scientific_calculator import Tool
            from process.app_core.tools.registry import ToolRegistry
            registry = ToolRegistry(timeout_seconds=300)  # through the real worker: --tool-worker when frozen
            try:
                registry.register_local(Tool({}, {}))
                result = registry.execute(Tool.TOOL_NAME, {'expression': '2**10'})
            finally: registry.close()
            if result.is_error or result.content != '1024': raise SystemExit(f'A built-in tool failed in its worker: {result.content}')
            # As DiscordLauncher starts it, but --dry-run stops before the credentials check and the login.
            from process.app_core.kernel.code_paths import CodePaths
            worker = subprocess.run([*CodePaths.current().discord_command(), '--dry-run'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
            if worker.returncode or 'RIKO_DISCORD_DRY_RUN_OK' not in worker.stdout:
                raise SystemExit(f'The Discord worker failed its dry run (exit {worker.returncode}):\n{worker.stdout}{worker.stderr}')
            # The packaged first run (electron/release.cjs saveSetup): the setup page's choices on stdin, through the same
            # command line, answer a configuration that resolves bundled:<backend> into this bundle (a stand-in without one).
            if shipped:
                if library: bundle, backend = library.parents[2], library.parent.name
                else:
                    bundle, backend = Path(data) / 'bundle', shipped[0]
                    stand_in = bundled_library(backend, bundle); stand_in.parent.mkdir(parents=True); stand_in.touch()
                choices = {'backend': backend, 'repo': 'owner/model', 'filename': 'model.gguf', 'context': 8192, 'output': 1024, 'threads': 2, 'name': 'Riko'}
                setup = subprocess.run(CodePaths.current().server_command('--setup-config', data), input=json.dumps(choices), capture_output=True, text=True,
                    encoding='utf-8', errors='replace', env={**os.environ, 'RIKO_BUNDLE_ROOT': str(bundle)}, timeout=300)
                try: written = yaml.safe_load(json.loads(setup.stdout.splitlines()[-1])['config'])
                except (IndexError, KeyError, TypeError, ValueError, yaml.YAMLError): written = None
                if setup.returncode or not isinstance(written, dict) or written['runtime']['native_library'] != f'bundled:{backend}':
                    raise SystemExit(f'The first-run configuration failed (exit {setup.returncode}):\n{setup.stdout}{setup.stderr}')
        finally: os.chdir(previous)
    print('RIKO_RELEASE_CHECK_OK', flush=True)


def main():
    if '--release-check' in sys.argv:
        arguments = sys.argv[sys.argv.index('--release-check') + 1:]
        release_check(arguments[0] if arguments else None)
        return
    if '--tool-worker' in sys.argv:
        import runpy
        runpy.run_module('process.app_core.tools.worker', run_name='__main__')
        return
    if '--discord-worker' in sys.argv:
        import runpy
        runpy.run_module('discord_bot', run_name='__main__')
        return
    if '--setup-config' in sys.argv or '--validate-config' in sys.argv:
        # The packaged first run asks for its configuration (electron/release.cjs saveSetup); --validate-config checks a
        # file. One line of JSON out; no data root, port or model (configuration/first_run.py).
        from process.app_core.configuration.first_run import command
        raise SystemExit(command(sys.argv))
    this = sys.modules[__name__]  # uvicorn, Server and load_config resolve through __getattr__ (and tests replace them there)
    from process.app_core.configuration.paths import working_directory
    data = working_directory().absolute()  # RIKO_DATA_DIR, else the checkout
    # Explicit, and absolute, for load_config (earlier releases kept the to-do list in this cwd) and every child: tool
    # workers, MCP servers and the task MCP server find the same data root.
    os.environ['RIKO_DATA_DIR'] = str(data)
    os.chdir(data)
    config = this.load_config(recover='setup')  # the app's lifespan reports a broken section in setup mode
    from process.app_core.configuration.debug_logging import configure_logging
    configure_logging(config)
    # Debug our application, not WebSocket frames. Uvicorn DEBUG dumps every
    # voice.level/chat.delta packet and can dominate the capture/UI event loop.
    # Redirected stdout uses the ANSI code page on Windows (cp932 has no em dash); never let output stop startup.
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(errors='backslashreplace')
    from process.app_core.desktop.api_guard import backend_port
    try: port = backend_port()  # RIKO_PORT, else 8765: packaged Electron passes the free port its windows use
    except ValueError as exc: raise SystemExit(str(exc)) from None
    print(f"Riko AI server: http://127.0.0.1:{port} — Ctrl+C to stop", flush=True)
    import desktop_server
    app = desktop_server.create_app(config)  # this config, loaded once; the lifespan builds the services once uvicorn starts
    backend = app.state.backend
    # Hold the port before the model loads, so no other process (or account) can listen on it and
    # collect the API token Electron sends. In development this start's token is minted only now, so a
    # squatter could only have seen the previous one; packaged Electron sends its token only after the
    # line below. Requests queue until startup finishes.
    import socket
    import time
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE if os.name == 'nt' else socket.SO_REUSEADDR, 1)
    deadline = time.monotonic() + 20  # a previous backend may still be shutting down after a quick relaunch
    while True:
        try: listener.bind(('127.0.0.1', port)); break
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise SystemExit(f'Port {port} is already in use ({exc}); stop the other process or set RIKO_PORT, and try again.') from None
            time.sleep(.5)
    listener.listen(2048)
    from process.app_core.configuration.debug_logging import redact_handlers, redact_secrets
    redact_secrets(*(backend.api_secrets() or ()))  # this start's token and confirmation key never reach a log
    # electron/release.cjs listeningLine: main sends the token only once this names the port it gave us.
    if os.environ.get('RIKO_MANAGED') == '1': print(f'RIKO_BACKEND_LISTENING port={port}', flush=True)
    settings = this.uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info", timeout_graceful_shutdown=2)
    redact_handlers('uvicorn', 'uvicorn.error', 'uvicorn.access')  # the stderr handlers uvicorn installed as its Config was built
    server = this.Server(settings, backend.stop_turn)
    if os.environ.get('RIKO_MANAGED') == '1':
        import faulthandler
        import threading
        def managed_shutdown():
            for line in sys.stdin:
                if line.strip() == 'shutdown': break
            # Electron kills us 15 s after asking (electron/release.cjs). If teardown hangs, write every thread's
            # stack to the launch log and exit at 14 s instead of outliving the app with its port and the model.
            try: faulthandler.dump_traceback_later(14, exit=True)
            except Exception: pass  # no usable stderr
            server.should_exit = True
        threading.Thread(target=managed_shutdown, name='release-shutdown', daemon=True).start()
    try: server.run(sockets=[listener])
    except KeyboardInterrupt: pass  # uvicorn re-raises Ctrl+C after its graceful shutdown; uvicorn.run() swallows it too


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
