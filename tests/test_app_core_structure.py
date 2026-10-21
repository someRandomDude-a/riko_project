"""Guard package boundaries and entry points after the app-core layout refactor."""
import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from process import app_core
from process.app_core.conversation.chat import ChatService
from process.app_core.kernel.messages import ChatMessage
from process.app_core.events.bus import event_bus
from process.app_core.runtime.session import SessionManager
from process.app_core.tools.builtin.scientific_calculator import Tool
from process.app_core.tools.registry import ToolRegistry


CORE = Path(app_core.__file__).parent
# Old path -> (new path, names) for the modules moved into kernel/ (and read_gguf into inference/): the old modules
# re-export the same objects for one release, then they and this table go.
SHIMS = {
    'runtime.workers': ('kernel.workers', ['DaemonExecutor']),
    'runtime.cancellation': ('kernel.cancellation', ['TurnCancelled']),
    'runtime.lifecycle': ('kernel.lifecycle', ['close_bounded', 'run_bounded']),
    'runtime.torch_device': ('kernel.torch_device', ['preserve_torch_globals', 'resolve', 'validate']),
    'conversation.messages': ('kernel.messages', ['ChatMessage', 'ModelResponse', 'Role', 'ToolCall', 'ToolResult', 'conversation_sections']),
    'conversation.streaming': ('kernel.streaming', ['WordDeltas']),
    'conversation.output_filter': ('kernel.output_filter', ['OutputFilter', 'clean_output']),
    'inference.metrics': ('kernel.metrics', ['InferenceMetrics']),
    'inference.background_budget': ('kernel.background_budget', ['check_budget', 'validate_budget']),
    'inference.llama_context': ('kernel.cancellation', ['BackgroundPreempted']),
    'resources.vram_estimate': ('inference.gguf', ['read_gguf']),
}
HEAVY = {'numpy', 'torch', 'yaml', 'uvicorn', 'fastapi', 'pypdf', 'sounddevice', 'openai'}


def test_feature_packages_keep_core_root_small_and_preserve_public_exports():
    assert {path.name for path in CORE.glob('*.py')} == {'__init__.py', 'factory.py'}
    packages = {'animation', 'audio', 'configuration', 'conversation', 'desktop',
        'emotion', 'events', 'inference', 'kernel', 'persistence', 'resources', 'runtime', 'tools'}
    assert all((CORE / name / '__init__.py').is_file() for name in packages)
    assert app_core.ChatService is ChatService
    assert app_core.ChatMessage is ChatMessage
    assert app_core.SessionManager is SessionManager
    assert app_core.event_bus is event_bus


def test_the_public_facade_loads_each_name_on_first_use():
    assert set(app_core.__all__) <= set(dir(app_core))
    assert all(getattr(app_core, name) is not None for name in app_core.__all__)
    with pytest.raises(AttributeError): app_core.not_a_public_name
    from process.app_core import factory  # a submodule, not an export
    assert factory.create_chat_service


def test_moved_modules_keep_their_old_import_paths_for_one_release():
    for old, (new, names) in SHIMS.items():
        before, after = importlib.import_module(f'process.app_core.{old}'), importlib.import_module(f'process.app_core.{new}')
        assert all(getattr(before, name) is getattr(after, name) for name in names), old


def loaded_after(*modules):
    """The app_core modules and heavy dependencies a fresh interpreter holds after importing modules."""
    code = ('import importlib, json, sys\nfor name in sys.argv[1:]: importlib.import_module(name)\n'
        f"print(json.dumps([m for m in sys.modules if m.startswith('process.app_core') or m in {sorted(HEAVY)}]))")
    result = subprocess.run([sys.executable, '-c', code, *modules], capture_output=True, text=True, timeout=120,
        env={**os.environ, 'PYTHONPATH': str(CORE.parents[1])})
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout))


def test_leaf_imports_load_only_their_own_modules():
    # The task MCP server, each tool worker and kernel/ used to load ~40 app_core modules through the eager facade. The
    # facade holds only the settings registry (kernel/schema.py, standard library only), whose sections load on first read.
    kernel = {f'process.app_core.kernel.{path.stem}' for path in (CORE / 'kernel').glob('*.py') if path.stem != '__init__'}
    facade = {'process.app_core', 'process.app_core.kernel', 'process.app_core.kernel.schema'}
    assert loaded_after('process.app_core') == facade
    assert loaded_after(*kernel) == {*facade, *kernel}
    assert loaded_after('process.app_core.persistence.tasks') == {*facade, 'process.app_core.persistence', 'process.app_core.persistence.tasks'}
    tool = 'process.app_core.tools.builtin.scientific_calculator'
    assert loaded_after(tool) == {*facade, 'process.app_core.tools', 'process.app_core.tools.builtin', 'process.app_core.tools.builtin.base', tool}


def test_tool_registry_reports_to_the_activity_observer_it_is_given():
    from process.app_core.desktop.state import DesktopState
    from process.app_core.tools.registry import RegisteredTool, ToolActivity
    calls = []
    class Recorder(ToolActivity):
        def tool_started(self, name, arguments): calls.append(('started', name, arguments)); return 'activity-1'
        def tool_finished(self, name, result, error=False, activity_id=None): calls.append(('finished', name, result, error, activity_id))
    registry, silent = ToolRegistry(activity=Recorder()), ToolRegistry()
    try:
        registry.tools['echo'] = RegisteredTool('echo', 'Echo', {}, lambda args: args['text'])
        assert registry.execute('echo', {'text': 'hi'}, 'call-1').content == 'hi'
        assert calls == [('started', 'echo', {'text': 'hi'}), ('finished', 'echo', 'hi', False, 'activity-1')]
        assert type(silent.activity) is ToolActivity  # no desktop singleton unless the factory passes one
    finally: registry.close(); silent.close()
    assert all(callable(getattr(DesktopState, name, None)) for name in ('tool_started', 'tool_finished', 'notify'))


def test_all_local_imports_resolve_including_lazily_loaded_modules():
    for path in CORE.rglob('*.py'):
        parts = path.relative_to(CORE).with_suffix('').parts
        package = 'process.app_core'
        parents = parts[:-1]
        if parents: package += '.' + '.'.join(parents)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.ImportFrom): continue
            module = '.' * node.level + (node.module or '')
            resolved = importlib.util.resolve_name(module, package) if node.level else module
            if not resolved.startswith('process.app_core.'): continue
            relative = resolved.removeprefix('process.app_core.').replace('.', '/')
            assert (CORE / (relative + '.py')).is_file() or (CORE / relative / '__init__.py').is_file(), (path, node.lineno, resolved)


def test_relocated_builtin_tools_still_run_in_disposable_worker():
    registry = ToolRegistry(timeout_seconds=5)
    try:
        registry.register_local(Tool({}, {}))
        registered = registry.tools['scientific_calculator']
        assert registered.isolated['module'] == 'process.app_core.tools.builtin.scientific_calculator'
        result = registry.execute('scientific_calculator', {'expression': 'sqrt(144) + 1'}, 'layout-test')
        assert not result.is_error
        assert result.content == '13.0'
    finally: registry.close()
