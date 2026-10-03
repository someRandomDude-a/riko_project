"""Guard package boundaries and entry points after the app-core layout refactor."""
import ast
import importlib.util
from pathlib import Path

from process import app_core
from process.app_core.conversation.chat import ChatService
from process.app_core.conversation.messages import ChatMessage
from process.app_core.events.bus import event_bus
from process.app_core.runtime.session import SessionManager
from process.app_core.tools.builtin.scientific_calculator import Tool
from process.app_core.tools.registry import ToolRegistry


CORE = Path(app_core.__file__).parent


def test_feature_packages_keep_core_root_small_and_preserve_public_exports():
    assert {path.name for path in CORE.glob('*.py')} == {'__init__.py', 'factory.py'}
    packages = {'animation', 'audio', 'configuration', 'conversation', 'desktop',
        'emotion', 'events', 'inference', 'persistence', 'resources', 'runtime', 'tools'}
    assert all((CORE / name / '__init__.py').is_file() for name in packages)
    assert app_core.ChatService is ChatService
    assert app_core.ChatMessage is ChatMessage
    assert app_core.SessionManager is SessionManager
    assert app_core.event_bus is event_bus


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
