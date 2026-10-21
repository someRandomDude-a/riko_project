"""Ratchet on app_core's package graph; later changes only shrink EDGES.

Nodes are app_core's subpackages (integrations/discord counts as integrations). The root's __init__.py (the lazy public
facade) and factory.py (the composition root) may import anything, and no subpackage may import them or a Code/*.py entry
point. An edge is 'load' when a module-level import makes it, so it runs whenever the importing module loads, and 'lazy'
when only imports inside functions do; TYPE_CHECKING imports never run and do not count. Lazy edges are coupling too:
EDGES must equal the scanned graph, so a new edge, or a lazy edge becoming a load-time one, fails until it is added here
on purpose, and an edge that disappears (or turns lazy) fails until EDGES says so. kernel/ imports nothing else from
app_core, and the whole graph, lazy edges included, has no cycles.
"""
import ast
import importlib.util
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / 'Code' / 'process' / 'app_core'
ROOT_MODULES = {'__init__', 'factory'}
ENTRY_POINTS = {'desktop_server', 'discord_bot', 'run_server', 'task_mcp_server'}

EDGES = {
    'animation': {'events': 'load', 'kernel': 'load', 'persistence': 'load'},
    'audio': {'desktop': 'load', 'events': 'load', 'kernel': 'load', 'persistence': 'load'},
    # Features register their settings sections (kernel/schema.py); app_core/__init__.py, not configuration, imports them.
    'configuration': {'kernel': 'load', 'persistence': 'load'},
    'conversation': {'emotion': 'lazy', 'kernel': 'load', 'persistence': 'load'},
    # Media, avatar and import paths are checked as text before anything resolves them (kernel/safe_paths.py).
    'desktop': {'kernel': 'load'},
    'emotion': {'kernel': 'load'},
    # ResourceEvents computes resource.<topic> on a kernel DaemonExecutor, so a publisher's thread never runs a getter.
    'events': {'kernel': 'load'},
    # The HTTP layer (wave 5: create_app and one router per URL domain, formerly Code/desktop_server.py, an entry point the
    # scan skipped) serves every feature; no package imports it.
    'http': {'configuration': 'load', 'desktop': 'load', 'emotion': 'lazy', 'events': 'load', 'integrations': 'load', 'kernel': 'load',
        'persistence': 'load', 'resources': 'load', 'runtime': 'load'},
    'inference': {'events': 'lazy', 'kernel': 'load'},
    'integrations': {'audio': 'load', 'desktop': 'lazy', 'events': 'load', 'kernel': 'load', 'persistence': 'load'},
    'persistence': {'emotion': 'lazy', 'events': 'lazy', 'kernel': 'load'},
    # The estimate resolves a torch model's device (kernel/torch_device.py) only when it computes one.
    'resources': {'audio': 'lazy', 'events': 'lazy', 'inference': 'load', 'kernel': 'lazy'},
    'runtime': {'animation': 'lazy', 'audio': 'load', 'events': 'load', 'kernel': 'load', 'persistence': 'load', 'tools': 'lazy'},
    'tools': {'events': 'load', 'kernel': 'load', 'persistence': 'load'},  # the factory registers the desktop tools
}


def imports(path):
    """(module, names, inside a function, line) for each import that can run in path."""
    parts = path.relative_to(CORE).with_suffix('').parts
    package, found = '.'.join(('process', 'app_core', *parts[:-1])), []
    def visit(nodes, lazy):
        for node in nodes:
            if isinstance(node, ast.If) and ast.unparse(node.test) in ('TYPE_CHECKING', 'typing.TYPE_CHECKING'):
                visit(node.orelse, lazy)
                continue
            if isinstance(node, ast.Import): found.extend((alias.name, [], lazy, node.lineno) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = importlib.util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
                found.append((module, [alias.name for alias in node.names], lazy, node.lineno))
            visit(ast.iter_child_nodes(node), lazy or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))
    visit(ast.parse(path.read_text(encoding='utf-8')).body, False)
    return found


def reached(module, name):
    """The app_core package an import reaches, 'root' for the facade or factory, or None outside app_core."""
    if module == 'process.app_core': return name if name and (CORE / name / '__init__.py').is_file() else 'root'
    if module == 'process' and name in (None, 'app_core'): return 'root'  # `import process`, `from process import app_core`
    if not module.startswith('process.app_core.'): return None
    package = module.split('.')[2]
    return 'root' if package in ROOT_MODULES else package


def scan():
    """({source: {target: 'load' | 'lazy'}}, [imports no subpackage may make])."""
    edges, problems = {}, []
    for path in sorted(CORE.rglob('*.py')):
        parts = path.relative_to(CORE).parts
        if len(parts) == 1: continue  # the root's __init__.py and factory.py
        for module, names, lazy, line in imports(path):
            where = f'{path.relative_to(CORE).as_posix()}:{line}'
            if module.split('.')[0] in ENTRY_POINTS: problems.append(f'{where} imports the entry point {module}')
            for name in names or [None]:
                target = reached(module, name)
                if target == 'root': problems.append(f'{where} imports the app_core facade or factory; import the defining module')
                elif target and target != parts[0]:
                    kinds = edges.setdefault(parts[0], {})
                    kinds[target] = 'load' if not lazy or kinds.get(target) == 'load' else 'lazy'
    return edges, problems


def cycles(edges):
    """Strongly connected components with more than one package (Tarjan)."""
    index, low, stack, found = {}, {}, [], []
    def visit(node):
        index[node] = low[node] = len(index)
        stack.append(node)
        for target in edges.get(node, {}):
            if target not in index:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in stack: low[node] = min(low[node], index[target])
        if low[node] == index[node]:
            component = []
            while not component or component[-1] != node: component.append(stack.pop())
            if len(component) > 1: found.append(sorted(component))
    for node in sorted({*edges, *(target for targets in edges.values() for target in targets)}):
        if node not in index: visit(node)
    return found


def test_kernel_imports_nothing_else_from_app_core_and_no_subpackage_imports_the_root_or_an_entry_point():
    edges, problems = scan()
    assert (CORE / 'kernel' / '__init__.py').is_file() and 'kernel' not in edges, edges.get('kernel')
    assert not problems, '\n'.join(problems)


def test_package_graph_has_no_cycles_even_through_lazy_imports():
    assert cycles(scan()[0]) == []
    assert cycles({'a': {'b': 'lazy'}, 'b': {'c': 'load'}, 'c': {'a': 'lazy'}, 'd': {'a': 'load'}}) == [['a', 'b', 'c']]


def test_cross_package_imports_match_the_allowlist():
    edges = scan()[0]
    found = {(source, target, kind) for source, targets in edges.items() for target, kind in targets.items()}
    allowed = {(source, target, kind) for source, targets in EDGES.items() for target, kind in targets.items()}
    new = [f'{source} -> {target} ({kind})' for source, target, kind in sorted(found - allowed)]
    stale = [f'{source} -> {target} ({kind})' for source, target, kind in sorted(allowed - found)]
    assert not new, 'New coupling between app_core packages; if it is intended, add it to EDGES:\n' + '\n'.join(new)
    assert not stale, 'These edges are gone (or changed kind); update EDGES to match:\n' + '\n'.join(stale)


def test_the_scan_sees_load_lazy_type_checking_and_root_imports(tmp_path, monkeypatch):
    package = tmp_path / 'app_core'
    for name in ('kernel', 'events', 'audio'): (package / name).mkdir(parents=True); (package / name / '__init__.py').write_text('')
    (package / 'audio' / 'speech.py').write_text('from typing import TYPE_CHECKING\nfrom ..kernel.workers import DaemonExecutor\n'
        'if TYPE_CHECKING:\n    from ..events.bus import EventBus\nelse:\n    EventBus = None\n'
        'def later():\n    from ..events import bus\n    from .. import factory\n    from process import app_core\nimport process\n')
    monkeypatch.setattr(sys.modules[__name__], 'CORE', package)
    edges, problems = scan()
    assert edges == {'audio': {'kernel': 'load', 'events': 'lazy'}}
    assert [problem.split(' ')[0] for problem in problems] == ['audio/speech.py:9', 'audio/speech.py:10', 'audio/speech.py:11']  # every way to the facade
