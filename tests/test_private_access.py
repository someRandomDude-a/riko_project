"""Ratchet on reaching into another object's private state; later changes only shrink PRIVATE.

Scans app_core and the Code/*.py entry points (not tests) for reads and writes of single-underscore attributes on anything
but self, cls or super(), such as initiative.py taking session._voice_lock, written directly or through getattr, setattr,
hasattr or delattr with a literal name. Not counted: dunders, a def, class or class/module-level name the same module
defines, and attributes of modules imported from outside Riko (os._exit). Another instance of the module's own class is
counted too (keyed on names, an unrelated self._lock would otherwise hide a session._lock read). PRIVATE pins each (file, attribute) count:
more fails until the access is justified here, fewer fails until the count is lowered.
"""
import ast
from collections import Counter
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1] / 'Code'
RIKO = {'process', 'desktop_server', 'discord_bot', 'run_server', 'task_mcp_server'}
REFLECTION = {'getattr', 'setattr', 'hasattr', 'delattr'}  # getattr(session, '_closed', False) is the same coupling

PRIVATE = {
    ('process/app_core/emotion/compat.py', '_julia_original_forward'): 1, ('process/app_core/emotion/compat.py', '_update_attention_mask'): 1,  # a transformers encoder it patches
    ('process/app_core/runtime/warmup.py', '_backend'): 1, ('process/app_core/runtime/warmup.py', '_embed'): 1, ('process/app_core/runtime/warmup.py', '_interpret'): 1,
    ('run_server.py', '_dyld_get_image_name'): 3, ('run_server.py', '_dyld_image_count'): 1,  # dyld's own C functions via ctypes
}


def own_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): names.add(node.name)
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for statement in node.body:
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, (ast.AnnAssign, ast.AugAssign)) else []
                names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def external_modules(tree):
    """Names bound by imports from outside Riko: their private names are their own business."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split('.')[0] for alias in node.names if alias.name.split('.')[0] not in RIKO)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module.split('.')[0] not in RIKO:
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def private_access(paths):
    counts = Counter()
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        own, external = own_names(tree), external_modules(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute): name, receiver = node.attr, node.value
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in REFLECTION and len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)): name, receiver = node.args[1].value, node.args[0]
            else: continue
            if not name.startswith('_') or (name.startswith('__') and name.endswith('__')) or name in own: continue
            if isinstance(receiver, ast.Name) and (receiver.id in ('self', 'cls') or receiver.id in external): continue
            if isinstance(receiver, ast.Call) and ast.unparse(receiver.func) == 'super': continue
            counts[(path.relative_to(CODE).as_posix(), name)] += 1
    return counts


def test_private_attribute_access_across_modules_matches_the_allowlist():
    found = private_access([*sorted((CODE / 'process' / 'app_core').rglob('*.py')), *sorted(CODE.glob('*.py'))])
    grown = [f'{file}: .{name} x{count} (allowed {PRIVATE.get((file, name), 0)})' for (file, name), count in sorted(found.items()) if count > PRIVATE.get((file, name), 0)]
    stale = [f'{file}: .{name} x{found.get((file, name), 0)} (allowed {count})' for (file, name), count in sorted(PRIVATE.items()) if found.get((file, name), 0) < count]
    assert not grown, 'New access to another object\'s private attribute; add a public method or property instead:\n' + '\n'.join(grown)
    assert not stale, 'Fewer private accesses than PRIVATE allows; lower or delete these entries:\n' + '\n'.join(stale)


def test_the_scan_skips_own_state_dunders_and_external_modules(tmp_path, monkeypatch):
    path = tmp_path / 'sample.py'
    path.write_text('import os\nclass Box:\n    _shared = 1\n    def __init__(self, other): self._value = other._value; self._lock = None; self.__dict__\n'
        '    def peek(self, session): return session._voice_lock, super()._hidden, os._exit, Box._shared, getattr(session, "_voice_lock")\n'
        '    def poke(self, session): setattr(session, "_closed", True); return hasattr(self, "_x"), getattr(os, "_exit"), session._lock\n')
    monkeypatch.setattr(sys.modules[__name__], 'CODE', tmp_path)
    # getattr/setattr count like attribute access; this module's own self._lock does not hide session._lock
    assert private_access([path]) == {('sample.py', '_voice_lock'): 2, ('sample.py', '_value'): 1, ('sample.py', '_closed'): 1, ('sample.py', '_lock'): 1}
