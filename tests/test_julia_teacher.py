"""Julia-1's code comes from its Hugging Face repository (emotion/teacher.py): it is downloaded at a pinned commit, checked
file by file before it runs, imported under a private name without touching sys.path or an installed `julia`, and one
model is shared by the emotion engine and memory's System 1 when they ask for the same."""
import gc
import hashlib
import logging
import re
import sys
import threading
import time
import types

import pytest

from process.app_core.emotion import JuliaEmotionEngine, teacher
from process.app_core.kernel.model_pins import COMMIT, JULIA_CODE_SHA256, JULIA_MODEL_ID, JULIA_REVISION, commit
from process.app_core.persistence.memory import JuliaMemoryDecider

SNAPSHOT = teacher.snapshot  # conftest's autouse fixture replaces it in every test; this is the real one
LOADS = []  # every load_model call a toy snapshot makes: (snapshot, device, options)


@pytest.fixture
def fresh(monkeypatch):
    """Empty teacher tables, and no riko_julia_* module left behind."""
    for name, value in (('_RUNTIMES', {}), ('_MODELS', __import__('weakref').WeakValueDictionary()), ('_DEVICES', {}), ('_LOADING', {}), ('_PACKAGES', {})):
        monkeypatch.setattr(teacher, name, value)
    LOADS.clear()
    yield
    for name in [name for name in sys.modules if name.startswith('riko_julia_')]: sys.modules.pop(name)


def toy(root, *, fail=False, delay=0):
    """A snapshot folder whose julia package imports relatively, like Julia-1's (julia/__init__.py: from .inference import load_model)."""
    package = root / 'julia'
    (package / 'router').mkdir(parents=True)
    (package / '__init__.py').write_text('from .inference import load_model\n' + ('raise ImportError("broken upstream")\n' if fail else ''))
    (package / 'router' / '__init__.py').write_text('')
    (package / 'router' / 'engine.py').write_text('class Engine:\n    def __init__(self, source, device, options): self.source, self.device, self.options = source, device, options\n')
    (package / 'inference.py').write_text(
        'import time\nfrom .router.engine import Engine\n'
        'def load_model(source, device="cpu", **options):\n'
        f'    import test_julia_teacher; test_julia_teacher.LOADS.append((source, device, options)); time.sleep({delay})\n'
        '    return Engine(source, device, options)\n')
    (package / '__pycache__').mkdir()
    (package / '__pycache__' / 'stale.cpython-311.py').write_text('ignored')
    return root


def test_a_snapshot_loads_under_its_own_name_and_leaves_sys_path_and_an_installed_julia_alone(tmp_path, monkeypatch, fresh):
    installed = types.ModuleType('julia')  # PyJulia, say
    monkeypatch.setitem(sys.modules, 'julia', installed)
    search = list(sys.path)
    first = teacher.runtime(toy(tmp_path / 'one'))
    assert re.fullmatch(r'riko_julia_[0-9a-f]{16}', first.__name__) and sys.modules[first.__name__] is first
    assert sys.modules['julia'] is installed and sys.path == search and str(tmp_path / 'one') not in sys.path
    assert first.load_model is sys.modules[first.__name__ + '.inference'].load_model  # relative imports resolved through it
    assert teacher.runtime(tmp_path / 'one') is first  # once per folder
    second = teacher.runtime(toy(tmp_path / 'two'))
    assert second is not first and second.__name__ != first.__name__ and second.load_model is not first.load_model


def test_a_failing_import_leaves_no_partial_modules_and_is_tried_again(tmp_path, fresh):
    source = toy(tmp_path / 'broken', fail=True)
    for _ in range(2):
        with pytest.raises(ImportError, match='broken upstream'): teacher.runtime(source)
        assert not [name for name in sys.modules if name.startswith('riko_julia_')]


def test_one_model_per_snapshot_device_and_options_loaded_once_even_when_both_users_start_together(tmp_path, fresh):
    source = toy(tmp_path / 'snap', delay=.2)
    options = dict(strict_encoding=True, max_length=8192, head_length=512)
    results = []
    threads = [threading.Thread(target=lambda: results.append(teacher.shared_model(str(source), 'cpu', **options))) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(10)
    assert len(LOADS) == 1 and results[0][0] is results[1][0] and results[0][1] == 'cpu'
    assert LOADS[0] == (str(source), 'cpu', options)
    assert teacher.shared_model(source, 'mps', **options)[0] is results[0][0] and len(LOADS) == 1  # Julia maps MPS to the CPU
    other = teacher.shared_model(source, 'cpu', **{**options, 'head_length': 7168})[0]
    assert other is not results[0][0] and len(LOADS) == 2
    del results, other
    gc.collect()  # every user let go (close()): the next asks load again rather than keeping 600 MB alive
    teacher.shared_model(source, 'cpu', **options)
    assert len(LOADS) == 3


def test_snapshot_downloads_the_pinned_commit_and_refuses_a_branch_before_any_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(
        snapshot_download=lambda model_id, **options: calls.append((model_id, options)) or str(tmp_path / 'snapshots' / options['revision'])))
    assert SNAPSHOT() == str(tmp_path / 'snapshots' / JULIA_REVISION)
    assert calls == [(JULIA_MODEL_ID, {'revision': JULIA_REVISION, 'cache_dir': None})]
    SNAPSHOT('someone/else', 'f' * 40, tmp_path / 'cache')
    assert calls[-1] == ('someone/else', {'revision': 'f' * 40, 'cache_dir': str(tmp_path / 'cache')})
    for revision in ('main', 'v1.0', JULIA_REVISION[:7], JULIA_REVISION.upper(), None):
        with pytest.raises(ValueError, match='full 40-character commit hash'): SNAPSHOT(JULIA_MODEL_ID, revision)
    assert len(calls) == 2


def test_the_pin_is_a_commit_with_a_manifest_of_julias_code():
    assert COMMIT.fullmatch(JULIA_REVISION) and commit('x', JULIA_REVISION) == JULIA_REVISION
    assert len(JULIA_CODE_SHA256) == 18 and all(name.startswith('julia/') and name.endswith('.py') for name in JULIA_CODE_SHA256)
    assert {'julia/__init__.py', 'julia/inference.py', 'julia/router/engine.py'} <= set(JULIA_CODE_SHA256)
    assert all(re.fullmatch(r'[0-9a-f]{64}', digest) for digest in JULIA_CODE_SHA256.values())


def test_verify_runs_only_the_pinned_code_and_names_what_differs(tmp_path, monkeypatch, caplog):
    source = toy(tmp_path / 'snap')
    manifest = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source / 'julia').rglob('*.py') if '__pycache__' not in path.parts}
    monkeypatch.setattr(teacher, 'JULIA_CODE_SHA256', manifest)
    assert teacher.verify(source, JULIA_MODEL_ID, JULIA_REVISION) is manifest  # what the import is confined to (bytecode never runs)
    (source / 'julia' / 'inference.py').write_text('import os\nos.system("anything")\n')
    (source / 'julia' / 'extra.py').write_text('')
    (source / 'julia' / 'router' / 'engine.py').unlink()
    with pytest.raises(RuntimeError, match=r'missing: julia/router/engine\.py; unexpected: julia/extra\.py; changed: julia/inference\.py') as error:
        teacher.verify(source, JULIA_MODEL_ID, JULIA_REVISION)
    assert JULIA_REVISION in str(error.value)
    with caplog.at_level(logging.WARNING, logger=teacher.__name__):
        assert teacher.verify(source, JULIA_MODEL_ID, 'e' * 40) is None  # the user chose another commit: it runs, unverified
    assert 'runs unverified' in caplog.text


def test_the_engine_and_the_decider_load_the_pinned_checked_snapshot_through_the_shared_model(tmp_path, monkeypatch):
    calls = []
    model, checked = types.SimpleNamespace(predict=lambda **kwargs: {}), {'julia/__init__.py': '0' * 64}
    monkeypatch.setattr(teacher, 'snapshot', lambda *args: calls.append(('snapshot', args)) or str(tmp_path))
    monkeypatch.setattr(teacher, 'verify', lambda *args: calls.append(('verify', args)) or checked)
    monkeypatch.setattr(teacher, 'shared_model', lambda source, device, manifest=None, **options: calls.append(('load', source, device, manifest, options)) or (model, 'cpu'))
    engine = JuliaEmotionEngine(None, revision='a' * 40, cache_dir=tmp_path / 'cache', max_length=4096, context_tokens=1024)
    assert engine.ensure_loaded() is model and engine.resolved_source == str(tmp_path) and engine.revision == 'a' * 40
    assert engine.load_options == {'strict_encoding': True, 'max_length': 4096, 'head_length': 3072}
    # The code is imported against the manifest verify() checked (teacher.runtime), never re-read loosely.
    assert calls == [('snapshot', (JULIA_MODEL_ID, 'a' * 40, tmp_path / 'cache')), ('verify', (str(tmp_path), JULIA_MODEL_ID, 'a' * 40)),
        ('load', str(tmp_path), 'cpu', checked, engine.load_options)]
    calls.clear()
    decider = JuliaMemoryDecider(JULIA_MODEL_ID, None, 2048)
    decider.decide('I like tea.')
    assert calls == [('snapshot', (JULIA_MODEL_ID, JULIA_REVISION, None)), ('verify', (str(tmp_path), JULIA_MODEL_ID, JULIA_REVISION)),
        ('load', str(tmp_path), 'cpu', checked, {'strict_encoding': True, 'max_length': 2048, 'head_length': 512})]
    calls.clear()
    JuliaMemoryDecider(JULIA_MODEL_ID, options=engine.load_options).decide('x')
    assert calls[-1] == ('load', str(tmp_path), 'cpu', checked, engine.load_options)
    calls.clear()
    folder = tmp_path / 'mine'
    folder.mkdir()
    JuliaEmotionEngine(folder).ensure_loaded()  # emotion.model_path: the user's own folder, not downloaded or checked
    assert calls == [('load', str(folder), 'cpu', None, JuliaEmotionEngine(folder).load_options)]


def test_a_refused_snapshot_falls_back_and_never_imports_its_code(tmp_path, monkeypatch, fresh):
    source = toy(tmp_path / 'snap')
    monkeypatch.setattr(teacher, 'snapshot', lambda *args: str(source))
    engine = JuliaEmotionEngine(None)  # the real manifest: the toy's code is not Julia-1's
    assert engine.ensure_loaded() is None and not LOADS and not [name for name in sys.modules if name.startswith('riko_julia_')]
    assert engine.observe_input('This is great!', final=True).source == 'julia_1_fallback'
    assert JuliaMemoryDecider(JULIA_MODEL_ID).decide('x')['confidence'] == .2 and not LOADS


def pinned(monkeypatch, source):
    """Make the toy at source the pinned commit's code, as verify() would accept it."""
    manifest = teacher.code_hashes(source)
    monkeypatch.setattr(teacher, 'JULIA_CODE_SHA256', manifest)
    return teacher.verify(source, JULIA_MODEL_ID, JULIA_REVISION)


def plant(folder, name, code):
    """Bytecode of code at folder/name, valid however its source changes (UNCHECKED_HASH: the default loader never looks)."""
    import py_compile
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / (name + '.src.py')
    source.write_text(code)
    py_compile.compile(str(source), cfile=str(folder / name), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH, doraise=True)
    source.unlink()


def test_only_the_checked_source_runs_never_planted_bytecode_a_sourceless_package_or_an_unlisted_module(tmp_path, monkeypatch, fresh):
    source = toy(tmp_path / 'snap')
    (source / 'julia' / 'inference.py').write_text((source / 'julia' / 'inference.py').read_text() + 'WHO = "checked"\n')
    (source / 'julia' / '__init__.py').write_text('from .inference import load_model, WHO\n'
        'try:\n    from . import extra\nexcept ImportError:\n    extra = None\n')
    manifest = pinned(monkeypatch, source)
    tag = sys.implementation.cache_tag
    # What the default import system would run first: a cached .pyc beside the source, a sourceless package folder of the
    # same name, and a module added after the check.
    plant(source / 'julia' / '__pycache__', f'inference.{tag}.pyc', 'WHO = "planted bytecode"\ndef load_model(*a, **k): pass\n')
    plant(source / 'julia' / 'router' / '__pycache__', f'engine.{tag}.pyc', 'class Engine: WHO = "planted engine"\n')
    plant(source / 'julia' / 'inference', '__init__.pyc', 'WHO = "planted package"\ndef load_model(*a, **k): pass\n')
    (source / 'julia' / 'extra.py').write_text('WHO = "unlisted"\n')
    before = sorted(path.relative_to(source).as_posix() for path in source.rglob('*'))
    module = teacher.runtime(source, manifest)
    assert module.WHO == 'checked' and module.extra is None and not hasattr(sys.modules[module.__name__ + '.router.engine'].Engine, 'WHO')
    assert module.__file__ == str(source.resolve() / 'julia' / '__init__.py') and module.__path__ == [str(source.resolve() / 'julia')]
    assert sorted(path.relative_to(source).as_posix() for path in source.rglob('*')) == before  # no bytecode written either
    with pytest.raises(ModuleNotFoundError, match='checked Julia 1 code'): __import__(module.__name__ + '.cuda')


def test_a_file_changed_after_the_check_is_refused_when_imported(tmp_path, monkeypatch, fresh):
    source = toy(tmp_path / 'snap')
    manifest = pinned(monkeypatch, source)
    (source / 'julia' / 'router' / 'engine.py').write_text('import os\nclass Engine: pass\n')  # swapped between verify() and the import
    with pytest.raises(ImportError, match=r'engine\.py changed after Julia 1 was checked'): teacher.runtime(source, manifest)
    assert not [name for name in sys.modules if name.startswith('riko_julia_')]


def test_an_unchecked_folder_runs_its_own_source_as_hashed_at_import(tmp_path, fresh):
    source = toy(tmp_path / 'mine')  # emotion.model_path, or a commit the user chose: no manifest, but still no bytecode
    tag = sys.implementation.cache_tag
    plant(source / 'julia' / '__pycache__', f'inference.{tag}.pyc', 'def load_model(*a, **k): return "planted"\n')
    assert teacher.shared_model(source, 'cpu')[0].__class__.__name__ == 'Engine' and LOADS


def test_an_offline_snapshot_of_the_pinned_commit_resolves_without_huggingface_hubs_file_listing(tmp_path, monkeypatch):
    from huggingface_hub import constants
    monkeypatch.setattr(constants, 'HF_HUB_OFFLINE', True)  # every request refused: nothing can reach the network
    folder = tmp_path / 'models--SupersonicLabs--Julia-1' / 'snapshots' / JULIA_REVISION
    (folder / 'julia').mkdir(parents=True)
    (folder / 'julia' / '__init__.py').write_text('')
    assert not (tmp_path / 'models--SupersonicLabs--Julia-1' / 'trees').exists()  # written by an older huggingface_hub, or copied
    assert SNAPSHOT(JULIA_MODEL_ID, JULIA_REVISION, tmp_path) == str(folder)
    with pytest.raises(Exception) as error: SNAPSHOT(JULIA_MODEL_ID, 'b' * 40, tmp_path)  # not cached: the first failure is raised
    assert 'offline' in str(error.value).lower()


def test_snapshot_falls_back_to_the_cached_folder_only_when_the_hub_fails(tmp_path, monkeypatch):
    calls = []
    def download(model_id, **options):
        calls.append(options.get('local_files_only', False))
        if not options.get('local_files_only'): raise ConnectionError('Hub unreachable')
        if model_id == 'missing/model': raise FileNotFoundError('not cached')
        return 'cached'
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=download))
    assert SNAPSHOT() == 'cached' and calls == [False, True]
    with pytest.raises(ConnectionError, match='Hub unreachable'): SNAPSHOT('missing/model', JULIA_REVISION)


def test_conftest_never_lets_a_test_download_julia():
    with pytest.raises(RuntimeError, match='never download'): teacher.snapshot()
    assert JuliaEmotionEngine(None).ensure_loaded() is None
