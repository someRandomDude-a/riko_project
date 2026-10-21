"""Julia-1, which the emotion engine (emotion/julia.py) and memory's System 1 (persistence/memory.py) both run.

Julia-1's repository ships the Python package that loads it, so loading it runs code from Hugging Face. Here that code
is downloaded at a pinned commit (kernel/model_pins.py; emotion.revision, memory.system1_revision), checked file by file
against the pin's SHA-256 manifest before anything imports it, and imported under a private name (riko_julia_<digest>
of the snapshot folder) without touching sys.path or sys.modules['julia']: an installed package named julia (PyJulia)
stays importable, and two snapshots never share modules. Both callers share one loaded model per (snapshot, device,
options), loaded once even when their warmups start together (runtime/warmup.py).

The import is confined to the files of the manifest (_Snapshots, a finder only for riko_julia_* names): each module is
compiled from the very bytes whose SHA-256 was checked, never from bytecode (__pycache__, a sourceless package folder),
an extension module or a file the manifest lacks, and no bytecode is written. A folder that is not the pinned commit (the
user's own emotion.model_path, or another commit they chose) runs from the .py files hashed when it is first imported.

The tables below are guarded by _LOCK, which nothing holds while importing or loading (each key has its own lock for
that) and under which nothing is published or called back.
"""
import hashlib
import importlib.abc
import importlib.util
import logging
from pathlib import Path
import sys
import threading
import weakref

from ..kernel.model_pins import JULIA_CODE_SHA256, JULIA_MODEL_ID, JULIA_REVISION, commit
from ..kernel.torch_device import resolve
from .compat import compatible_engine

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_RUNTIMES = {}  # {snapshot folder: its imported julia package}
_MODELS = weakref.WeakValueDictionary()  # {(folder, device, options): model}: freed once every user lets go (close())
_DEVICES = {}  # {(folder, device, options): the device the model loaded on}
_LOADING = {}  # {key: the lock one import or load of that key holds}
_PACKAGES = {}  # {riko_julia_<digest>: {module name: (file, its SHA-256, is a package)}}: all _Snapshots will import


def load_julia(julia, source, device, **options):
    """Load without leaking Julia's process-wide torch settings; a failed GPU load retries on CPU.

    Julia's FastEngine moves inputs to the device only for CUDA, so on MPS every predict fails
    ('Passed CPU tensor to MPS op'): run it on CPU there until upstream supports MPS.
    """
    from ..kernel.torch_device import preserve_torch_globals
    target = resolve(device)
    if target == 'mps': logger.info('Julia 1 does not support MPS yet; using CPU'); target = 'cpu'
    with preserve_torch_globals():
        try: return julia.load_model(source, device=target, **options), target
        except Exception as exc:
            if target == 'cpu': raise
            logger.warning('Julia 1 could not load on %s; retrying on CPU: %s', target, exc)
            return julia.load_model(source, device='cpu', **options), 'cpu'


def snapshot(model_id=JULIA_MODEL_ID, revision=JULIA_REVISION, cache_dir=None):
    """The Hugging Face snapshot folder of model_id at revision (…/snapshots/<commit>), downloaded once; offline, a cached
    snapshot of that commit resolves without the network. A branch or tag is refused before any download."""
    commit('revision', revision)
    from huggingface_hub import snapshot_download
    options = dict(revision=revision, cache_dir=str(cache_dir) if cache_dir else None)
    try: return snapshot_download(model_id, **options)
    except Exception as exc: failure = exc
    # Asked for a commit, huggingface_hub needs that commit's file listing (trees/<commit>.json in the cache) or the Hub; a
    # snapshot an older huggingface_hub wrote, or a copied cache, has no listing, so offline or with the Hub unreachable look
    # for the snapshot folder itself (verify() still checks its code; missing weights fail the load).
    try: return snapshot_download(model_id, local_files_only=True, **options)
    except Exception: pass
    raise failure


def code_hashes(source):
    """{relative path: SHA-256} of the snapshot's julia/**/*.py, the only files its import runs (bytecode is never read)."""
    root = Path(source)
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / 'julia').rglob('*.py')) if '__pycache__' not in path.relative_to(root).parts}


def verify(source, model_id, revision):
    """The manifest to import the snapshot's code against: the pin's, once the snapshot's code matches it exactly (else
    RuntimeError); None for another commit, which runs unverified, with a warning (choosing one is a confirmed,
    security-sensitive settings change)."""
    if (model_id, revision) != (JULIA_MODEL_ID, JULIA_REVISION):
        logger.warning('Julia 1 %s at %s: its bundled Python code runs unverified (only %s at %s is checked file by file)',
            model_id, revision, JULIA_MODEL_ID, JULIA_REVISION)
        return None
    found = code_hashes(source)
    missing, extra = sorted(JULIA_CODE_SHA256.keys() - found.keys()), sorted(found.keys() - JULIA_CODE_SHA256.keys())
    changed = sorted(name for name in found.keys() & JULIA_CODE_SHA256.keys() if found[name] != JULIA_CODE_SHA256[name])
    if missing or extra or changed:
        problems = [f'{label}: {", ".join(names)}' for label, names in (('missing', missing), ('unexpected', extra), ('changed', changed)) if names]
        raise RuntimeError(f'Julia 1 code in {source} does not match commit {revision}, so it was not run ({"; ".join(problems)})')
    return JULIA_CODE_SHA256


class _Checked(importlib.abc.Loader):
    """Runs one file of a snapshot's julia package from the bytes it hashes, so what runs is exactly what was checked."""
    def __init__(self, path, digest): self.path, self.digest = path, digest

    def create_module(self, spec): return None

    def exec_module(self, module):
        data = self.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.digest:
            raise ImportError(f'{self.path} changed after Julia 1 was checked, so it was not run')
        exec(compile(data, str(self.path), 'exec', dont_inherit=True), module.__dict__)


class _Snapshots(importlib.abc.MetaPathFinder):
    """Finds riko_julia_<digest> and its submodules only among the files of its manifest; any other name under it is not
    found (the default finders, which would take bytecode or an extension module first, are never asked)."""
    def find_spec(self, name, path=None, target=None):
        files = _PACKAGES.get(name.partition('.')[0])
        if files is None: return None
        if name not in files: raise ModuleNotFoundError(f'No module named {name!r} in the checked Julia 1 code', name=name)
        file, digest, package = files[name]
        return importlib.util.spec_from_file_location(name, file, loader=_Checked(file, digest),
            submodule_search_locations=[str(file.parent)] if package else None)


_FINDER = _Snapshots()


def _modules(name, root, manifest):
    """{module name: (file, SHA-256, is a package)} for the julia/**/*.py files of a manifest, under the package name."""
    files = {}
    for relative, digest in manifest.items():
        parts = relative.split('/')
        if parts[0] != 'julia' or not relative.endswith('.py'): continue
        package = parts[-1] == '__init__.py'
        files['.'.join([name, *parts[1:-1]] + ([] if package else [parts[-1][:-3]]))] = (root.joinpath(*parts), digest, package)
    return files


def _key_lock(key):
    with _LOCK: return _LOADING.setdefault(key, threading.Lock())


def _import(root, manifest):
    if not (root / 'julia' / '__init__.py').is_file():
        # Only a user's own emotion.model_path can lack the code (verify() requires it in a pinned snapshot): use an
        # installed julia package, as the import search did when it had the folder first.
        return importlib.import_module('julia')
    name = 'riko_julia_' + hashlib.sha256(str(root).encode('utf-8')).hexdigest()[:16]
    files = _modules(name, root, code_hashes(root) if manifest is None else manifest)
    with _LOCK:
        _PACKAGES[name] = files
        if _FINDER not in sys.meta_path: sys.meta_path.insert(0, _FINDER)
    try: return importlib.import_module(name)
    except BaseException:
        for loaded in [key for key in sys.modules if key == name or key.startswith(name + '.')]: sys.modules.pop(loaded, None)
        raise


def runtime(source, manifest=None):
    """The julia package of the snapshot at source, imported once per folder under its own module name, from the files of
    manifest ({relative path: SHA-256}: verify()'s), else of the folder's own .py files as they are now."""
    root = Path(source).resolve()
    with _LOCK: module = _RUNTIMES.get(root)
    if module is not None: return module
    with _key_lock(('runtime', root)):
        with _LOCK: module = _RUNTIMES.get(root)
        if module is None:
            module = _import(root, manifest)
            with _LOCK: _RUNTIMES[root] = module
    return module


def shared_model(source, device, manifest=None, **options):
    """(model, device it loaded on) for the snapshot at source, loaded once per (folder, device, options) and shared by
    every caller asking for the same: memory and emotion hold one copy (about 600 MB) when their settings agree. manifest:
    verify()'s, which the code is imported against (runtime())."""
    target = resolve(device)
    key = (Path(source).resolve(), 'cpu' if target == 'mps' else target, tuple(sorted(options.items())))
    with _LOCK:
        model = _MODELS.get(key)
        if model is not None: return model, _DEVICES[key]
    with _key_lock(key):
        with _LOCK:
            model = _MODELS.get(key)
            if model is not None: return model, _DEVICES[key]
        model, active = load_julia(runtime(source, manifest), source, device, **options)
        compatible_engine(model)
        with _LOCK:
            try: _MODELS[key] = model
            except TypeError: pass  # a model that cannot be weakly referenced is simply not shared
            _DEVICES[key] = active
    return model, active
