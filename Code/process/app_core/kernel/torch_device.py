"""Torch device names for in-process models: auto, cpu, cuda[:n] (ROCm torch reports HIP as cuda) and mps.

torch is imported only to probe `auto` or to guard a load, so validating configuration stays cheap.
"""
from contextlib import contextmanager
import re
import threading

_LOAD_LOCK = threading.Lock()
_THREADS = []


def validate(name):
    value = str(name).strip().lower()
    if value in {'auto', 'cpu', 'cuda', 'mps'} or re.fullmatch(r'cuda:\d+', value): return value
    raise ValueError(f'Unsupported torch device {name!r}: use auto, cpu, cuda, cuda:<n> or mps')


def resolve(name):
    """auto prefers CUDA (also ROCm/HIP), then Apple MPS, then CPU; explicit names pass through."""
    value = validate(name)
    if value != 'auto': return value
    import torch
    if torch.cuda.is_available(): return 'cuda'
    mps = getattr(torch.backends, 'mps', None)
    return 'mps' if mps is not None and mps.is_available() else 'cpu'


@contextmanager
def preserve_torch_globals():
    """Undo process-wide torch settings a model load changes (Julia's configure() sets threads to
    JULIA_CPU_THREADS or 4 and matmul precision to 'high'), so its default cap of 4 no longer slows every
    other torch user; an explicit JULIA_CPU_THREADS still applies. Loads are serialised so concurrent
    warmup loads cannot save each other's capped thread count."""
    import os
    import torch
    with _LOAD_LOCK:
        if not _THREADS: _THREADS.append(torch.get_num_threads())
        precision = torch.get_float32_matmul_precision()
        try: yield
        finally:
            requested = os.environ.get('JULIA_CPU_THREADS', '')
            torch.set_num_threads(int(requested) if requested.isdigit() and int(requested) > 0 else _THREADS[0])
            torch.set_float32_matmul_precision(precision)
