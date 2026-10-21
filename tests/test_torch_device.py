"""Device policy for in-process torch models (Julia, memory classifier, embeddings)."""
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from process.app_core.configuration.config import MemoryConfig
from process.app_core.emotion import JuliaEmotionEngine
from process.app_core.emotion.julia import load_julia
from process.app_core.persistence.memory import MemoryStore
from process.app_core.kernel.torch_device import resolve, validate


def fake_torch(cuda, mps):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


def test_device_names_and_auto_resolution(monkeypatch):
    assert [validate(name) for name in ('cpu', 'CUDA', 'cuda:1', 'mps', 'auto')] == ['cpu', 'cuda', 'cuda:1', 'mps', 'auto']
    for name in ('vulkan', 'cuda:', 'rocm', ''):
        with pytest.raises(ValueError, match='Unsupported torch device'): validate(name)
    for cuda, mps, expected in ((True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')):
        monkeypatch.setitem(sys.modules, 'torch', fake_torch(cuda, mps))
        assert resolve('auto') == expected
    assert resolve('mps') == 'mps'  # Explicit names pass through; loaders handle failure.
    assert JuliaEmotionEngine(None, device='mps').device == 'mps'
    with pytest.raises(ValueError): JuliaEmotionEngine(None, device='vulkan')


def test_julia_load_maps_mps_to_cpu_retries_gpu_on_cpu_and_restores_torch_globals():
    import torch
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    calls = []
    def load_model(source, device, **options):  # what Julia's configure() does on every load
        calls.append(device); torch.set_num_threads(1); torch.set_float32_matmul_precision('high')
        if device != 'cpu': raise RuntimeError('CUDA requested but unavailable')
        return SimpleNamespace(device=device, options=options)
    julia = SimpleNamespace(load_model=load_model)
    model, device = load_julia(julia, 'snapshot', 'cuda:1', max_length=512)
    assert (device, model.options, calls) == ('cpu', {'max_length': 512}, ['cuda:1', 'cpu'])
    calls.clear()
    assert load_julia(julia, 'snapshot', 'mps')[1] == 'cpu' and calls == ['cpu']
    seen = []
    worker = threading.Thread(target=lambda: seen.append(torch.get_num_threads())); worker.start(); worker.join()
    assert torch.get_num_threads() == threads and seen == [threads]
    assert torch.get_float32_matmul_precision() == precision
    with pytest.raises(RuntimeError): load_julia(SimpleNamespace(load_model=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('bad weights'))), 'snapshot', 'cpu')
    assert torch.get_num_threads() == threads


def test_memory_embedder_uses_configured_device_with_cpu_retry(tmp_path, monkeypatch):
    devices = []
    class SentenceTransformer:
        def __init__(self, name, device):
            devices.append(device)
            if device != 'cpu': raise AssertionError('Torch not compiled with CUDA enabled')
        def encode(self, texts, convert_to_numpy): return np.ones((len(texts), 4))
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=SentenceTransformer))
    config = MemoryConfig(store_file=tmp_path / 'memories.json', index_file=tmp_path / 'index', system1_enabled=False, device='cuda')
    memory = MemoryStore(config, start_worker=False)
    try:
        assert memory._embed(['hello']).shape == (1, 4) and devices == ['cuda', 'cpu']
    finally: memory.close()
    with pytest.raises(ValueError, match='Unsupported torch device'):
        MemoryStore(MemoryConfig(store_file=tmp_path / 'other.json', system1_enabled=False, device='gpu'), start_worker=False)


def test_an_explicit_julia_thread_count_still_applies_after_loading(monkeypatch):
    import torch
    from process.app_core.kernel.torch_device import preserve_torch_globals
    threads = torch.get_num_threads()
    monkeypatch.setenv('JULIA_CPU_THREADS', '2')
    try:
        with preserve_torch_globals(): torch.set_num_threads(4)  # what Julia's configure() does
        assert torch.get_num_threads() == 2
    finally: torch.set_num_threads(threads)
