"""One ASR factory: auto/default resolve to what CTranslate2 can run here, and a pair it cannot run falls back once."""
import logging
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from process.app_core.audio import asr

MAC = {'cpu': {'float32', 'int8', 'int8_float32'}}
RTX = {**MAC, 'cuda': {'float32', 'float16', 'bfloat16', 'int8', 'int8_float32', 'int8_float16', 'int8_bfloat16'}}
PASCAL = {**MAC, 'cuda': {'float32', 'int8', 'int8_float32'}}


def machine(monkeypatch, types):
    monkeypatch.setattr(asr, 'supported_types', lambda: {device: frozenset(names) for device, names in types.items()})
    monkeypatch.setattr(asr, '_warned', set())


def whisper(monkeypatch):
    built = []
    class WhisperModel:
        def __init__(self, name, **options): built.append((name, options))
        def transcribe(self, samples, **options): return iter([SimpleNamespace(text=' heard you ')]), None
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=WhisperModel))
    return built


@pytest.mark.parametrize('types, voice, device, precision, note', [
    (MAC, {}, 'cpu', 'int8', ''),
    (MAC, {'asr_device': 'cuda', 'asr_compute_type': 'int8_float16'}, 'cpu', 'int8', 'no CUDA device is visible to CTranslate2; cpu does not support int8_float16'),
    (MAC, {'asr_device': 'cpu'}, 'cpu', 'int8', ''),
    (MAC, {'asr_device': 'cpu', 'asr_compute_type': 'int8_float16'}, 'cpu', 'int8', 'cpu does not support int8_float16'),
    (MAC, {'asr_device': 'auto', 'asr_compute_type': 'float16'}, 'cpu', 'int8', 'cpu does not support float16'),
    (MAC, {'asr_device': 'cuda', 'asr_compute_type': 'float32'}, 'cpu', 'float32', 'no CUDA device is visible to CTranslate2'),
    (MAC, {'asr_device': 'mps'}, 'cpu', 'int8', 'unknown device mps'),
    (RTX, {}, 'cuda', 'int8_float16', ''),
    (RTX, {'asr_device': 'cuda', 'asr_compute_type': 'int8_float16'}, 'cuda', 'int8_float16', ''),
    (RTX, {'asr_device': 'cpu', 'asr_compute_type': 'default'}, 'cpu', 'int8', ''),
    (PASCAL, {}, 'cuda', 'int8_float32', ''),
    (PASCAL, {'asr_device': 'cuda', 'asr_compute_type': 'int8_float16'}, 'cuda', 'int8_float32', 'cuda does not support int8_float16'),
])
def test_resolution_keeps_runnable_pairs_and_replaces_the_rest(monkeypatch, types, voice, device, precision, note):
    machine(monkeypatch, types)
    assert asr.resolve(voice) == (device, precision, note)


def test_resolved_default_is_accepted_by_this_ctranslate2_build():
    ctranslate2 = pytest.importorskip('ctranslate2')
    asr.supported_types.cache_clear()
    try:
        device, precision, note = asr.resolve({})
        assert not note and precision in ctranslate2.get_supported_compute_types(device)
        assert asr.resolve({'asr_device': 'cpu', 'asr_compute_type': 'int8_float16'})[:2] == ('cpu', 'int8')
    finally: asr.supported_types.cache_clear()


def test_probe_lists_cuda_only_with_a_visible_device(monkeypatch):
    def supported(device):
        if device == 'cuda': raise ValueError('This CTranslate2 package was not compiled with CUDA support')
        return {'float32', 'int8'}
    monkeypatch.setitem(sys.modules, 'ctranslate2', SimpleNamespace(get_cuda_device_count=lambda: 0, get_supported_compute_types=supported))
    asr.supported_types.cache_clear()
    try: assert asr.supported_types() == {'cpu': frozenset({'float32', 'int8'})}
    finally: asr.supported_types.cache_clear()


def test_factory_builds_the_resolved_pair_and_warns_once(monkeypatch, caplog):
    machine(monkeypatch, MAC)
    built = whisper(monkeypatch)
    voice = {'asr_device': 'cuda', 'asr_compute_type': 'int8_float16'}
    with caplog.at_level(logging.WARNING, logger=asr.__name__):
        asr.create_whisper(voice); asr.create_whisper(voice)
    assert built == [('distil-small.en', {'device': 'cpu', 'compute_type': 'int8'})] * 2
    assert len([record for record in caplog.records if 'cannot run here' in record.getMessage()]) == 1


def test_discord_transcription_builds_through_the_factory(monkeypatch):
    from process.app_core.integrations.discord.api import transcribe_pcm
    machine(monkeypatch, MAC)
    built = whisper(monkeypatch)
    service = asr.AsrService({'asr_device': 'cpu', 'asr_compute_type': 'int8_float16'})
    session = SimpleNamespace(transcribe=service.transcribe_pcm)
    assert transcribe_pcm(session, bytes(3200)) == 'heard you'
    assert built == [('distil-small.en', {'device': 'cpu', 'compute_type': 'int8'})] and service.model is not None


def test_only_the_factory_constructs_whisper_or_defaults_the_pair():
    root = Path(__file__).resolve().parents[1] / 'Code'
    pattern = re.compile(r"WhisperModel\(|asr_device['\"]\s*[,:]\s*['\"]cuda|asr_compute_type['\"]\s*[,:]\s*['\"]int8_float16")
    hits = [f'{path.relative_to(root)}:{number}' for path in root.rglob('*.py') if path.relative_to(root).as_posix() != 'process/app_core/audio/asr.py'
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1) if pattern.search(line)]
    assert not hits, 'Build faster-whisper through audio/asr.py:create_whisper:\n' + '\n'.join(hits)
