"""The real native bridge: a riko-native build (tools/llama_cpp/riko-native.cpp at the pinned llama.cpp with the probe patch)
and a small GGUF, through the InProcessLlamaProvider the app uses. Every other test mocks the library, so only this one
checks the ctypes signatures against the C ABI, the slot layout, streaming, cancellation and the emotion probe.
.github/workflows/native-smoke.yml builds a Metal bundle and runs it; locally, for example:

    RIKO_TEST_NATIVE_LIBRARY=.native/metal/libriko-native.dylib RIKO_TEST_GGUF=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
        python -m pytest -m native

RIKO_TEST_EXPECT_GPU=1 also requires llama.cpp to have put layers on a GPU, so a silent CPU fallback fails.
RIKO_TEST_EXPECT_DEVICES=1 requires the bridge's riko_devices export (a bridge built before it passes without).
"""
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace

import pytest

LIBRARY, MODEL = os.environ.get('RIKO_TEST_NATIVE_LIBRARY'), os.environ.get('RIKO_TEST_GGUF')
pytestmark = [pytest.mark.native, pytest.mark.skipif(not (LIBRARY and MODEL),
    reason='set RIKO_TEST_NATIVE_LIBRARY and RIKO_TEST_GGUF to load a real riko-native build and GGUF')]
COUNT = 'Count from one to twenty in words, separated by spaces.'


@pytest.fixture
def provider(tmp_path):
    """The provider load_config and factory.create_chat_service would build, on a small context so any machine fits it."""
    from process.app_core.configuration.config import load_config
    from process.app_core.inference import llama_native
    path = tmp_path / 'character_config.yaml'
    path.write_text(f'''runtime:
  provider: llama_cpp
  native_library: {json.dumps(str(Path(LIBRARY).resolve()))}
  model_path: {json.dumps(str(Path(MODEL).resolve()))}
  n_ctx: 2048
  max_output_tokens: 64
  temperature: 0
  seed: 7
  parallel_slots: 2
  request_timeout_seconds: 300
initiative:
  context_window_tokens: 1024
  max_output_tokens: 128
memory:
  reflection_context_window_tokens: 1024
  reflection_max_output_tokens: 128
''', encoding='utf-8')
    provider = llama_native.InProcessLlamaProvider(load_config(path).runtime)
    try: yield provider
    finally:
        provider.close()
        deadline = time.monotonic() + 30  # close() may hand the last riko_destroy to a cleanup thread
        while llama_native._LIVE and time.monotonic() < deadline: time.sleep(.05)
        # A context still live at exit makes llama_native's atexit hook os._exit(1) to keep Metal from asserting.
        assert not llama_native._LIVE, 'the native context was not destroyed'


def probe_samples(provider, slot):
    """riko.emotion_probe.sample events from one raw streamed /v1/responses request on `slot`."""
    from process.app_core.inference.responses import sse_events
    body = {'input': [{'role': 'user', 'content': COUNT}], 'stream': True, 'max_output_tokens': 40, 'temperature': 0, 'id_slot': slot}
    with provider._inference_client() as client, client.stream('POST', '/v1/responses', json=body) as response:
        provider._check_response(response)
        events = list(sse_events(response.iter_lines()))
    assert events[-1]['type'] == 'response.completed'
    return [event for event in events if event.get('type') == 'riko.emotion_probe.sample']


def test_provider_loads_streams_counts_cancels_and_closes(provider):
    import run_server
    from process.app_core.kernel.messages import ChatMessage
    from process.app_core.kernel.cancellation import BackgroundPreempted
    provider.warmup()  # loads the GGUF, checks /slots, and answers one token on every slot, as the backend does at startup
    notes = provider.native.notes
    offload = re.search(r'offloaded (\d+)/(\d+) layers', notes)
    if os.environ.get('RIKO_TEST_EXPECT_GPU') == '1': assert offload and int(offload[1]) > 0, notes
    library = Path(LIBRARY).resolve()  # every library the bundle holds loaded from the bundle (@loader_path / $ORIGIN rpath)
    loaded = run_server.loaded_libraries({file.name for file in library.parent.iterdir()} | run_server.DRIVER_LIBRARIES)
    assert library.name in loaded and run_server.native_bundle_problems(library, loaded) == []
    props = provider.client.get('/props').json()
    assert props['riko_emotion_probe'] == 'disabled' and props['build_info']
    slots = provider.client.get('/slots').json()
    assert [slot['id'] for slot in slots] == [0, 1] and all(slot['n_ctx'] >= 2048 for slot in slots)
    devices = provider.devices()  # riko_devices: optional, so an older bridge answers None and telemetry falls back
    if os.environ.get('RIKO_TEST_EXPECT_DEVICES') == '1': assert devices is not None, 'this bridge has no riko_devices'
    if devices is not None:
        assert any(device['type'] == 'cpu' for device in devices), devices
        if os.environ.get('RIKO_TEST_EXPECT_GPU') == '1':
            gpus = [device for device in devices if device['type'] in {'gpu', 'igpu'} and device['memory_total'] > 0]
            assert gpus and any(device['model'] and device['main'] for device in gpus), devices

    messages = [ChatMessage('system', 'You are a concise assistant.'), ChatMessage('user', 'Name three colours.')]
    assert 10 < provider.count_tokens(messages) < 200
    assert 1 <= provider.count_text_tokens('Hello world') <= 4
    deltas = []
    reply = provider.generate(messages, on_delta=deltas.append, max_output_tokens=32)  # the live lane: slot 0
    assert len(deltas) > 1 and ''.join(deltas) == reply.message.content and reply.message.content.strip()
    assert reply.finish_reason in {'stop', 'length'} and 0 < reply.usage['output_tokens'] <= 32
    background = provider.initiative.generate(messages, max_output_tokens=8, context_limit=1024)  # a background slot
    assert background.message.content.strip() and background.usage['output_tokens'] <= 8

    stop = threading.Event()  # cancelling mid-reply returns through the native cancel callback and frees the slot
    with pytest.raises(BackgroundPreempted):
        provider.generate([ChatMessage('user', COUNT)], on_delta=lambda _: stop.set(), cancelled=stop.is_set, max_output_tokens=256)
    assert ''.join(provider.stream([ChatMessage('user', 'Say hello.')], max_output_tokens=8)).strip()


def test_emotion_probe_samples_the_live_slot_only(provider):
    import torch
    from process.app_core.kernel.messages import ChatMessage
    from process.app_core.emotion.probe_hook import FEATURE_VERSION, FEATURE_WIDTH, ProbeHook
    captures, identities = [], []

    class Probe:  # what factory.create_chat_service attaches, reduced to the calls the provider makes
        active_group, config = None, SimpleNamespace(interval_tokens=4)
        def activate(self, group): self.active_group = group
        def capture(self, hidden, transcript, group, **options): captures.append((hidden, transcript, group, options))
        def close(self): pass

    provider.attach_probe(ProbeHook(lambda identity, idle: identities.append(identity) or Probe()), 4)
    provider.warmup()
    assert provider.client.get('/props').json()['riko_emotion_probe'] == FEATURE_VERSION
    assert len(identities) == 1 and len(identities[0]['gguf_sha256']) == 64 and identities[0]['server_build']
    assert isinstance(provider.probe, Probe) and provider.probe_error == ''

    samples = probe_samples(provider, 0)
    assert samples and all(sample['feature_version'] == FEATURE_VERSION and len(sample['features']) == FEATURE_WIDTH
        and all(math.isfinite(value) for value in sample['features']) and sample['prefix_bytes'] > 0 for sample in samples)
    assert probe_samples(provider, 1) == []  # the patch arms the capture on slot 0 only

    reply = provider.generate([ChatMessage('user', COUNT)], on_delta=lambda _: None, max_output_tokens=40, emotion_turn_id='turn-1')
    assert captures, 'no probe sample matched the visible reply'
    for hidden, transcript, group, options in captures:
        assert group == 'turn-1' and tuple(hidden.shape) == (256,) and bool(torch.isfinite(hidden).all())
        assert transcript.startswith(f'user: {COUNT}\nassistant: ') and 0 < options['offset'] <= len(reply.message.content)
    captures.clear()
    provider.initiative.generate([ChatMessage('user', COUNT)], max_output_tokens=24, context_limit=1024)
    assert captures == []
    provider.replay_lane.generate([ChatMessage('user', COUNT)], max_output_tokens=40, emotion_turn_id='replay-1')  # EmotionProbe.replay's lane
    assert captures and all(group == 'replay-1' and options['replay'] for _, _, group, options in captures)


def test_a_probe_that_cannot_start_leaves_the_real_model_answering(provider):
    from process.app_core.kernel.messages import ChatMessage
    from process.app_core.emotion.probe_hook import ProbeHook
    def unavailable(identity, idle): raise RuntimeError('Unable to load Julia 1 emotion model: offline')
    provider.attach_probe(ProbeHook(unavailable), 4)
    provider.warmup()
    assert provider.probe is None and 'Julia 1' in provider.probe_error
    assert provider.generate([ChatMessage('user', 'Say hello.')], max_output_tokens=8).message.content.strip()
