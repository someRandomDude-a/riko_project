import threading
from pathlib import Path

import httpx
import pytest

from process.app_core.configuration.config import RuntimeConfig, MemoryConfig
from process.app_core.inference.llama_native import InProcessLlamaProvider
from process.app_core.inference.llama_context import SlotScheduler, native_arguments, BackgroundPreempted
from process.app_core.conversation.messages import ChatMessage


def test_live_slot_is_never_used_by_background():
    scheduler = SlotScheduler(3)
    with scheduler.lease('reflection') as (a, _), scheduler.lease('initiative') as (b, _):
        assert {a, b} == {1, 2}
        with scheduler.lease('live') as (slot, _): assert slot == 0


def test_initiative_preempts_reflection_and_overtakes_queued_reflection():
    scheduler = SlotScheduler(2)
    order = []
    with scheduler.lease('reflection') as (_, stop):
        def run(role):
            with scheduler.lease(role): order.append(role)
        reflection = threading.Thread(target=run, args=('reflection',))
        initiative = threading.Thread(target=run, args=('initiative',))
        reflection.start(); initiative.start()
        assert stop.wait(1)
    reflection.join(1); initiative.join(1)
    assert order == ['initiative', 'reflection']


def test_closing_scheduler_wakes_waiters_and_active_jobs():
    scheduler = SlotScheduler(2)
    cancelled = threading.Event()
    with scheduler.lease('reflection') as (_, stop):
        def waiting():
            try:
                with scheduler.lease('reflection'): pass
            except BackgroundPreempted: cancelled.set()
        worker = threading.Thread(target=waiting); worker.start()
        scheduler.close()
        assert stop.is_set() and cancelled.wait(1)
        worker.join(1)


@pytest.mark.parametrize('count', [1, 5, True])
def test_slot_count_validation(count):
    with pytest.raises(ValueError): InProcessLlamaProvider(RuntimeConfig(model_path=Path('model.gguf'), parallel_slots=count))


def test_server_flags_allocate_per_slot_context_and_one_model():
    config = RuntimeConfig(model_path=Path('model.gguf'), parallel_slots=3, n_ctx=4096, flash_attn=True,
        type_k='q8_0', type_v='q8_0', n_threads=4)
    args = native_arguments(config, config.model_path)
    assert args.count('--model') == 1
    assert args[args.index('--ctx-size')+1] == '12288'
    assert args[args.index('--parallel')+1] == '3'
    assert args[args.index('--cache-type-v')+1] == 'q8_0'
    assert '--kv-unified' in args and '--no-context-shift' in args


def test_http_inference_pins_roles_and_enables_prompt_cache(monkeypatch):
    requests = []
    def handler(request):
        import json
        if request.url.path == '/apply-template': return httpx.Response(200, json={'prompt': 'rendered prompt'})
        if request.url.path == '/tokenize': return httpx.Response(200, json={'tokens': [1, 2, 3]})
        assert request.url.path == '/v1/responses'
        payload = json.loads(request.content); requests.append(payload)
        return httpx.Response(200, text='data: {"type":"response.output_text.delta","delta":"Hello "}\n\ndata: {"type":"response.output_text.delta","delta":"world"}\n\ndata: {"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Hello world"}]}]}}\n\n')
    provider = InProcessLlamaProvider(RuntimeConfig(model_path=Path('model.gguf')))
    provider.client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
    monkeypatch.setattr(provider, '_start', lambda: None)
    monkeypatch.setattr(provider, '_inference_client', lambda: httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler)))
    try:
        assert provider.generate([ChatMessage('user','hi')]).message.content == 'Hello world'
        assert provider.reflection.generate([ChatMessage('user','reflect')]).message.content == 'Hello world'
        assert provider.initiative.generate([ChatMessage('user','consider')]).message.content == 'Hello world'
        assert [r['id_slot'] for r in requests] == [0, 1, 1]
        assert all(r['cache_prompt'] for r in requests)
        assert all('input' in r and 'max_output_tokens' in r and 'messages' not in r and 'max_tokens' not in r for r in requests)
    finally: provider.close()


def test_background_context_overflow_stops_before_inference(monkeypatch):
    requests = []
    def handler(request):
        requests.append(request.url.path)
        if request.url.path == '/apply-template': return httpx.Response(200, json={'prompt': 'large prompt'})
        if request.url.path == '/tokenize': return httpx.Response(200, json={'tokens': list(range(4000))})
        raise AssertionError('Inference must not start for an oversized background prompt')
    provider = InProcessLlamaProvider(RuntimeConfig(model_path=Path('model.gguf')))
    provider.client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
    monkeypatch.setattr(provider, '_start', lambda: None)
    monkeypatch.setattr(provider, '_inference_client', lambda: httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(ValueError, match='exceeding context'):
            provider.reflection.generate([ChatMessage('user', 'reflect')], max_output_tokens=1024, context_limit=4096)
        assert requests == ['/apply-template', '/tokenize']
    finally: provider.close()


def test_managed_provider_token_packs_before_responses_inference(monkeypatch):
    import json
    captured = []
    def handler(request):
        body = json.loads(request.content)
        if request.url.path == '/apply-template':
            assert body['tools'][0]['function']['name'] == 'example'
            return httpx.Response(200, json={'prompt': json.dumps(body)})
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'tokens': list(range(len(body['content'])))})
        assert request.url.path == '/v1/responses'
        captured.append(body)
        return httpx.Response(200, text='data: {"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}}\n\n')
    provider = InProcessLlamaProvider(RuntimeConfig(model_path=Path('model.gguf'), n_ctx=1024))
    provider.client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
    monkeypatch.setattr(provider, '_start', lambda: None)
    monkeypatch.setattr(provider, '_inference_client', lambda: httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler)))
    messages = [ChatMessage('system', 'keep identity'), ChatMessage('user', 'old' * 1000), ChatMessage('assistant', 'old answer'), ChatMessage('user', 'current')]
    try:
        response = provider.generate(messages, max_output_tokens=100, tools=[{'type':'function','function':{'name':'example','parameters':{'type':'object'}}}])
        assert len(captured) == 1
        assert [m.content for m in response.context_messages] == ['keep identity', 'current']
        assert len(messages) == 4
        assert captured[0]['input'][0]['content'] == 'keep identity'
    finally: provider.close()


def test_memory_dispatches_parallel_reflections_without_duplicate_sources(tmp_path, monkeypatch):
    from process.app_core.persistence.memory import MemoryStore, MemoryRecord
    from types import SimpleNamespace
    config = MemoryConfig(store_file=tmp_path/'memory.json', embeddings_enabled=False, system1_enabled=False)
    store = MemoryStore(config, reflection_provider=SimpleNamespace(owner=SimpleNamespace(reflection_parallelism=3)), start_worker=False)
    entered = set(); ready = threading.Event(); release = threading.Event()
    def reflect(record):
        with store.lock:
            assert record.id not in entered
            entered.add(record.id)
            if len(entered) == 3: ready.set()
        release.wait(2)
        with store.lock: store._find(record.id).reflection_status = 'complete'
    monkeypatch.setattr(store, '_reflect', reflect)
    store.records = [MemoryRecord(text=f'memory {i}', classification_status='complete', reflection_status='pending') for i in range(3)]
    store.start()
    try: assert ready.wait(1)
    finally: release.set(); store.close()


def test_warmup_does_not_propagate_optional_component_failure():
    from process.app_core.runtime.warmup import warm_components
    ready = []
    def fail(): raise RuntimeError('service offline')
    warm_components([('offline', fail), ('ready', lambda: ready.append(True))], 1)
    assert ready == [True]


def test_native_warmup_probes_every_slot_without_using_chat_history(monkeypatch):
    import json
    requests = []
    def handler(request):
        assert request.url.path == '/v1/responses'
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={})
    provider = InProcessLlamaProvider(RuntimeConfig(model_path=Path('model.gguf'), parallel_slots=4))
    provider.client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
    monkeypatch.setattr(provider, '_start', lambda: None)
    try:
        provider.warmup()
        assert [r['id_slot'] for r in requests] == [0, 1, 2, 3]
        assert all(r['max_output_tokens'] == 1 for r in requests)
    finally: provider.close()


def test_missing_native_library_has_actionable_startup_error():
    provider = InProcessLlamaProvider(RuntimeConfig(model_path=Path('model.gguf')))
    try:
        with pytest.raises(RuntimeError, match='runtime.native_library'): provider.warmup()
    finally: provider.close()


def test_provider_reasoning_is_separate_from_reply_and_not_synthesized():
    from process.app_core.inference.providers import assemble_stream
    text,reasoning=[],[]
    result=assemble_stream([
        {'choices':[{'delta':{'reasoning_content':'Provider returned this.'}}]},
        {'choices':[{'delta':{'content':'Hello there'},'finish_reason':'stop'}]},
    ],text.append,on_reasoning=reasoning.append)
    assert reasoning==['Provider returned this.']
    assert result.message.content=='Hello there'
    assert ''.join(text)=='Hello there'
