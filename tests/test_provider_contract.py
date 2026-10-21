"""The provider contract (inference/provider.py): every caller uses InferenceProvider members directly and BaseProvider
defaults them, the scheduler keeps the KV pool's one-initiative assumption and gives example replay slot 0 only at
background priority, and the optional emotion probe can fail to start without taking the model down."""
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from process.app_core.configuration.config import RuntimeConfig
from process.app_core.emotion.probe_hook import FEATURE_VERSION, ProbeHook
from process.app_core.inference.llama_context import SlotScheduler, fingerprint
from process.app_core.inference.llama_native import InProcessLlamaProvider, NativeClient
from process.app_core.inference.provider import BaseProvider, InferenceProvider, Lane, ProviderCapabilities, ROLES
from process.app_core.kernel.cancellation import BackgroundPreempted
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths
from conftest import client_for  # the backend fixture comes from conftest.py
from test_llama_native import fake_runtime
from test_private_access import CODE

MEMBERS = sorted({name for protocol in (Lane, InferenceProvider) for name in (*vars(protocol), *vars(protocol).get('__annotations__', {}))
    if not name.startswith('_')})


def eventually(condition, timeout=10):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline: return False
        time.sleep(.01)
    return True


def test_a_provider_that_only_generates_gets_every_member_from_base_provider():
    class Echo(BaseProvider):
        def generate(self, messages, *, tools=None, on_delta=None, **options):
            for word in ('Hello ', 'there.'): on_delta and on_delta(word)
            return ModelResponse(ChatMessage('assistant', 'Hello there.'))
    provider = Echo()
    assert all(hasattr(provider, name) for name in MEMBERS) and {'generate', 'cancel', 'lanes', 'probe_host', 'capabilities'} <= set(MEMBERS)
    assert provider.lanes == {role: provider for role in ROLES} and provider.probe_host is None
    assert provider.capabilities == ProviderCapabilities() and provider.capabilities.background_parallelism == 1
    assert ''.join(provider.stream([ChatMessage('user', 'Hi')])) == 'Hello there.'
    assert provider.count_tokens([ChatMessage('user', 'Hi there')]) > 0 and provider.count_text_tokens('Hi there') > 0
    provider.warmup(); provider.cancel(); provider.set_foreground(True); provider.set_pause_background(False); provider.close()
    assert provider.expression_idle() is True
    provider.set_expression_idle(lambda: False)
    assert provider.expression_idle() is False and BaseProvider().expression_idle() is True  # per instance
    with pytest.raises(NotImplementedError): BaseProvider().generate([])
    assert ProviderCapabilities(slots=4).background_parallelism == 3


def test_every_provider_family_declares_the_whole_contract(tmp_path):
    from process.app_core.inference.llama_server import LlamaServerProvider
    from process.app_core.inference.providers import OpenAIProvider
    library = tmp_path / 'riko-native.dll'; library.write_bytes(b'library')
    openai = OpenAIProvider(RuntimeConfig(provider='openai_compatible', base_url='http://127.0.0.1:9/v1', api_key='x', model='m'))
    server = LlamaServerProvider(RuntimeConfig(provider='llama_server', parallel_slots=3))
    native = InProcessLlamaProvider(RuntimeConfig(provider='llama_cpp', model_path=tmp_path / 'model.gguf', native_library=library, parallel_slots=3))
    try:
        for provider in (openai, server, native): assert isinstance(provider, BaseProvider) and all(hasattr(provider, name) for name in MEMBERS)
        assert openai.capabilities == ProviderCapabilities() and openai.probe_host is None and openai.lanes['initiative'] is openai
        assert server.capabilities == ProviderCapabilities(slots=3, exact_tokens=True) and server.probe_host is None
        assert native.capabilities == ProviderCapabilities(slots=3, exact_tokens=True, latent_probe=True) and native.probe_host is native
        assert native.lanes == {'live': native, 'initiative': native.initiative, 'reflection': native.reflection}
        assert native.capabilities.background_parallelism == 2 and native.probe is None and native.probe_error == ''
        openai.cancel()  # no provider-wide cancel: each request stops through its own `cancelled`
    finally: openai.close(); server.close(); native.close()


def test_no_caller_probes_a_provider_with_hasattr_or_getattr():
    members = set(MEMBERS) - {'close'} | {'probe', 'probe_factory', 'set_probe_interval', 'probe_idle', 'supports_latent_probe', 'reflection_parallelism', 'owner'}
    found = []
    for path in [*sorted((CODE / 'process' / 'app_core').rglob('*.py')), *sorted(CODE.glob('*.py'))]:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'getattr', 'hasattr'} and len(node.args) > 1): continue
            target, name = ast.unparse(node.args[0]), node.args[1]
            if 'provider' in target or (isinstance(name, ast.Constant) and name.value in members):
                found.append(f'{path.relative_to(CODE).as_posix()}:{node.lineno} {ast.unparse(node)}')
    assert not found, 'Use the InferenceProvider member (BaseProvider defaults it):\n' + '\n'.join(found)  # close stays generic (close_bounded)


def test_the_session_drives_the_provider_through_its_declared_members(monkeypatch):
    from process.app_core.conversation.chat import ChatService
    from process.app_core.desktop.state import DesktopState
    from process.app_core.runtime.session import SessionManager
    class Speech:
        def __init__(self, *args): pass
        def submit(self, *args): return False
        def cancel(self): pass
        def close(self): pass
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    calls = []
    class Recorder(BaseProvider):
        def generate(self, messages, **options):
            calls.append(('generate', options['emotion_turn_id'] is not None))
            return ModelResponse(ChatMessage('assistant', 'Hi.'))
        def set_foreground(self, active): calls.append(('foreground', active))
        def cancel(self): calls.append('cancel')
    provider = Recorder()
    config = SimpleNamespace(raw={'animation': False}, root=Path('.'), paths=DataPaths.at(Path('.')), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    session = SessionManager(config, ChatService(provider, system_prompt='Riko'), DesktopState())
    try:
        assert provider.expression_idle() is True  # the session's quiet check, declared rather than assigned
        session._active_turn, session._speech_pending = 'turn', 1  # a sentence still queued to play
        assert provider.expression_idle() is False
        session._speech_pending = 0
        session.respond('hello', speak=False)
        assert calls == [('foreground', True), ('generate', True), ('foreground', False)]
        session.cancel()
        assert calls[-1] == 'cancel'
    finally: session.close()


def hold(scheduler, role, acquired, release, slots):
    """Lease role on a thread, record its slot and stop event, and keep it until release is set."""
    def run():
        try:
            with scheduler.lease(role) as (slot, stop):
                slots[role] = (slot, stop)
                acquired.set()
                release.wait(10)
        except BackgroundPreempted: slots[role] = 'preempted'; acquired.set()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_at_most_one_initiative_runs_and_a_waiting_one_does_not_hold_reflections_back():
    scheduler, slots = SlotScheduler(3), {}
    first, second, reflection = threading.Event(), threading.Event(), threading.Event()
    release_first, release_rest = threading.Event(), threading.Event()
    with scheduler.lease('initiative') as (slot, _):
        assert slot == 1
        waiting = hold(scheduler, 'initiative', second, release_rest, slots)
        assert eventually(lambda: any(t[0] == 1 for t in scheduler.waiting))
        hold(scheduler, 'reflection', reflection, release_rest, slots)
        assert reflection.wait(10) and slots['reflection'][0] == 2  # the slot the second initiative could not take
        assert not second.is_set() and any(t[0] == 1 for t in scheduler.waiting)  # kv_budget sizes the pool for one
    assert second.wait(10) and slots['initiative'][0] == 1
    release_rest.set(); waiting.join(10)


def test_cancel_reaches_only_the_live_request():
    scheduler = SlotScheduler(2)
    with scheduler.lease('initiative') as (_, background), scheduler.lease('probe_replay') as (slot, replay):
        assert slot == 0
        scheduler.cancel_live()
        assert not background.is_set() and not replay.is_set()
    with scheduler.lease('live') as (_, live), scheduler.lease('reflection') as (_, reflection):
        scheduler.cancel_live()
        assert live.is_set() and not reflection.is_set()


def test_replay_gets_slot_zero_only_while_no_live_request_wants_it():
    scheduler, slots = SlotScheduler(2), {}  # pause_background is off: a replay is preempted anyway
    with scheduler.lease('probe_replay') as (_, replay):
        scheduler.set_foreground(True)
        assert replay.is_set()  # a turn started: the replay on the live slot stops
    acquired, release = threading.Event(), threading.Event()
    hold(scheduler, 'probe_replay', acquired, release, slots)
    assert eventually(lambda: any(t[0] == 3 for t in scheduler.waiting)) and not acquired.is_set()
    scheduler.set_foreground(False)
    assert acquired.wait(10) and slots['probe_replay'][0] == 0 and not scheduler.idle()
    live, release_live = threading.Event(), threading.Event()
    hold(scheduler, 'live', live, release_live, slots)  # a live request waiting for slot 0 preempts the replay
    assert slots['probe_replay'][1].wait(10)
    release.set()
    assert live.wait(10) and slots['live'][0] == 0
    release_live.set()
    assert eventually(scheduler.idle)


def native_provider(tmp_path, monkeypatch, props, generated=None):
    """An InProcessLlamaProvider over a fake library: /slots fits, /props reports props, replies say 'Hi'."""
    model = tmp_path / 'model.gguf'; model.write_bytes(b'weights')
    library = tmp_path / 'riko-native.dll'; library.write_bytes(b'library')
    runtime, created = fake_runtime([]), []
    def request(handle, path, body, output, cancel, user):
        path = path.decode()
        if path == '/slots': data = json.dumps([{'n_ctx': 8192}, {'n_ctx': 8192}])
        elif path == '/props': data = json.dumps(props)
        elif path == '/apply-template': data = json.dumps({'prompt': 'rendered'})
        elif path == '/tokenize': data = json.dumps({'tokens': [1]})
        else:
            (generated if generated is not None else []).append(json.loads(body)['id_slot'])
            events = [{'type': 'response.output_text.delta', 'delta': 'Hi'}, {'type': 'response.completed', 'response': {'status': 'completed',
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Hi'}]}]}}]
            data = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
        payload = data.encode()
        buffer = ctypes.create_string_buffer(payload)
        output(200, ctypes.cast(buffer, ctypes.c_void_p), len(payload), user)
        return 0
    runtime.intervals = []
    runtime.dll = SimpleNamespace(riko_request=request, riko_stop=lambda _: None, riko_destroy=lambda _: None,
        riko_set_interval=lambda handle, interval: runtime.intervals.append(interval) or 0)
    monkeypatch.setattr('process.app_core.inference.llama_native.NativeRuntime', lambda library, args, interval: created.append(interval) or runtime)
    provider = InProcessLlamaProvider(RuntimeConfig(provider='llama_cpp', model_path=model, native_library=library))
    return provider, created


@pytest.mark.parametrize('failure', ['stale library', 'probe start'])
def test_a_probe_that_cannot_start_leaves_the_model_up_and_says_why(tmp_path, monkeypatch, failure):
    props = {'riko_emotion_probe': 'disabled' if failure == 'stale library' else FEATURE_VERSION, 'build_info': 'b1', 'chat_template': 'tpl'}
    provider, created = native_provider(tmp_path, monkeypatch, props)
    identities = []
    def make_probe(identity, idle):
        identities.append(identity)
        raise RuntimeError('Unable to load Julia 1 emotion model: offline')
    provider.attach_probe(ProbeHook(make_probe), 16)
    try:
        provider.warmup()  # the model loads; the probe does not
        assert created == [16] and provider.probe is None
        assert ('riko-native library that captures ' + FEATURE_VERSION if failure == 'stale library' else 'Julia 1') in provider.probe_error
        assert provider.generate([ChatMessage('user', 'Hello')]).message.content == 'Hi'
        provider.generate([ChatMessage('user', 'Again')])
        assert created == [16]  # loaded once: a failed probe no longer reloads the model on every request
        provider.set_probe_interval(8)  # a live settings change must not speed the unused capture back up
        assert provider.native.intervals == [512]  # the bridge cannot switch capture off: it samples as rarely as it allows
        if failure == 'probe start':  # the identity names existing probe data: its keys and values are unchanged
            library = tmp_path / 'riko-native.dll'
            assert identities == [{'gguf_sha256': hashlib.sha256(b'weights').hexdigest(), 'server_build': 'b1', 'chat_template': 'tpl',
                'feature_version': FEATURE_VERSION, 'type_k': 'f16', 'type_v': 'f16', 'flash_attn': 'auto', 'n_ctx': 8192,
                'runtime_fingerprint': hashlib.sha256(library.name.encode() + b'library').hexdigest()}]
        else: assert identities == []  # checked before hashing the model
    finally: provider.close()


def test_neural_status_reports_why_the_probe_did_not_start(backend, monkeypatch, tmp_path):
    class Host(BaseProvider):
        probe, probe_error, replay_lane = None, 'The emotion probe did not start: torch is missing', object()
        @property
        def probe_host(self): return self
    backend.chat = SimpleNamespace(provider=Host())
    client = client_for(backend)
    status = client.get('/api/neural/status').json()
    assert status['available'] is False and status['probe_error'] == Host.probe_error and 'torch is missing. Chat works without it' in status['note']
    assert client.post('/api/neural/train').json()['detail'] == Host.probe_error + '; fix the cause and restart Python'
    replays = []
    key = 'a' * 64
    (tmp_path / 'models' / 'qwen' / 'expression probe' / key).mkdir(parents=True)
    (tmp_path / 'models' / 'qwen' / 'expression probe' / key / 'examples.json').write_text(json.dumps({'examples': [{'text': 'hi'}]}))
    Host.probe = SimpleNamespace(replay=lambda examples, lane: replays.append((examples, lane)), status=lambda: {'mode': 'probe'})
    assert client.post('/api/neural/replay/' + key).json()['queued'] and replays == [([{'text': 'hi'}], Host.replay_lane)]
    def busy(examples, lane): raise RuntimeError('Example replay is already running')
    Host.probe = SimpleNamespace(replay=busy)
    assert client.post('/api/neural/replay/' + key).status_code == 409


def test_observers_see_each_generation_and_their_failures_never_fail_it(tmp_path, monkeypatch):
    provider, _ = native_provider(tmp_path, monkeypatch, {})
    seen = []
    class Observer:
        def on_start(self, generation): seen.append(('start', generation.role, generation.slot, generation.group))
        def on_event(self, generation, event, visible):
            seen.append(('event', visible, len(generation.messages)))
            raise ValueError('a broken observer')
        def on_finish(self, generation): seen.append(('finish', generation.role))
    provider.observers.append(Observer())
    try:
        assert provider.generate([ChatMessage('user', 'Hello')], emotion_turn_id='turn-1').message.content == 'Hi'
        provider.initiative.generate([ChatMessage('user', 'Consider')])
        assert seen[0] == ('start', 'live', 0, 'turn-1') and ('event', 'Hi', 1) in seen and ('finish', 'live') in seen
        assert seen[-1] == ('finish', 'initiative') and any(entry[:3] == ('start', 'initiative', 1) for entry in seen)
    finally: provider.close()


def test_the_probe_hook_captures_only_on_the_capture_slot():
    captured, activated = [], []
    probe = SimpleNamespace(active_group='g', activate=activated.append, capture=lambda *args, **kwargs: captured.append(kwargs), close=lambda: None)
    hook = ProbeHook(lambda identity, idle: probe)
    sample = {'type': 'riko.emotion_probe.sample', 'feature_version': FEATURE_VERSION, 'prefix_bytes': 2, 'features': [0.5] * 256}
    background = SimpleNamespace(role='initiative', slot=1, group='g', cancelled=lambda: False, messages=[ChatMessage('user', 'x')])
    replay = SimpleNamespace(role='probe_replay', slot=0, group='g', cancelled=lambda: False, messages=[ChatMessage('user', 'x')])
    hook.on_start(replay); hook.on_event(replay, sample, 'Hi')  # no probe yet: nothing happens
    assert not captured and not activated
    hook.start({}, lambda: True)
    for generation in (background, replay):
        hook.on_start(generation)
        hook.on_event(generation, sample, 'Hi')
    hook.on_event(replay, {**sample, 'features': [0.5] * 255}, 'Hi')  # a width this build does not parse
    assert activated == ['g'] and len(captured) == 1 and captured[0]['replay'] is True and captured[0]['offset'] == 2


def test_fingerprints_hash_once_per_file_state_and_match_the_original_digest(tmp_path, monkeypatch):
    first, second = tmp_path / 'model-00001-of-00002.gguf', tmp_path / 'model-00002-of-00002.gguf'
    first.write_bytes(b'one'); second.write_bytes(b'two')
    opened, real = [], Path.open
    monkeypatch.setattr(Path, 'open', lambda self, *args, **kwargs: opened.append(self.name) or real(self, *args, **kwargs))
    assert fingerprint([first, second]) == hashlib.sha256(b'onetwo').hexdigest() and opened == [first.name, second.name]
    assert fingerprint([first, second]) == hashlib.sha256(b'onetwo').hexdigest() and len(opened) == 2  # cached
    assert fingerprint([first], names=True) == hashlib.sha256(first.name.encode() + b'one').hexdigest()  # the library digest's form
    second.write_bytes(b'changed')  # another size: hashed again
    assert fingerprint([first, second]) == hashlib.sha256(b'onechanged').hexdigest() and opened[-2:] == [first.name, second.name]


def teacher():
    from process.app_core.emotion.julia import JuliaEmotionEngine
    engine = JuliaEmotionEngine(None)
    engine._load_attempted = True
    engine._model = SimpleNamespace(predict=lambda **kw: {})
    return engine


def small_probe(directory, **options):
    from process.app_core.emotion.probe import EmotionProbe, ProbeConfig
    config = ProbeConfig.from_raw({'hidden_units': [32, 16], 'rank': 8, 'min_samples': 32, 'retrain_every': 32, 'epochs': 2})
    identity = {'gguf_sha256': 'a' * 64, 'server_build': 'b1-test', 'chat_template': 'tpl', 'feature_version': FEATURE_VERSION,
        'type_k': 'f16', 'type_v': 'f16', 'flash_attn': 'auto', 'n_ctx': 8192, 'runtime_fingerprint': 'c' * 64}
    return EmotionProbe(directory, identity, teacher(), config, **options)


def test_the_probe_asks_julia_through_its_public_api_and_keeps_its_data_key(tmp_path):
    engine = teacher()
    assert engine.ensure_loaded() is engine._model and engine.resolved_source is None and engine.fingerprint() is None
    questions = engine.question_set()
    questions['emotion']['criteria'].clear()  # a copy: the engine's questions are unchanged
    assert engine.question_set()['emotion']['criteria']
    probe = small_probe(tmp_path, idle=lambda: False)
    try: assert probe.key == '42cbfc8c099fc7c7b9b35d09cb5dfe435cdd3e2d433517051272b2622fbdbbb5'  # computed before this change: user data still matches
    finally: probe.close()


def test_replay_waits_for_idle_retries_a_preempted_example_and_runs_once_at_a_time(tmp_path):
    idle, calls, release = threading.Event(), [], threading.Event()
    probe = small_probe(tmp_path, idle=idle.is_set)
    class Lane:
        def generate(self, messages, **options):
            calls.append((messages[1].content, options['max_output_tokens'], options['emotion_turn_id'].startswith('replay-')))
            if len(calls) == 1: raise BackgroundPreempted('a live turn took the slot')
            assert release.wait(10)
            return ModelResponse(ChatMessage('assistant', 'ok'))
    try:
        probe.replay([{'text': 'first'}, {'text': ''}, 'not an example', {'text': 'second'}], Lane())
        with pytest.raises(RuntimeError, match='already running'): probe.replay([{'text': 'x'}], Lane())
        assert probe.status()['replaying'] and not calls  # not idle yet
        idle.set()
        release.set()
        assert eventually(lambda: not probe.replaying)
        assert calls == [('first', 96, True), ('first', 96, True), ('second', 96, True)] and probe.error == ''
    finally: release.set(); probe.close()
