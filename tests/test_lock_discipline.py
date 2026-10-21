"""Bus listeners run on the publisher's thread and take SessionManager._voice_lock, so
publishing (or running callbacks) while holding any other lock creates lock-order deadlocks."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace

from process.app_core.conversation.chat import ChatDeps
from process.app_core.conversation.history import ConversationHistory
from process.app_core.desktop.state import DesktopState
from process.app_core.events.bus import EventBus, event_bus
from process.app_core.events.outbox import Outbox
from process.app_core.events.resources import ResourceEvents
from process.app_core.runtime.actions import ActionController
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths

ROOT = Path(__file__).resolve().parents[1] / 'Code'
EMITTERS = {'publish', 'emit', '_emit', 'notify', 'on_prediction', 'on_fallback', 'callback'}
# _voice_lock is the outermost runtime lock; _capture_lock is taken only by request handlers above it.
ALLOWED = {('process/app_core/runtime/session.py', 'self._voice_lock'), ('process/app_core/runtime/session.py', 'self._capture_lock')}


def emitting_calls(node):
    """Calls in a with-body, skipping nested functions (they run later, without the lock)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)): continue
        if isinstance(child, ast.Call):
            function = child.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', '')
            receiver = ast.unparse(function.value) if isinstance(function, ast.Attribute) else ''
            if name in EMITTERS and not (name == 'notify' and 'cond' in receiver.lower()): yield child
        yield from emitting_calls(child)


def test_nothing_publishes_or_calls_back_while_holding_a_component_lock():
    violations = []
    for path in sorted(ROOT.rglob('*.py')):
        relative = path.relative_to(ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.With): continue
            for item in node.items:
                lock = ast.unparse(item.context_expr)
                if not any(word in lock.lower() for word in ('lock', 'condition')) or (relative, lock) in ALLOWED: continue
                violations += [f'{relative}:{call.lineno} {ast.unparse(call)[:80]} (under {lock})' for call in emitting_calls(node)]
    assert not violations, 'Publish after releasing the lock (see events/outbox.py):\n' + '\n'.join(violations)


def run_both(first, second, timeout=5):
    threads = [threading.Thread(target=first, daemon=True), threading.Thread(target=second, daemon=True)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout)
    return [thread.is_alive() for thread in threads]


def test_wake_calibration_timeout_does_not_deadlock_with_runtime_snapshot(monkeypatch):
    """Regression: WakeWord.feed published under WakeWord.lock while runtime_snapshot held _voice_lock."""
    import process.app_core.runtime.session as session_module
    from process.app_core.audio.wake_capture import WakeCapture

    class FakeSpeech:
        def __init__(self, *args): pass
        def submit(self, *args): return False
        def submit_clip(self, *args, **kwargs): return False
        def cancel(self): pass
        def close(self): pass

    in_publish, voice_held = threading.Event(), threading.Event()
    def gate(event):
        if event.type == 'voice.calibration_discarded' and threading.current_thread().name == 'vad':
            in_publish.set()
            voice_held.wait(2)
    unsubscribe = event_bus.subscribe(gate)
    monkeypatch.setattr(session_module, 'SpeechQueue', FakeSpeech)
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), provider=BaseProvider())
    config = SimpleNamespace(raw={}, root=Path('.'), paths=DataPaths.at(Path('.')), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    session = session_module.SessionManager(config, chat, DesktopState())
    session.wake.recording = WakeCapture(max_seconds=0.01)  # next frame times out the sample
    def vad():
        threading.current_thread().name = 'vad'
        session.wake.feed(bytes(1024), False)
    def http():  # what runtime_snapshot does: _voice_lock, then wake.status()
        in_publish.wait(2)
        with session._voice_lock:
            voice_held.set()
            session.wake.status()
    try: assert run_both(vad, http) == [False, False]
    finally:
        unsubscribe()
        try: session.close()
        except Exception: pass


def test_action_events_do_not_deadlock_with_a_caller_holding_the_voice_lock():
    """Regression: ActionController published under its lock while barge-in held _voice_lock."""
    bus_listener_started, voice_held = threading.Event(), threading.Event()
    voice_lock = threading.RLock()
    actions = ActionController()
    def listener(event):  # like SessionManager._playback_event / desktop snapshots
        if event.type == 'action.started' and threading.current_thread().name == 'animation':
            bus_listener_started.set()
            voice_held.wait(2)
            with voice_lock: pass
    unsubscribe = event_bus.subscribe(listener)
    def animation():
        threading.current_thread().name = 'animation'
        actions.start('motion', {'name': 'idle'})
    def barge_in():  # SessionManager.cancel under _voice_lock asks the controller for active actions
        bus_listener_started.wait(2)
        with voice_lock:
            voice_held.set()
            actions.active()
    try: assert run_both(animation, barge_in) == [False, False]
    finally:
        unsubscribe()
        actions.close()


def test_outbox_delivers_in_queue_order_even_when_another_thread_is_draining():
    bus, seen = EventBus(), []
    outbox = Outbox(bus)
    release = threading.Event()
    def slow(event):
        seen.append(event.payload['n'])
        if event.payload['n'] == 0: release.wait(2)
    bus.subscribe(slow)
    outbox.put('step', n=0)
    drainer = threading.Thread(target=outbox.flush, daemon=True)
    drainer.start()
    outbox.put('step', n=1)
    outbox.flush()  # returns at once: the drainer owns delivery and will publish n=1 next
    release.set()
    drainer.join(2)
    assert seen == [0, 1]


def test_actions_publish_started_then_cancelled_in_order():
    seen = []
    unsubscribe = event_bus.subscribe(lambda event: seen.append(event.type) if event.type.startswith('action.') else None)
    actions = ActionController()
    try:
        first = actions.start('gesture', {'name': 'nod'})
        actions.start('gesture', {'name': 'wave'})  # same lane: cancels the first
        actions.cancel(first.id)  # already cancelled: no event
        actions.close()  # cancels the second
    finally: unsubscribe()
    assert seen == ['action.started', 'action.cancelled', 'action.started', 'action.cancelled']


def test_resource_emit_never_publishes_a_superseded_value_last():
    bus, published = EventBus(), []
    bus.subscribe(lambda event: published.append(event.payload['value']) if event.type == 'resource.voice' else None)
    first_computing, newer_done = threading.Event(), threading.Event()
    values = iter(['old', 'new'])
    def getter():
        value = next(values)
        if value == 'old':
            first_computing.set()
            newer_done.wait(2)  # the older computation finishes after the newer one
        return {'value': value}
    bridge = ResourceEvents(bus, {'voice': getter})
    try:
        older = threading.Thread(target=bridge.emit, args=('voice',), daemon=True)
        older.start()
        first_computing.wait(2)
        bridge.emit('voice')
        newer_done.set()
        older.join(2)
    finally: bridge.close()
    assert published == ['new']
