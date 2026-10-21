import threading

from process.app_core.events.bus import EventBus
from process.app_core.events.resources import ResourceEvents


def test_resource_updates_come_from_mutation_events_and_suppress_duplicates():
    bus = EventBus()
    state = {'busy': False}
    events = []
    bus.subscribe(events.append)
    bridge = ResourceEvents(bus, {'initiative': lambda: dict(state)})
    try:
        assert bridge.snapshot()['initiative'] == state
        assert not events
        bus.publish('initiative.started'); bridge.settle()
        assert events[-1].type == 'resource.initiative'
        total = sum(e.type == 'resource.initiative' for e in events)
        bus.publish('initiative.started'); bridge.settle()
        assert sum(e.type == 'resource.initiative' for e in events) == total
        state['busy'] = True
        bus.publish('initiative.started'); bridge.settle()
        assert events[-1].payload == {'busy': True}
    finally: bridge.close()
    state['busy'] = False
    bus.publish('initiative.finished')
    assert events[-1].type == 'initiative.finished'


def test_initial_resources_tolerate_unavailable_runtime():
    def unavailable(): raise RuntimeError('Not started')
    bridge = ResourceEvents(EventBus(), {'voice': unavailable})
    try: assert bridge.snapshot() == {'voice': None}
    finally: bridge.close()


def test_a_publisher_holding_a_component_lock_never_runs_a_getter():
    """Regression: WakeWord publishes voice.wake_status under WakeWord.lock, and the voice getter (runtime_snapshot) takes
    _voice_lock, which respond() holds while it calls into WakeWord: the bridge computed it on the publisher's thread."""
    bus, published, component = EventBus(), [], threading.Lock()
    bus.subscribe(lambda event: published.append((event.payload, threading.current_thread().name)) if event.type == 'resource.voice' else None)
    def voice():
        with component: return {'phase': 'listening'}  # needs the lock the publisher holds
    bridge = ResourceEvents(bus, {'voice': voice})
    try:
        returned = threading.Event()
        def wake_word():
            with component:
                bus.publish('voice.wake_status')  # synchronously calling voice() here would deadlock on the plain Lock
                returned.set()
        thread = threading.Thread(target=wake_word, daemon=True)
        thread.start()
        assert returned.wait(5)
        thread.join(5)
        bridge.settle()
        assert published == [({'phase': 'listening'}, 'resource-events-0')]  # computed and published on the bridge's worker
    finally: bridge.close()


def test_events_while_a_getter_computes_coalesce_and_the_last_value_wins():
    bus, published = EventBus(), []
    bus.subscribe(lambda event: published.append(event.payload['value']) if event.type == 'resource.tasks' else None)
    computing, release, values = threading.Event(), threading.Event(), iter(range(100))
    def tasks():
        value = next(values)
        if value == 0: computing.set(); release.wait(5)
        return {'value': value}
    bridge = ResourceEvents(bus, {'tasks': tasks})
    try:
        bus.publish('task.changed')
        assert computing.wait(5)
        for _ in range(50): bus.publish('task.changed')  # one emit waits for these; the queue never fills
        release.set()
        bridge.settle()
        assert published == [0, 1]  # the first reading, then one more after every later change
        bridge.close()
        bus.publish('task.changed')
        assert published == [0, 1]  # closed: nothing is computed or published
    finally: bridge.close()
