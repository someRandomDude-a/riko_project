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
        bus.publish('initiative.started')
        assert events[-1].type == 'resource.initiative'
        total = sum(e.type == 'resource.initiative' for e in events)
        bus.publish('initiative.started')
        assert sum(e.type == 'resource.initiative' for e in events) == total
        state['busy'] = True
        bus.publish('initiative.started')
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
