from process.app_core.events.bus import EventBus
from process.app_core.persistence.tasks import TaskStore
from process.app_core.persistence.task_file_events import TaskFileEvents


def test_external_file_changes_push_once_without_initiative(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    watcher = TaskFileEvents(store, bus, start=False)
    try:
        assert store.events_managed
        store.create('External task', actor='external', reason='test')
        watcher.sample()
        assert len(events) == 1
        assert events[0].type == 'task.changed'
        assert events[0].payload['external']
        watcher.sample()
        assert len(events) == 1
    finally: watcher.close()
    assert not store.events_managed


def test_local_mutation_event_prevents_duplicate_file_notifications(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    watcher = TaskFileEvents(store, bus, start=False)
    try:
        task = store.create('Local task', actor='user', reason='test')
        bus.publish('task.changed', task=task)
        watcher.sample()
        assert len(events) == 1
    finally: watcher.close()
