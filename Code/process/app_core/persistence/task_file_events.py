"""One OS/file sampler bridges optional external SQLite writers to subscriptions."""
import threading


class TaskFileEvents:
    def __init__(self, store, bus, *, start=True):
        self.store, self.bus = store, bus
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.signature = self.files()
        self.version = store.change_version()
        store.events_managed = True
        self.unsubscribe = bus.subscribe(self.observe)
        self.thread = threading.Thread(target=self.run, daemon=True, name='task-file-events')
        if start: self.thread.start()

    def files(self):
        result = []
        for path in (self.store.path, self.store.path.with_name(self.store.path.name + '-wal')):
            try:
                stat = path.stat()
                result.append((stat.st_mtime_ns, stat.st_size))
            except OSError: result.append(None)
        return tuple(result)

    def observe(self, event):
        if event.type != 'task.changed': return
        with self.lock:
            self.version = event.payload.get('change_version') or self.store.change_version()

    def sample(self):
        with self.lock:
            current = self.files()
            if current == self.signature: return
            if not self.store.path.exists(): return
            version = self.store.change_version()
            self.signature = current
            if version == self.version or self.stop.is_set(): return
            self.version = version
        self.bus.publish('task.changed', external=True, change_version=version)

    def run(self):
        while not self.stop.wait(2):
            try: self.sample()
            except Exception: pass # A locked/replaced DB is retried at the OS boundary.

    def close(self):
        self.stop.set()
        self.unsubscribe()
        self.store.events_managed = False
        if self.thread.is_alive(): self.thread.join(.5)
