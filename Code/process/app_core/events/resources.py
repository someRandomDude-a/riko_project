"""Snapshots originate at mutation events, never periodic status requests."""
import json
import threading


class ResourceEvents:
    def __init__(self, bus, getters):
        self.bus, self.getters = bus, getters
        self.lock = threading.RLock()
        self.previous = {}
        self.unsubscribe = bus.subscribe(self.observe)

    def snapshot(self):
        result = {}
        for topic, getter in self.getters.items():
            try: result[topic] = getter()
            except Exception: result[topic] = None # Runtime may be starting/stopping.
        return result

    def emit(self, topic):
        try: value = self.getters[topic]()
        except Exception: return
        key = json.dumps(value, sort_keys=True, default=str)
        with self.lock:
            if self.previous.get(topic) == key: return
            self.previous[topic] = key
            self.bus.publish('resource.' + topic, **value)

    def observe(self, event):
        kind = event.type
        if kind.startswith('resource.'): return
        if kind.startswith('tool.approval'): self.emit('approvals')
        elif kind.startswith(('initiative.', 'environment.')): self.emit('initiative')
        elif kind.startswith('animation.'): self.emit('animation')
        elif kind == 'avatar.models_changed': self.emit('avatar_models')
        elif kind == 'task.changed': self.emit('tasks')
        elif kind in {'voice.starting','voice.ready','voice.stopped','voice.error','voice.wake_status','voice.activated','voice.follow_up','voice.started','voice.resumed','voice.utterance_ended','voice.transcribing','voice.transcript','voice.waiting'}: self.emit('voice')
        elif kind == 'runtime.ready':
            for topic in self.getters: self.emit(topic)

    def close(self): self.unsubscribe()
