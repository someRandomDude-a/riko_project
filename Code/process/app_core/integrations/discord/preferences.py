"""Small Discord-only preferences; runtime/model settings remain in SettingsStore."""
import json

from ...persistence.atomic import atomic_write


class Preferences:
    def __init__(self, path):
        self.path, self.channels, self.initiative_channel = path, {}, None
        if not path.exists(): return
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or not isinstance(value.get('channels', {}), dict): raise ValueError('Invalid Discord preferences; file preserved')
        for key, options in value.get('channels', {}).items():
            if not str(key).isdigit() or not isinstance(options, dict) or any(name not in {'audio', 'reasoning'} or type(enabled) is not bool for name, enabled in options.items()):
                raise ValueError('Invalid Discord channel preferences; file preserved')
            self.channels[str(key)] = dict(options)
        target = value.get('initiative_channel')
        if target is not None and (type(target) is not int or target <= 0): raise ValueError('Invalid initiative channel')
        self.initiative_channel = target

    def get(self, channel_id, name): return self.channels.get(str(channel_id), {}).get(name, False)

    def save(self):
        atomic_write(self.path, json.dumps({'channels': self.channels, 'initiative_channel': self.initiative_channel}, indent=2))

    def set(self, channel_id, name, enabled):
        if name not in {'audio', 'reasoning'} or type(enabled) is not bool: raise ValueError('Invalid Discord preference')
        self.channels.setdefault(str(channel_id), {})[name] = enabled
        self.save()
