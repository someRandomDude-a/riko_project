"""Local wake acknowledgements selected by emotion and current runtime state."""
from copy import deepcopy
import math
from pathlib import Path
import time

from ..desktop.media import resolve_media
from ..events.bus import event_bus

DEFAULTS = {'enabled': True, 'volume': .65, 'max_clip_seconds': 3.0,
            'cooldown_seconds': .5, 'rules': []}
STATES = {'idle', 'thinking', 'speaking', 'tool', 'sleeping', '*'}
AUDIO_TYPES = {'.wav', '.flac', '.ogg'}


def validate_settings(raw):
    if not isinstance(raw, dict): raise ValueError('wake_feedback must be a mapping')
    settings = {**deepcopy(DEFAULTS), **raw}
    if type(settings['enabled']) is not bool: raise ValueError('wake_feedback.enabled must be boolean')
    for key, low, high in [('volume', 0, 1), ('max_clip_seconds', .1, 10), ('cooldown_seconds', 0, 30)]:
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'wake_feedback.{key} must be between {low} and {high}')
    rules = settings['rules']
    if not isinstance(rules, list) or len(rules) > 100: raise ValueError('wake_feedback.rules must be a list of at most 100 rules')
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) - {'emotion', 'state', 'audio', 'animation', 'duration_seconds', 'volume'}:
            raise ValueError('Invalid wake feedback rule fields')
        state = rule.get('state', '*')
        if not isinstance(state, str) or state not in STATES: raise ValueError('Invalid wake feedback model state')
        emotion = rule.get('emotion', '*')
        if not isinstance(emotion, str) or not emotion.strip() or len(emotion) > 64: raise ValueError('Invalid wake feedback emotion')
        if not rule.get('audio') and not rule.get('animation'): raise ValueError('Wake feedback rule needs audio or animation')
        for key, extensions in [('audio', AUDIO_TYPES), ('animation', {'.vrma'})]:
            path = rule.get(key)
            if path is not None and (not isinstance(path, str) or len(path) > 2048 or Path(path).suffix.lower() not in extensions):
                raise ValueError(f'Invalid wake feedback {key} asset')
        for key, low, high in [('duration_seconds', .2, 10), ('volume', 0, 1)]:
            value = rule.get(key, 2.0 if key == 'duration_seconds' else settings['volume'])
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Invalid wake feedback {key}')
    return settings


def select_rule(rules, emotion, model_state):
    matches = [rule for rule in rules if rule.get('emotion', '*').casefold() in {'*', emotion.casefold()}
               and rule.get('state', '*') in {'*', model_state}]
    # Specific state wins over an emotion-only fallback; list order breaks ties.
    return max(matches, key=lambda rule: 2 * (rule.get('state', '*') != '*') + (rule.get('emotion', '*') != '*'), default=None)


class WakeFeedback:
    def __init__(self, config, speech, actions):
        self.config, self.speech, self.actions = config, speech, actions
        self.settings = validate_settings(config.raw.get('wake_feedback', {}))
        self.last_trigger = -math.inf

    def trigger(self, emotion, model_state, *, audio_enabled=True):
        now = time.monotonic()
        if not self.settings['enabled'] or now - self.last_trigger < self.settings['cooldown_seconds']: return False
        rule = select_rule(self.settings['rules'], emotion or 'neutral', model_state)
        if rule is None: return False
        self.last_trigger = now
        payload = {'emotion': emotion or 'neutral', 'model_state': model_state}
        directory = self.config.raw.get('desktop', {}).get('effects_directory', 'effects/greenscreens')
        # Treat each asset independently: a bad sound must not suppress animation.
        if rule.get('animation'):
            try:
                path = resolve_media(self.config.root, rule['animation'], directory, extensions={'.vrma'})
                self.actions.start('wake_animation', {**payload, 'path': str(path)}, duration=rule.get('duration_seconds', 2.0))
            except (ValueError, RuntimeError) as exc:
                event_bus.publish('wake.feedback.error', asset='animation', error=str(exc))
        if audio_enabled and rule.get('audio'):
            try:
                path = resolve_media(self.config.root, rule['audio'], directory, extensions=AUDIO_TYPES)
                self.speech.submit_clip(path, volume=rule.get('volume', self.settings['volume']),
                    max_seconds=self.settings['max_clip_seconds'], context=payload)
            except (ValueError, RuntimeError) as exc:
                event_bus.publish('wake.feedback.error', asset='audio', error=str(exc))
        return True
