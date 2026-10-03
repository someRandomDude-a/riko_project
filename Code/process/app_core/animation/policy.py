"""Versioned state/intents and rule eligibility; replaceable by learned policies."""
from dataclasses import asdict, dataclass, field


@dataclass
class AnimationState:
    revision: int
    mode: str
    emotion: str = 'neutral'
    intensity: float = .5
    geometry: dict = field(default_factory=dict)
    interaction: dict = field(default_factory=dict)
    bones: list = field(default_factory=list)
    expressions: list = field(default_factory=list)
    voice: dict = field(default_factory=dict)
    version: int = 1

    def as_dict(self): return asdict(self)


@dataclass
class MotionIntent:
    state_revision: int
    intent_id: str
    procedural: str
    source: str = 'rules'
    asset: dict | None = None
    layer: str = 'base'
    transition_seconds: float = .3
    strength: float = .65
    confidence: float = 1.0
    version: int = 1

    def as_dict(self): return asdict(self)


def eligible_intents(state, entries):
    assets = [entry for entry in entries if entry['layer'] == 'base' and entry['loop']
              and state.mode in entry['states'] and (not entry['emotions'] or state.emotion in entry['emotions'])
              and set(entry['mask'] or entry['bones']).issubset(state.bones)
              and set(entry.get('expressions', [])).issubset(state.expressions)]
    assets.sort(key=lambda item: not bool(item['emotions']))
    choices = [{'id': entry['id'], 'description': entry['name'][:120], 'asset': entry, 'procedural': state.mode} for entry in assets]
    choices.append({'id': 'builtin-' + state.mode, 'description': f'Expressive {state.mode} posture', 'procedural': state.mode})
    if state.mode == 'idle' and state.emotion in {'joy', 'amusement', 'excitement', 'love'}:
        choices.append({'id': 'builtin-playful', 'description': 'Playful anime idle', 'procedural': 'playful'})
    return choices


def motion_intent(state, choices, settings, selection=None):
    chosen, source, confidence = choices[0], 'rules', 1.0
    if isinstance(selection, dict):
        candidate = next((item for item in choices if item['id'] == selection.get('choice')), None)
        score = selection.get('confidence', 0)
        if candidate and type(score) in (int, float) and settings['min_confidence'] <= score <= 1:
            chosen, source, confidence = candidate, 'julia_1', score
    asset = chosen.get('asset')
    return MotionIntent(state.revision, chosen['id'], chosen['procedural'], source=source,
        asset=asset, transition_seconds=asset['transition_seconds'] if asset else settings['transition_seconds'],
        strength=state.intensity, confidence=confidence)
