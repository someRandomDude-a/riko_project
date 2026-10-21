from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any
import uuid


EMOTION_THEMES = {
    "neutral": {"hue": "#8b93a6", "effect": "none"},
    "joy": {"hue": "#ffd166", "effect": "stars"},
    "amusement": {"hue": "#ffcf70", "effect": "sparkles"},
    "affection": {"hue": "#ff8fab", "effect": "hearts"},
    "love": {"hue": "#ff4d8d", "effect": "hearts"},
    "excitement": {"hue": "#ff8c42", "effect": "bursts"},
    "sadness": {"hue": "#5dade2", "effect": "rain"},
    "anger": {"hue": "#ef476f", "effect": "flames"},
    "fear": {"hue": "#9b5de5", "effect": "shiver"},
    "surprise": {"hue": "#00bbf9", "effect": "rays"},
    "confusion": {"hue": "#a0c4ff", "effect": "question_marks"},
    "embarrassment": {"hue": "#f28482", "effect": "blush"},
    "calm": {"hue": "#80ed99", "effect": "breathe"},
}


@dataclass
class EffectRule:
    name: str
    emotion: str
    asset: str | None = None
    min_intensity: float = 0.0
    min_confidence: float = 0.0
    duration_seconds: float = 8.0
    opacity: float = 0.65
    brightness: float = 1.0
    enabled: bool = True
    conditions: list[dict[str, Any]] = field(default_factory=list)


class EffectLibrary:
    """Discovers greenscreen videos and stores visual emotion rules."""
    EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v"}

    def __init__(self, directory: str | Path, rules: list[EffectRule] | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.assets = self.discover()
        rules_path = self.directory / "effects.rules.json"
        if rules is None and rules_path.exists():
            try:
                rules = [EffectRule(**item) for item in json.loads(rules_path.read_text(encoding="utf-8"))]
            except (OSError, ValueError, TypeError):
                rules = None
        self.rules = self.default_rules() if rules is None else rules

    def discover(self) -> dict[str, list[Path]]:
        assets: dict[str, list[Path]] = {}
        for path in self.directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in self.EXTENSIONS: continue
            match = re.search(r"(love|joy|happy|amusement|affection|excitement|sadness|sad|anger|angry|fear|surprise|confusion|embarrassment|calm|neutral)", path.stem.lower())
            name = match.group(1) if match else "neutral"
            emotion = {"happy": "joy", "sad": "sadness", "angry": "anger"}.get(name, name)
            assets.setdefault(emotion, []).append(path)
        return assets

    def default_rules(self) -> list[EffectRule]:
        return [EffectRule(name=f"{emotion}-default", emotion=emotion, asset=str(paths[0]) if paths else None)
                for emotion, paths in self.assets.items()]

    def rules_json(self) -> list[dict[str, Any]]:
        return [{**rule.__dict__} for rule in self.rules]

    def save_rules(self, path: Path):
        path.write_text(json.dumps(self.rules_json(), indent=2), encoding="utf-8")

    def active_rule(self, emotion: str, intensity: float, confidence: float) -> EffectRule | None:
        candidates = [r for r in self.rules if r.enabled and r.emotion == emotion and intensity >= r.min_intensity and confidence >= r.min_confidence]
        return max(candidates, key=lambda rule: rule.min_intensity, default=None)

    def find_asset(self, name: str) -> Path | None:
        """A discovered asset named name (file name or stem), else name in the library's folder when it is a bare file
        name. The model chose name, so nothing here touches the filesystem; resolve_media checks the result."""
        for paths in self.assets.values():
            for path in paths:
                if path.name == name or path.stem == name: return path
        return self.directory / name if isinstance(name, str) and name and Path(name).name == name and '\\' not in name else None


class EffectsModel:
    """The overlay's video effect: the one playing (or queued) and the last to finish, with the renderer's acknowledgements.
    Emits 'effect' and the effect's 'surface_result'. Which effects exist (EffectLibrary) and which files may play
    (resolve_media) are the desktop tools' collaborators (DesktopServices)."""
    __slots__ = ('events', '_lock', '_active', '_last')

    def __init__(self, events):
        self.events, self._lock = events, threading.Lock()
        self._active = self._last = None

    def snapshot(self):
        with self._lock: return {'effect': dict(self._active) if self._active else None, 'last_effect': dict(self._last) if self._last else None}

    def trigger(self, name: str, *, opacity: float = 0.65, brightness: float = 1.0, duration: float = 8.0, asset: str | None = None):
        effect = {'id': str(uuid.uuid4()), 'status': 'queued', 'error': '', "name": name, "opacity": max(0.0, min(1.0, opacity)), "brightness": max(0.0, brightness), "asset": asset, 'duration': duration}
        with self._lock: self._active = effect
        self.events.emit("effect", dict(effect))
        return effect['id']

    def stop(self):
        with self._lock:
            if self._active: self._last = {**self._active, 'status': 'cancelled'}
            self._active = None
        self.events.emit("effect", None)

    def acknowledge(self, command_id, status, error=''):
        """The renderer's result for the active effect: playing, completed or error; False for any other (a stale) effect."""
        with self._lock:
            item = self._active
            if not item or item['id'] != command_id: return False
            if status not in {'playing', 'completed', 'error'}: raise ValueError('Invalid effect result')
            item['status'], item['error'] = status, error
            if status in {'completed', 'error'}:
                self._last = dict(item)
                self._active = None
        self.events.emit('surface_result', {'surface': 'effect', 'id': command_id, 'status': status, 'error': error})
        return True
