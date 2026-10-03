from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


EMOTIONS = (
    "neutral", "joy", "amusement", "affection", "excitement", "sadness",
    "anger", "fear", "surprise", "confusion", "embarrassment", "calm",
)


@dataclass
class EmotionState:
    primary: str = "neutral"
    secondary: str | None = None
    intensity: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = 0.0
    source: str = "julia_1"
    evidence: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turn_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "intensity": self.intensity,
            "valence": self.valence,
            "arousal": self.arousal,
            "confidence": self.confidence,
            "source": self.source,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
            "turn_id": self.turn_id,
        }


@dataclass
class EmotionEvent:
    stream: str
    text_delta: str
    state: EmotionState
