from __future__ import annotations

from .models import EmotionState


class ExpressionActionBridge:
    """Map Julia-1 emotion decisions into renderer-facing expression events."""

    def __init__(self, on_emotion=None):
        self.on_emotion = on_emotion

    def update(self, state: EmotionState) -> None:
        if self.on_emotion:
            self.on_emotion(state)
