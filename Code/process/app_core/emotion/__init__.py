"""Emotion interpretation and future avatar-expression boundaries."""

from .julia import JuliaEmotionEngine
from .models import EmotionState, EmotionEvent
from .actions import ExpressionActionBridge

__all__ = ["JuliaEmotionEngine", "EmotionState", "EmotionEvent", "ExpressionActionBridge"]
