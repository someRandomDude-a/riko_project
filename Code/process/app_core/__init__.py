"""Provider-neutral application core for Riko.

The names below load on first use (PEP 562), so importing a leaf module (a store, a kernel utility, the tool worker)
no longer imports the whole core and its failure modes with it. Inside app_core, import from the defining module.
This root also names the packages that declare settings (_declare_settings), the one place allowed to know them all.
"""
from importlib import import_module
from typing import TYPE_CHECKING

from .kernel import schema as _schema

_EXPORTS = {
    "AppConfig": "configuration.config", "load_config": "configuration.config",
    "ChatMessage": "kernel.messages", "ModelResponse": "kernel.messages", "ToolCall": "kernel.messages", "ToolResult": "kernel.messages",
    "ModelProvider": "inference.provider", "InferenceProvider": "inference.provider", "BaseProvider": "inference.provider",
    "ProviderCapabilities": "inference.provider", "ChatService": "conversation.chat",
    "JuliaEmotionEngine": "emotion", "EmotionState": "emotion", "ExpressionActionBridge": "emotion",
    "RuntimeEvent": "events.bus", "event_bus": "events.bus", "SessionManager": "runtime.session",
    "ActionController": "runtime.actions", "CompanionAction": "runtime.actions",
}
__all__ = list(_EXPORTS)

if TYPE_CHECKING:
    from .configuration.config import AppConfig, load_config
    from .kernel.messages import ChatMessage, ModelResponse, ToolCall, ToolResult
    from .inference.provider import BaseProvider, InferenceProvider, ModelProvider, ProviderCapabilities
    from .conversation.chat import ChatService
    from .emotion import JuliaEmotionEngine, EmotionState, ExpressionActionBridge
    from .events.bus import RuntimeEvent, event_bus
    from .runtime.session import SessionManager
    from .runtime.actions import ActionController, CompanionAction


def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = globals()[name] = getattr(import_module("." + _EXPORTS[name], __name__), name)
    return value


def __dir__(): return sorted({*globals(), *__all__})


def _declare_settings():
    """Import every module that registers a settings section. kernel/schema.py runs this once, the first time load_config or
    Settings reads the registry, so it is complete wherever load_config runs (the backend, a test or a script), with
    nothing else imported first. Configuration imports none of these packages."""
    from .configuration import schema  # noqa: F401  the top level, memory, emotion, tools, tasks, logging, desktop, avatar, presets
    from .kernel import audio_config  # noqa: F401  voice, speech, sovits_ping_config
    from .inference import settings  # noqa: F401  runtime
    from .audio import asr, wake_feedback  # noqa: F401  voice.asr_*, wake_feedback
    from .animation import library  # noqa: F401
    from .emotion import probe  # noqa: F401  emotion.probe
    from .runtime import initiative  # noqa: F401
    from .desktop.avatar_models import review_settings  # desktop imports nothing but kernel from app_core, so its check is added here
    _schema.register(_schema.Section('avatar', review=review_settings))


_schema.loader(_declare_settings)
