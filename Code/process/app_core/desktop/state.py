from __future__ import annotations

from .activity import ActivityLog
from .effects import EffectsModel
from .geometry import SurfaceGeometry
from .listeners import DesktopEvents
from .presence import PlaybackFlags, Presence
from .whiteboard import WhiteboardCommand, WhiteboardModel  # noqa: F401  WhiteboardCommand's old home, for one release

# The state.snapshot payload Electron reads, in this order: each component contributes its own keys.
SNAPSHOT_KEYS = ('emotion', 'speech', 'mic', 'audio', 'audio_volume', 'sleep', 'effect', 'last_effect', 'tools', 'actions',
    'whiteboard', 'whiteboard_visible', 'whiteboard_clear', 'whiteboard_pages', 'whiteboard_page', 'avatar_geometry', 'displays',
    'whiteboard_geometry', 'board_persistence_error', 'notifications', 'incoming', 'discord')


class DesktopState:
    """The bridge between the model's tools and the desktop UI: a thin facade over its components, which share one
    DesktopEvents. Each owns its lock and emits only after releasing it, so a listener can read them all; snapshot() takes
    each lock in turn and never two at once. factory.create_desktop_state builds them over the data root's files; a
    DesktopState() builds its own, with nothing saved. New code takes the component it needs."""
    # Declared state only (tests/test_declared_attributes.py); the desktop tools' collaborators are in DesktopServices.
    __slots__ = ('events', 'geometry', 'board', 'effects', 'playback', 'activity', 'presence')

    def __init__(self, *, events=None, geometry=None, board=None, effects=None, playback=None, activity=None, presence=None):
        self.events = events = events if events is not None else DesktopEvents()
        self.geometry = geometry if geometry is not None else SurfaceGeometry(events)
        self.board = board if board is not None else WhiteboardModel(events, self.geometry)
        self.effects = effects if effects is not None else EffectsModel(events)
        self.playback = playback if playback is not None else PlaybackFlags(events)
        self.activity = activity if activity is not None else ActivityLog(events)
        self.presence = presence if presence is not None else Presence(events)

    def subscribe(self, listener): return self.events.subscribe(listener)

    def snapshot(self):
        parts = {**self.presence.snapshot(), **self.playback.snapshot(), **self.effects.snapshot(), **self.activity.snapshot(),
                 **self.board.snapshot(), **self.geometry.snapshot()}
        return {key: parts[key] for key in SNAPSHOT_KEYS}

    # What the session, the HTTP routes, the initiative and the animation runtime read and change; each call is the component's.
    mic_enabled = property(lambda self: self.playback.mic_enabled)
    audio_enabled = property(lambda self: self.playback.audio_enabled)
    audio_volume = property(lambda self: self.playback.audio_volume)
    sleep_mode = property(lambda self: self.playback.sleep_mode)
    def toggle_mic(self): return self.playback.toggle_mic()
    def set_mic(self, enabled): return self.playback.set_mic(enabled)
    def toggle_audio(self): return self.playback.toggle_audio()
    def set_audio_volume(self, volume): return self.playback.set_audio_volume(volume)
    def set_sleep(self, enabled=True): return self.playback.set_sleep(enabled)
    def emotion_snapshot(self): return self.presence.emotion_snapshot()
    def set_emotion(self, emotion): self.presence.set_emotion(emotion)
    def set_speech(self, text: str, seconds: float = 12.0): self.presence.set_speech(text, seconds)
    def tool_started(self, name, arguments): return self.activity.tool_started(name, arguments)
    def tool_finished(self, name, result, error=False, activity_id=None): self.activity.tool_finished(name, result, error, activity_id)
    def record_action(self, action): self.activity.record_action(action)
    def notify(self, source, text, level='info'): return self.activity.notify(source, text, level)
    def observe_input(self, source, text, *, message_id, context=None): self.activity.observe_input(source, text, message_id=message_id, context=context)
    def set_discord(self, value): self.activity.set_discord(value)
    def configure_board_store(self, path): self.board.load(path)
    def add_whiteboard(self, kind, payload): return self.board.add(kind, payload)
    def board_page(self, action, page=None): return self.board.pages(action, page)
    def board_result(self, command_id, timeout=2): return self.board.result(command_id, timeout)
    def clear_whiteboard(self): return self.board.clear()
    def set_whiteboard_surface(self, *, visible=None, geometry=None): self.board.set_surface(visible=visible, geometry=geometry)
    def update_geometry(self, target, **geometry): self.geometry.update(target, **geometry)
    def set_displays(self, displays): self.geometry.set_displays(displays)
    def save_avatar_geometry(self): self.geometry.save_avatar_geometry()
    def trigger_effect(self, name, **options): return self.effects.trigger(name, **options)
    def stop_effect(self): self.effects.stop()

    def surface_result(self, surface, command_id, status, error='', bounds=None):
        """The renderer's acknowledgement (POST /api/surfaces/result) for a whiteboard command or the active effect."""
        if surface not in {'whiteboard', 'effect'} or status not in {'rendered', 'playing', 'completed', 'error'}: raise ValueError('Invalid surface result')
        if surface == 'whiteboard': return self.board.acknowledge(command_id, status, error, bounds)
        if bounds is not None: raise ValueError('Invalid bounds')
        return self.effects.acknowledge(command_id, status, error)


_DESKTOP_STATE = DesktopState()


def get_desktop_state() -> DesktopState:
    """Transitional, for one release: a process-wide DesktopState for code that is handed none (a desktop tool built without
    DesktopServices). The backend's own is factory.create_desktop_state's, which desktop_server.create_app hands to everything."""
    return _DESKTOP_STATE
