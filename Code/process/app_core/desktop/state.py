from __future__ import annotations

import threading
import time
import uuid
import math
import json
import logging
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable
from .geometry import validate_geometry


@dataclass
class WhiteboardCommand:
    kind: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = 'queued'
    error: str = ''
    page: str = 'page-1'
    bounds: dict = field(default_factory=dict)


class DesktopState:
    """Thread-safe bridge between model tools and the desktop UI."""
    def __init__(self):
        self._lock = threading.RLock()
        self._listeners: list[Callable[[str, Any], None]] = []
        self.whiteboard: list[WhiteboardCommand] = []
        self.whiteboard_visible = False
        self.whiteboard_clear = None
        self.whiteboard_pages = ['page-1']
        self.whiteboard_page = 'page-1'
        self.surface_condition = threading.Condition(self._lock)
        self.last_effect = None
        self.speech_bubble = ""
        self.speech_until = 0.0
        self._speech_timer = None
        self.avatar_geometry = {"x": 0, "y": 0, "width": 480, "height": 720, "screen": 0}
        self.whiteboard_geometry = {"x": 500, "y": 100, "width": 900, "height": 700, "screen": 0}
        self.mic_enabled = True
        self.audio_enabled = True
        self.audio_volume = 1.0
        self.sleep_mode = False
        self.emotion_state = None
        self.active_effect = None
        self.tool_activity = []
        self.actions = []
        self.displays = []
        self._board_file = None
        self.board_persistence_error = ''
        self.notifications = []
        self.incoming = []
        self.discord = {'running': False, 'ready': False, 'status': 'stopped'}

    def subscribe(self, listener):
        with self._lock: self._listeners.append(listener)
        def unsubscribe():
            with self._lock:
                if listener in self._listeners: self._listeners.remove(listener)
        return unsubscribe

    def _emit(self, event, value=None):
        if event.startswith('whiteboard') or event == 'surface_result' and value.get('surface') == 'whiteboard':
            self._save_board()
        for listener in tuple(self._listeners):
            try: listener(event, value)
            except Exception: pass

    def snapshot(self):
        with self._lock:
            return {
                "emotion": self.emotion_state.as_dict() if self.emotion_state else None,
                "speech": self.speech_bubble if self.speech_until > time.time() else "",
                "mic": self.mic_enabled, "audio": self.audio_enabled, "audio_volume": self.audio_volume,
                "sleep": self.sleep_mode, "effect": dict(self.active_effect) if self.active_effect else None,
                'last_effect': dict(self.last_effect) if self.last_effect else None,
                "tools": [dict(item) for item in self.tool_activity],
                "actions": [dict(item) for item in self.actions],
                "whiteboard": [{"id": c.id, "kind": c.kind, "payload": deepcopy(c.payload), 'status': c.status, 'error': c.error, 'page': c.page, 'bounds': dict(c.bounds)} for c in self.whiteboard],
                'whiteboard_visible': self.whiteboard_visible,
                'whiteboard_clear': dict(self.whiteboard_clear) if self.whiteboard_clear else None,
                'whiteboard_pages': list(self.whiteboard_pages), 'whiteboard_page': self.whiteboard_page,
                "avatar_geometry": dict(self.avatar_geometry),
                'displays': deepcopy(self.displays),
                "whiteboard_geometry": dict(self.whiteboard_geometry),
                'board_persistence_error': self.board_persistence_error,
                'notifications': deepcopy(self.notifications),
                'incoming': deepcopy(self.incoming), 'discord': deepcopy(self.discord),
            }

    def configure_board_store(self, path):
        path = Path(path)
        self._board_file = path
        if not path.exists(): return
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            pages = raw['pages']
            if not isinstance(pages, list) or not pages or len(set(pages)) != len(pages) or not all(isinstance(p, str) for p in pages) or raw['page'] not in pages:
                raise ValueError('Invalid pages')
            commands = [WhiteboardCommand(**item) for item in raw['commands']]
            if len({c.id for c in commands}) != len(commands) or any(c.page not in pages or c.kind not in {'text', 'draw', 'image'} or not isinstance(c.payload, dict) for c in commands):
                raise ValueError('Invalid commands')
            def finite(value): return type(value) in (int, float) and math.isfinite(value)
            for command in commands:
                bounds, payload = command.bounds, command.payload
                if not isinstance(bounds, dict) or set(bounds) != {'x','y','width','height'} or not all(finite(v) for v in bounds.values()) or bounds['width'] <= 0 or bounds['height'] <= 0:
                    raise ValueError('Invalid command bounds')
                if command.kind == 'text' and not isinstance(payload.get('text'), str) or command.kind == 'image' and not isinstance(payload.get('path'), str):
                    raise ValueError('Invalid command content')
                if command.kind == 'draw' and (not isinstance(payload.get('points'), list) or not payload['points'] or len(payload['points']) > 10000 or not all(isinstance(p,list) and len(p)==2 and all(finite(v) for v in p) for p in payload['points'])):
                    raise ValueError('Invalid drawing')
                if 'size' in payload and (not finite(payload['size']) or not 1 <= payload['size'] <= 128): raise ValueError('Invalid size')
                if 'width' in payload and (not finite(payload['width']) or not 100 <= payload['width'] <= 2000): raise ValueError('Invalid width')
            validate_geometry('whiteboard', raw.get('geometry', {}))
            with self._lock:
                self.whiteboard_pages, self.whiteboard_page = pages, raw['page']
                self.whiteboard = commands
                self.whiteboard_geometry.update(raw.get('geometry', {}))
                self.whiteboard_visible = bool(raw.get('visible', False))
                for command in commands: command.status, command.error = 'queued', ''
        except (OSError, ValueError, TypeError, KeyError):
            # Preserve the damaged original while allowing continued use/recovery.
            self._board_file = path.with_name(path.stem + '.recovery.json')
            if self._board_file.exists() and not path.stem.endswith('.recovery'):
                self.configure_board_store(self._board_file)
            self.board_persistence_error = f'Invalid saved board preserved at {path.name}; new state saves to {self._board_file.name}'

    def notify(self, source, text, level='info'):
        item = {'id': str(uuid.uuid4()), 'source': source, 'text': str(text)[:500], 'level': level, 'timestamp': time.time()}
        with self._lock: self.notifications = [item, *self.notifications[:19]]
        self._emit('notification', item)
        return item

    def observe_input(self, source, text, *, message_id, context=None):
        if source not in {'discord', 'microphone', 'message'}: raise ValueError('Unknown input source')
        item = {'id': f'{source}:{message_id}', 'source': source, 'text': str(text)[:1000], 'timestamp': time.time(), **(context or {})}
        with self._lock:
            if any(entry['id'] == item['id'] for entry in self.incoming): return
            self.incoming = [item, *self.incoming[:19]]
        self._emit('input_received', item)

    def set_discord(self, value):
        with self._lock: self.discord = deepcopy(value)
        self._emit('discord', value)

    def _save_board(self):
        if self._board_file is None: return
        try:
            with self._lock:
                snapshot = self.snapshot()
                raw = {'version': 1, 'pages': self.whiteboard_pages, 'page': self.whiteboard_page,
                       'commands': snapshot['whiteboard'], 'geometry': self.whiteboard_geometry,
                       'visible': self.whiteboard_visible}
                self._board_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._board_file.with_suffix('.tmp')
                temporary.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
                temporary.replace(self._board_file)
        except OSError as exc:
            self.board_persistence_error = str(exc)
            logging.getLogger(__name__).exception('Unable to persist whiteboard')

    def set_speech(self, text: str, seconds: float = 12.0):
        with self._lock:
            if self._speech_timer: self._speech_timer.cancel()
            self.speech_bubble, self.speech_until = text, time.time() + seconds
            self._speech_timer = None
            if text and seconds > 0:
                self._speech_timer = threading.Timer(seconds, self._expire_speech, args=(self.speech_until,))
                self._speech_timer.daemon = True
                self._speech_timer.start()
        self._emit("speech", text)

    def _expire_speech(self, deadline):
        with self._lock:
            if self.speech_until != deadline: return
            self.speech_bubble = ''
            self._speech_timer = None
        self._emit('speech', '')

    def add_whiteboard(self, kind: str, payload: dict[str, Any]):
        payload = deepcopy(payload)
        command = WhiteboardCommand(kind, payload)
        with self._lock:
            command.page = self.whiteboard_page
            if kind == 'draw':
                xs, ys = zip(*payload['points'])
                stroke = payload.get('size', 6)
                command.bounds = {'x': min(xs)-stroke/2, 'y': min(ys)-stroke/2, 'width': max(1, max(xs)-min(xs))+stroke, 'height': max(1, max(ys)-min(ys))+stroke}
            else:
                x = payload.get('x')
                y = payload.get('y')
                if x is None: x = 40
                payload['auto_place'] = y is None
                if y is None: y = max((c.bounds.get('y', 0) + c.bounds.get('height', 100) + 24 for c in self.whiteboard if c.page == command.page), default=40)
                payload.update(x=x, y=y)
                command.bounds = {'x': x, 'y': y, 'width': payload.get('width', 420), 'height': 120}
            self.whiteboard.append(command)
            self.whiteboard_visible = True
        self._emit("whiteboard", command)
        return command.id

    def board_page(self, action, page=None):
        if action not in {'pages', 'new_page', 'page', 'next_page', 'previous_page'}:
            raise ValueError('Unknown page action')
        with self._lock:
            if action == 'new_page':
                page = f'page-{len(self.whiteboard_pages)+1}'
                self.whiteboard_pages.append(page)
            if action in {'new_page', 'page'}:
                if page not in self.whiteboard_pages: raise ValueError('Unknown page')
                self.whiteboard_page = page
            elif action in {'next_page', 'previous_page'}:
                index = self.whiteboard_pages.index(self.whiteboard_page) + (1 if action == 'next_page' else -1)
                self.whiteboard_page = self.whiteboard_pages[max(0, min(len(self.whiteboard_pages)-1, index))]
            self.whiteboard_visible = True
            result = {'pages': list(self.whiteboard_pages), 'current_page': self.whiteboard_page}
        self._emit('whiteboard_page', self.whiteboard_page)
        return result

    def board_result(self, command_id, timeout=2):
        with self.surface_condition:
            self.surface_condition.wait_for(lambda: not any(c.id == command_id and c.status == 'queued' for c in self.whiteboard), timeout=timeout)
            return next(({'id': c.id, 'page': c.page, 'bounds': dict(c.bounds), 'status': c.status, 'error': c.error} for c in self.whiteboard if c.id == command_id), {'id': command_id, 'status': 'removed'})

    def clear_whiteboard(self):
        with self._lock:
            self.whiteboard.clear()
            self.whiteboard_clear = {'id': str(uuid.uuid4()), 'status': 'queued', 'error': ''}
            command_id = self.whiteboard_clear['id']
            self.surface_condition.notify_all()
        self._emit("whiteboard_clear", None)
        return command_id

    def surface_result(self, surface, command_id, status, error='', bounds=None):
        if surface not in {'whiteboard', 'effect'} or status not in {'rendered', 'playing', 'completed', 'error'}: raise ValueError('Invalid surface result')
        if bounds is not None:
            if surface != 'whiteboard' or not isinstance(bounds, dict) or set(bounds) != {'x', 'y', 'width', 'height'}:
                raise ValueError('Invalid bounds')
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds.values()) or bounds['width'] <= 0 or bounds['height'] <= 0:
                raise ValueError('Invalid bounds')
        with self._lock:
            if surface == 'whiteboard':
                item = next((c for c in self.whiteboard if c.id == command_id), None)
                if status not in {'rendered', 'error'}: raise ValueError('Invalid whiteboard result')
                if item is None:
                    if not self.whiteboard_clear or self.whiteboard_clear['id'] != command_id: return False
                    if self.whiteboard_clear['status'] == status and self.whiteboard_clear['error'] == error: return True
                    self.whiteboard_clear.update(status=status, error=error)
                else:
                    dimensions_changed = bounds is not None and any(bounds[key] != item.bounds[key] for key in ('width', 'height'))
                    if item.status == status and item.error == error and not dimensions_changed: return True
                    item.status, item.error = status, error
                    if dimensions_changed:
                        # User moves/pan are local; only original-layout dimensions
                        # come back. Reflow later auto-placed items after font/image
                        # measurement changes so queued elements cannot overlap.
                        item.bounds.update(width=bounds['width'], height=bounds['height'])
                        bottom = item.bounds['y'] + item.bounds['height'] + 24
                        for later in self.whiteboard[self.whiteboard.index(item)+1:]:
                            if later.page != item.page: continue
                            if later.payload.get('auto_place') and later.bounds['y'] < bottom:
                                later.bounds['y'] = later.payload['y'] = bottom
                            bottom = max(bottom, later.bounds['y'] + later.bounds['height'] + 24)
                self.surface_condition.notify_all()
            else:
                item = self.active_effect
                if not item or item['id'] != command_id: return False
                if status not in {'playing', 'completed', 'error'}: raise ValueError('Invalid effect result')
                item['status'], item['error'] = status, error
                if status in {'completed', 'error'}:
                    self.last_effect = dict(item)
                    self.active_effect = None
        self._emit('surface_result', {'surface': surface, 'id': command_id, 'status': status, 'error': error})
        return True

    def set_whiteboard_surface(self, *, visible=None, geometry=None):
        with self._lock:
            if geometry is not None: validate_geometry('whiteboard', geometry, self.displays)
            if visible is not None: self.whiteboard_visible = visible
            if geometry: self.whiteboard_geometry.update(geometry)
        self._emit('whiteboard_surface', self.whiteboard_geometry)

    def update_geometry(self, target: str, **geometry):
        with self._lock:
            validate_geometry(target, geometry, self.displays)
            target_geometry = self.avatar_geometry if target == "avatar" else self.whiteboard_geometry
            target_geometry.update(geometry)
        self._emit(f"{target}_geometry", dict(target_geometry))

    def toggle_mic(self): self.mic_enabled = not self.mic_enabled; self._emit("mic", self.mic_enabled); return self.mic_enabled
    def toggle_audio(self): self.audio_enabled = not self.audio_enabled; self._emit("audio", self.audio_enabled); return self.audio_enabled
    def set_audio_volume(self, volume):
        if type(volume) not in (int, float) or not math.isfinite(volume) or not 0 <= volume <= 1:
            raise ValueError('Audio volume must be a finite number between 0 and 1')
        with self._lock: self.audio_volume = float(volume)
        self._emit('audio_volume', self.audio_volume)
        return self.audio_volume
    def set_sleep(self, enabled=True): self.sleep_mode = enabled; self._emit("sleep", enabled); return enabled

    def set_emotion(self, emotion): self.emotion_state = emotion; self._emit("emotion", emotion)

    def trigger_effect(self, name: str, *, opacity: float = 0.65, brightness: float = 1.0, duration: float = 8.0, asset: str | None = None):
        with self._lock:
            self.active_effect = {'id': str(uuid.uuid4()), 'status': 'queued', 'error': '', "name": name, "opacity": max(0.0, min(1.0, opacity)), "brightness": max(0.0, brightness), "asset": asset, 'duration': duration}
        self._emit("effect", self.active_effect)
        return self.active_effect['id']

    def stop_effect(self):
        with self._lock:
            if self.active_effect: self.last_effect = {**self.active_effect, 'status': 'cancelled'}
            self.active_effect = None
        self._emit("effect", None)

    def tool_started(self, name: str, arguments: dict):
        item = {"id": str(uuid.uuid4()), "name": name, "arguments": arguments, "status": "running", 'started_at': time.time()}
        with self._lock: self.tool_activity = [item, *self.tool_activity[:19]]
        self._emit("tool", item)
        return item['id']

    def tool_finished(self, name: str, result, error: bool = False, activity_id=None):
        with self._lock:
            for item in self.tool_activity:
                if item["name"] == name and item["status"] == "running" and (activity_id is None or item['id'] == activity_id):
                    item["status"] = "error" if error else "complete"; item["result"] = str(result)[:8000]
                    item['result_truncated'] = len(str(result)) > 8000
                    item['finished_at'] = time.time()
                    item['duration_ms'] = round((item['finished_at'] - item['started_at']) * 1000)
                    break
        self._emit("tool", self.tool_activity)


_DESKTOP_STATE = DesktopState()


def get_desktop_state() -> DesktopState:
    return _DESKTOP_STATE
