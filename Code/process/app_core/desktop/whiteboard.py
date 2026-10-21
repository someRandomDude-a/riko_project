"""The paged whiteboard the model writes on: its commands, the pending clear, the renderer's acknowledgements (result()
waits for them) and whiteboard.json, which also keeps the whiteboard window's visibility and geometry (SurfaceGeometry)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from .geometry import validate_geometry

PAGE_ACTIONS = {'pages', 'new_page', 'page', 'next_page', 'previous_page'}


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

    def as_dict(self):
        return {"id": self.id, "kind": self.kind, "payload": deepcopy(self.payload), 'status': self.status, 'error': self.error, 'page': self.page, 'bounds': dict(self.bounds)}


def finite(value): return type(value) in (int, float) and math.isfinite(value)


class WhiteboardModel:
    """Emits 'whiteboard', 'whiteboard_page', 'whiteboard_clear', 'whiteboard_surface' and the whiteboard's 'surface_result'.
    It saves the board on each of those and on SurfaceGeometry's 'whiteboard_geometry' (it subscribes to the shared
    DesktopEvents when built, before any other listener, so a failed save is in the snapshot that event publishes)."""
    __slots__ = ('events', 'geometry', '_lock', '_acknowledged', '_commands', '_visible', '_clear', '_pages', '_page',
                 '_path', '_file', '_save_lock', '_error')

    def __init__(self, events, geometry, *, board_file=None):
        """board_file: whiteboard.json (DataPaths.whiteboard). Nothing reads or writes it before load()."""
        self.events, self.geometry, self._path = events, geometry, board_file
        self._lock = threading.RLock()
        self._acknowledged = threading.Condition(self._lock)
        self._save_lock = threading.Lock()  # saves run in order, outside self._lock: a write can take a while
        self._commands: list[WhiteboardCommand] = []
        self._visible, self._clear = False, None
        self._pages, self._page = ['page-1'], 'page-1'
        self._file = None  # where saves go once load() has read the saved board
        self._error = ''
        events.subscribe(self._persist)

    def snapshot(self):
        with self._lock:
            return {'whiteboard': [command.as_dict() for command in self._commands], 'whiteboard_visible': self._visible,
                    'whiteboard_clear': dict(self._clear) if self._clear else None,
                    'whiteboard_pages': list(self._pages), 'whiteboard_page': self._page, 'board_persistence_error': self._error}

    def load(self, path=None):
        """Read the saved board (board_file unless path is given) and save there from now on. A damaged file is kept as it
        is: the board continues from, and saves to, <name>.recovery.json beside it."""
        if path is None and self._path is None: return
        path = Path(path if path is not None else self._path)
        self._file = path
        if not path.exists(): return
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            pages = raw['pages']
            if not isinstance(pages, list) or not pages or len(set(pages)) != len(pages) or not all(isinstance(p, str) for p in pages) or raw['page'] not in pages:
                raise ValueError('Invalid pages')
            commands = [WhiteboardCommand(**item) for item in raw['commands']]
            if len({c.id for c in commands}) != len(commands) or any(c.page not in pages or c.kind not in {'text', 'draw', 'image'} or not isinstance(c.payload, dict) for c in commands):
                raise ValueError('Invalid commands')
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
            geometry = raw.get('geometry', {})
            validate_geometry('whiteboard', geometry)
            with self._lock:
                self._pages, self._page = pages, raw['page']
                self._commands = commands
                self._visible = bool(raw.get('visible', False))
                for command in commands: command.status, command.error = 'queued', ''
            self.geometry.place('whiteboard', geometry, check_displays=False)  # saved before this run's displays are known
        except (OSError, ValueError, TypeError, KeyError):
            # Preserve the damaged original while allowing continued use/recovery.
            self._file = path.with_name(path.stem + '.recovery.json')
            if self._file.exists() and not path.stem.endswith('.recovery'):
                self.load(self._file)
            with self._lock: self._error = f'Invalid saved board preserved at {path.name}; new state saves to {self._file.name}'

    def _persist(self, event, value):
        if event.startswith('whiteboard') or event == 'surface_result' and value.get('surface') == 'whiteboard': self.save()

    def save(self):
        with self._save_lock:
            target = self._file
            if target is None: return
            geometry = self.geometry.whiteboard()
            with self._lock:
                raw = {'version': 1, 'pages': list(self._pages), 'page': self._page, 'commands': [command.as_dict() for command in self._commands],
                       'geometry': geometry, 'visible': self._visible}
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.tmp')
                # ASCII escapes: a lone surrogate from a model's tool call would fail UTF-8 encoding on every later save.
                temporary.write_text(json.dumps(raw), encoding='utf-8')
                temporary.replace(target)
            except (OSError, ValueError) as exc:
                with self._lock: self._error = str(exc)
                logging.getLogger(__name__).exception('Unable to persist whiteboard')

    def add(self, kind: str, payload: dict[str, Any]):
        payload = deepcopy(payload)
        command = WhiteboardCommand(kind, payload)
        with self._lock:
            command.page = self._page
            if kind == 'draw':
                xs, ys = zip(*payload['points'])
                stroke = payload.get('size', 6)
                command.bounds = {'x': min(xs)-stroke/2, 'y': min(ys)-stroke/2, 'width': max(1, max(xs)-min(xs))+stroke, 'height': max(1, max(ys)-min(ys))+stroke}
            else:
                x = payload.get('x')
                y = payload.get('y')
                if x is None: x = 40
                payload['auto_place'] = y is None
                if y is None: y = max((c.bounds.get('y', 0) + c.bounds.get('height', 100) + 24 for c in self._commands if c.page == command.page), default=40)
                payload.update(x=x, y=y)
                command.bounds = {'x': x, 'y': y, 'width': payload.get('width', 420), 'height': 120}
            self._commands.append(command)
            self._visible = True
            added = command.as_dict()
        self.events.emit("whiteboard", added)
        return command.id

    def pages(self, action, page=None):
        if action not in PAGE_ACTIONS: raise ValueError('Unknown page action')
        with self._lock:
            if action == 'new_page':
                page = f'page-{len(self._pages)+1}'
                self._pages.append(page)
            if action in {'new_page', 'page'}:
                if page not in self._pages: raise ValueError('Unknown page')
                self._page = page
            elif action in {'next_page', 'previous_page'}:
                index = self._pages.index(self._page) + (1 if action == 'next_page' else -1)
                self._page = self._pages[max(0, min(len(self._pages)-1, index))]
            self._visible = True
            result = {'pages': list(self._pages), 'current_page': self._page}
        self.events.emit('whiteboard_page', result['current_page'])
        return result

    def result(self, command_id, timeout=2):
        """The command's page, bounds and status once the renderer has acknowledged it, or as queued after timeout seconds."""
        with self._acknowledged:
            self._acknowledged.wait_for(lambda: not any(c.id == command_id and c.status == 'queued' for c in self._commands), timeout=timeout)
            return next(({'id': c.id, 'page': c.page, 'bounds': dict(c.bounds), 'status': c.status, 'error': c.error} for c in self._commands if c.id == command_id), {'id': command_id, 'status': 'removed'})

    def clear(self):
        with self._lock:
            self._commands.clear()
            self._clear = {'id': str(uuid.uuid4()), 'status': 'queued', 'error': ''}
            command_id = self._clear['id']
            self._acknowledged.notify_all()
        self.events.emit("whiteboard_clear", None)
        return command_id

    def acknowledge(self, command_id, status, error='', bounds=None):
        """The renderer's result for a command or the pending clear: rendered or error, with the command's measured size.
        False when it names neither (a stale command)."""
        if bounds is not None:
            if not isinstance(bounds, dict) or set(bounds) != {'x', 'y', 'width', 'height'}: raise ValueError('Invalid bounds')
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds.values()) or bounds['width'] <= 0 or bounds['height'] <= 0:
                raise ValueError('Invalid bounds')
        with self._lock:
            item = next((c for c in self._commands if c.id == command_id), None)
            if status not in {'rendered', 'error'}: raise ValueError('Invalid whiteboard result')
            if item is None:
                if not self._clear or self._clear['id'] != command_id: return False
                if self._clear['status'] == status and self._clear['error'] == error: return True
                self._clear.update(status=status, error=error)
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
                    for later in self._commands[self._commands.index(item)+1:]:
                        if later.page != item.page: continue
                        if later.payload.get('auto_place') and later.bounds['y'] < bottom:
                            later.bounds['y'] = later.payload['y'] = bottom
                        bottom = max(bottom, later.bounds['y'] + later.bounds['height'] + 24)
            self._acknowledged.notify_all()
        self.events.emit('surface_result', {'surface': 'whiteboard', 'id': command_id, 'status': status, 'error': error})
        return True

    def set_surface(self, *, visible=None, geometry=None):
        """Show or hide the whiteboard window and move it (validated against the displays; nothing changes when it is invalid)."""
        current = self.geometry.place('whiteboard', geometry) if geometry is not None else self.geometry.whiteboard()
        if visible is not None:
            with self._lock: self._visible = visible
        self.events.emit('whiteboard_surface', current)
