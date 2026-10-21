"""Shared validation for saved settings, API requests and model window tools, and SurfaceGeometry, where the avatar and
whiteboard windows are and on which displays."""
from __future__ import annotations

from copy import deepcopy
import json
import logging
import threading


def validate_geometry(target, geometry, displays=()):
    if target not in {'avatar', 'whiteboard'}:
        raise ValueError('Unknown surface')
    if not isinstance(geometry, dict) or set(geometry) - {'x', 'y', 'width', 'height', 'screen'}:
        raise ValueError('Invalid geometry fields')
    if any(type(value) is not int for value in geometry.values()):
        raise ValueError('Geometry must use integers')
    minimum = 100 if target == 'avatar' else 200
    if any(not minimum <= geometry[key] <= 4096 for key in ('width', 'height') if key in geometry):
        raise ValueError(f'Surface size must be {minimum}–4096')
    if geometry.get('screen', 0) < 0:
        raise ValueError('Screen index must be nonnegative')
    if 'screen' in geometry and displays and geometry['screen'] not in {display['index'] for display in displays}:
        raise ValueError('Unknown display index; inspect runtime displays')


class SurfaceGeometry:
    """The avatar and whiteboard windows' geometry, Electron's displays, and desktop_settings.json, which keeps the avatar's
    geometry (whiteboard.json keeps the whiteboard's: WhiteboardModel). Emits '<target>_geometry' and 'displays'."""
    __slots__ = ('events', '_lock', '_avatar', '_whiteboard', '_displays', '_settings_file', '_write', '_write_lock', '_settings_loaded')

    def __init__(self, events, *, settings_file=None, write=None):
        """settings_file: desktop_settings.json (DataPaths.desktop_settings), read only by load_settings; write(path, text)
        replaces it durably (persistence/atomic.atomic_write, which the factory passes; desktop imports only kernel)."""
        if settings_file is not None and write is None: raise TypeError('A settings file needs its writer')
        self.events, self._settings_file, self._write = events, settings_file, write
        self._lock, self._write_lock = threading.Lock(), threading.Lock()
        self._avatar = {"x": 0, "y": 0, "width": 480, "height": 720, "screen": 0}
        self._whiteboard = {"x": 500, "y": 100, "width": 900, "height": 700, "screen": 0}
        self._displays = []
        self._settings_loaded = False  # the avatar's saved geometry was restored: the first displays report keeps its screen

    @property
    def settings_loaded(self): return self._settings_loaded

    def avatar(self):
        with self._lock: return dict(self._avatar)

    def whiteboard(self):
        with self._lock: return dict(self._whiteboard)

    def snapshot(self):
        with self._lock: return {'avatar_geometry': dict(self._avatar), 'displays': deepcopy(self._displays), 'whiteboard_geometry': dict(self._whiteboard)}

    def place(self, target, geometry, *, check_displays=True):
        """Validate geometry for target (against the displays unless restoring a saved board) and apply it without an event:
        the caller reports the change. Returns the new geometry."""
        with self._lock:
            validate_geometry(target, geometry, self._displays if check_displays else ())
            current = self._avatar if target == 'avatar' else self._whiteboard
            current.update(geometry)
            return dict(current)

    def update(self, target, **geometry):
        self.events.emit(f'{target}_geometry', self.place(target, geometry))

    def set_displays(self, displays):
        """Electron's displays (the route checks ordered indices and one primary). The avatar moves to the primary display when
        its own is gone, or on the first report when no saved geometry placed it."""
        displays = deepcopy(displays)
        indices = {display['index'] for display in displays}
        with self._lock:
            if (not self._displays and not self._settings_loaded) or self._avatar['screen'] not in indices:
                self._avatar['screen'] = next(display['index'] for display in displays if display['primary'])
            self._displays = displays
        self.events.emit('displays', deepcopy(displays))

    def load_settings(self):
        """Restore the avatar's geometry from desktop_settings.json. A file that cannot be read or holds invalid geometry is
        ignored and left as it is. Returns whether it was restored."""
        with self._lock: self._settings_loaded = False
        path = self._settings_file
        if path is None or not path.exists(): return False
        try: self.update('avatar', **json.loads(path.read_text(encoding='utf-8'))['avatar_geometry'])
        except (OSError, ValueError, KeyError, TypeError):
            logging.getLogger(__name__).warning('Ignoring invalid desktop settings; original file preserved')
            return False
        with self._lock: self._settings_loaded = True
        return True

    def save_avatar_geometry(self):
        """Write the avatar's geometry to desktop_settings.json; OSError propagates with the file untouched. Writes run in
        order under their own lock, never this component's: a durable write can take a while."""
        if self._settings_file is None: return
        with self._write_lock:
            with self._lock: payload = json.dumps({'avatar_geometry': self._avatar})
            self._write(self._settings_file, payload)
