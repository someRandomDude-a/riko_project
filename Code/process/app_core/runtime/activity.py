"""Computer awareness for initiative: seconds since the last keyboard or mouse input, and which app is in front. Read-only
OS metadata, one adapter per platform: never hook input, read keys, capture the screen or look inside another process.

An adapter has `platform`, `capabilities()` (static checks only: environment, executables, files; cached, never a process)
and `sample(*, idle=False, active_app=False)`, which raises on a failure and raises ActivityUnsupported when it finds that
this desktop cannot answer at all (Initiative then treats the field as unsupported, not as an error).
"""
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


class ActivityUnsupported(RuntimeError):
    """This desktop cannot answer (no idle-time service): the field is unsupported here, not failing."""


def capability(supported, reason='', **extra):
    return {'supported': bool(supported), 'reason': '' if supported else reason, **extra}


def run_text(command, timeout=2):
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout


class WindowsActivity:
    """GetLastInputInfo, and the foreground window's title and process ID."""
    platform = 'win32'

    def capabilities(self):
        return {'idle': capability(True), 'active_app': capability(True, fields=['window_title', 'process_id'],
            description='the foreground window title and process ID')}

    def sample(self, *, idle=False, active_app=False):
        if os.name != 'nt': raise RuntimeError('WindowsActivity reads Win32 input and window state')
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        result = {}
        if idle:
            class LastInput(ctypes.Structure):
                _fields_ = [('cbSize', wintypes.UINT), ('dwTime', wintypes.DWORD)]
            info = LastInput()
            info.cbSize = ctypes.sizeof(info)
            if not user32.GetLastInputInfo(ctypes.byref(info)): raise ctypes.WinError(ctypes.get_last_error())
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.GetTickCount.restype = wintypes.DWORD
            result['idle_seconds'] = ((kernel32.GetTickCount() - info.dwTime) & 0xffffffff) / 1000
        if active_app:
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
            user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
            user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            handle = user32.GetForegroundWindow()
            title = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(handle) + 1)
            user32.GetWindowTextW(handle, title, len(title))
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
            # Title + process ID avoids inspecting process contents or requiring privileges.
            result['active_app'] = {'window_title': title.value, 'process_id': pid.value}
        return result


COREGRAPHICS = '/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics'  # in the dyld shared cache, not on disk: check its folder


class MacActivity:
    """Idle time from CoreGraphics (CGEventSourceSecondsSinceLastEventType over the HID system state and every input event
    type: no permission prompt), and the frontmost app from lsappinfo (its name, bundle ID and process ID; a window title
    would need Screen Recording permission, which Riko never asks for)."""
    platform = 'darwin'
    ASN = re.compile(r'ASN:0x[0-9a-f]+-0x[0-9a-f]+:', re.IGNORECASE)

    def __init__(self, run=run_text, idle=None, exists=os.path.exists, lsappinfo='/usr/bin/lsappinfo'):
        self.run, self.idle_function, self.exists, self.lsappinfo = run, idle, exists, lsappinfo
        self.cached = None

    def capabilities(self):
        if self.cached is None:
            self.cached = {'idle': capability(self.idle_function is not None or self.exists(os.path.dirname(COREGRAPHICS)), 'CoreGraphics is missing'),
                'active_app': capability(self.exists(self.lsappinfo), 'lsappinfo is missing', fields=['app_name', 'bundle_id', 'process_id'],
                    description="the frontmost app's name, bundle ID and process ID (never window titles, which need Screen Recording permission)")}
        return self.cached

    def idle_seconds(self):
        if self.idle_function is None:
            import ctypes
            seconds = ctypes.CDLL(COREGRAPHICS).CGEventSourceSecondsSinceLastEventType
            seconds.argtypes, seconds.restype = [ctypes.c_int32, ctypes.c_uint32], ctypes.c_double
            self.idle_function = lambda: seconds(1, 0xFFFFFFFF)  # kCGEventSourceStateHIDSystemState, kCGAnyInputEventType
        value = float(self.idle_function())
        if not math.isfinite(value) or value < 0: raise RuntimeError(f'CoreGraphics reported no idle time ({value})')
        return value

    def front_app(self):
        asn = self.run([self.lsappinfo, 'front']).strip()
        if not self.ASN.fullmatch(asn): raise RuntimeError(f'lsappinfo front named no application ({asn[:40]!r})')
        text = self.run([self.lsappinfo, 'info', '-only', 'name', '-only', 'pid', '-only', 'bundleid', asn])
        fields = {key: value[1:-1].replace('\\"', '"') if value.startswith('"') else value
                  for key, value in re.findall(r'"([^"]+)"\s*=\s*("(?:[^"\\]|\\.)*"|\S+)', text)}
        pid = fields.get('pid', '')
        return {'app_name': fields.get('LSDisplayName', ''), 'bundle_id': fields.get('CFBundleIdentifier', ''),
                'process_id': int(pid) if pid.isdigit() else None}

    def sample(self, *, idle=False, active_app=False):
        result = {}
        if idle: result['idle_seconds'] = self.idle_seconds()
        if active_app: result['active_app'] = self.front_app()
        return result


class LinuxActivity:
    """Idle time over D-Bus (GNOME's Mutter IdleMonitor, else org.freedesktop.ScreenSaver: seconds by the spec, milliseconds
    on KDE) through gdbus; the active window on X11 through xprop (Xlib's default error handler would exit this process on a
    window that closed meanwhile). Wayland does not let an app see the focused window: XWayland knows only X clients."""
    platform = 'linux'
    MUTTER = ('org.gnome.Mutter.IdleMonitor', '/org/gnome/Mutter/IdleMonitor/Core', 'org.gnome.Mutter.IdleMonitor.GetIdletime')
    SCREENSAVER = ('org.freedesktop.ScreenSaver', '/org/freedesktop/ScreenSaver', 'org.freedesktop.ScreenSaver.GetSessionIdleTime')
    # gdbus's answer when nobody offers the service or method; a timeout or any other failure is passing, and retried.
    MISSING = ('org.freedesktop.DBus.Error.ServiceUnknown', 'org.freedesktop.DBus.Error.NameHasNoOwner', 'org.freedesktop.DBus.Error.UnknownMethod',
               'org.freedesktop.DBus.Error.UnknownObject', 'org.freedesktop.DBus.Error.UnknownInterface')
    RETRY_SECONDS = 60  # a desktop found without an idle service is asked again after this (gnome-shell may start later)

    def __init__(self, environ=None, which=shutil.which, run=run_text, clock=time.monotonic):
        self.environ = os.environ if environ is None else environ
        self.which, self.run, self.clock = which, run, clock
        self.cached, self.idle_method, self.no_idle, self.no_idle_at = None, None, '', 0.0

    def session_bus(self):
        runtime = self.environ.get('XDG_RUNTIME_DIR')
        return bool(self.environ.get('DBUS_SESSION_BUS_ADDRESS')) or bool(runtime and Path(runtime, 'bus').exists())

    def wayland(self):
        return self.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland' or bool(self.environ.get('WAYLAND_DISPLAY'))

    def missing_idle(self):
        """Why this desktop has no idle-time service, until RETRY_SECONDS after it was found missing ('' otherwise)."""
        if self.no_idle and self.clock() - self.no_idle_at >= self.RETRY_SECONDS: self.no_idle = ''
        return self.no_idle

    def capabilities(self):
        if self.cached is None:
            gdbus, xprop = self.which('gdbus'), self.which('xprop')
            idle = capability(gdbus and self.session_bus(), 'no D-Bus session bus' if gdbus else 'gdbus is missing (GLib tools)')
            if self.wayland(): app = capability(False, 'Wayland does not tell apps which window is focused')
            elif not self.environ.get('DISPLAY'): app = capability(False, 'no X11 display')
            else: app = capability(xprop, 'xprop is missing (x11-utils)')
            self.cached = {'idle': idle, 'active_app': {**app, 'fields': ['window_title', 'app_name', 'process_id'],
                'description': 'the active window title, its app (WM_CLASS) and process ID'}}
        if self.missing_idle(): return {**self.cached, 'idle': capability(False, self.no_idle)}
        return self.cached

    def call(self, service):
        destination, path, method = service
        text = self.run([self.which('gdbus') or 'gdbus', 'call', '--session', '--dest', destination, '--object-path', path, '--method', method])
        match = re.fullmatch(r'\((?:u?int(?:32|64)|uint)?\s*(\d+),\)', text.strip())
        if not match: raise ValueError(f'unexpected {method} answer {text.strip()[:60]!r}')
        return int(match[1])

    def idle_seconds(self):
        if self.missing_idle(): raise ActivityUnsupported(self.no_idle)
        kde = 'KDE' in self.environ.get('XDG_CURRENT_DESKTOP', '').upper()
        methods = {'mutter': lambda: self.call(self.MUTTER) / 1000, 'screensaver': lambda: self.call(self.SCREENSAVER) / (1000 if kde else 1)}
        errors, missing = [], []
        for name in [self.idle_method] if self.idle_method else methods:
            try: value = methods[name]()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                stderr = getattr(exc, 'stderr', None) or ''
                detail = (stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr).strip() or str(exc).strip()
                if any(error in detail for error in self.MISSING): missing.append(name)
                errors.append(f'{name}: {detail[:160] or type(exc).__name__}')
                continue
            self.idle_method = name
            return value
        if self.idle_method or len(missing) < len(methods): raise RuntimeError('; '.join(errors))  # passing: asked again next poll
        self.no_idle, self.no_idle_at = 'this desktop offers no idle-time service (GNOME Mutter IdleMonitor or org.freedesktop.ScreenSaver)', self.clock()
        raise ActivityUnsupported(self.no_idle)

    def active_window(self):
        xprop = self.which('xprop') or 'xprop'
        active = re.search(r'window id # (0x[0-9a-f]+)', self.run([xprop, '-root', '_NET_ACTIVE_WINDOW']), re.IGNORECASE)
        if not active or int(active[1], 16) == 0: return {'window_title': '', 'app_name': '', 'process_id': None}
        text = self.run([xprop, '-id', active[1], '_NET_WM_PID', 'WM_CLASS', '_NET_WM_NAME'])
        def strings(name):
            line = re.search(rf'^{name}\([^)]*\) = (.*)$', text, re.MULTILINE)
            return [value.replace('\\"', '"').replace('\\\\', '\\') for value in re.findall(r'"((?:[^"\\]|\\.)*)"', line[1])] if line else []
        pid = re.search(r'^_NET_WM_PID\(CARDINAL\) = (\d+)$', text, re.MULTILINE)
        classes, title = strings('WM_CLASS'), strings('_NET_WM_NAME')
        return {'window_title': title[0] if title else '', 'app_name': classes[-1] if classes else '', 'process_id': int(pid[1]) if pid else None}

    def sample(self, *, idle=False, active_app=False):
        result = {}
        if idle: result['idle_seconds'] = self.idle_seconds()
        if active_app: result['active_app'] = self.active_window()
        return result


class NullActivity:
    """An OS with no adapter: both fields unsupported, and sampling returns nothing."""
    def __init__(self, platform=sys.platform): self.platform = platform

    def capabilities(self):
        reason = f'not available on {self.platform}'
        return {'idle': capability(False, reason), 'active_app': capability(False, reason, fields=[], description='')}

    def sample(self, *, idle=False, active_app=False): return {}


ADAPTERS = {'win32': WindowsActivity, 'darwin': MacActivity, 'linux': LinuxActivity}


def activity_adapter(platform=sys.platform):
    adapter = next((adapter for name, adapter in ADAPTERS.items() if platform.startswith(name)), None)
    return adapter() if adapter else NullActivity(platform)
