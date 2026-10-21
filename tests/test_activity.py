"""Computer awareness adapters (runtime/activity.py): every OS path is faked, so each runs on every CI OS."""
import subprocess
import sys

import pytest

from process.app_core.runtime.activity import (ActivityUnsupported, LinuxActivity, MacActivity, NullActivity, WindowsActivity,
    activity_adapter)


def runner(answers, calls=None):
    """A fake run_text: the first answer whose key is a word of the command; an exception is raised."""
    def run(command, timeout=2):
        if calls is not None: calls.append(command)
        for key, answer in answers.items():
            if key in command:
                if isinstance(answer, Exception): raise answer
                return answer
        raise subprocess.CalledProcessError(1, command, stderr=MISSING)
    return run


MISSING = 'Error: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown: The name is not activatable\n'  # gdbus, no such service


def test_registry_picks_the_platform_adapter_or_none():
    assert isinstance(activity_adapter('win32'), WindowsActivity) and isinstance(activity_adapter('darwin'), MacActivity)
    assert isinstance(activity_adapter('linux'), LinuxActivity)
    null = activity_adapter('freebsd14')
    assert isinstance(null, NullActivity) and null.sample(idle=True, active_app=True) == {}
    assert null.capabilities()['idle'] == {'supported': False, 'reason': 'not available on freebsd14'}
    assert WindowsActivity().capabilities()['active_app']['fields'] == ['window_title', 'process_id']


def test_mac_reads_coregraphics_idle_and_the_frontmost_app_without_window_titles():
    calls = []
    info = '"LSDisplayName"="Visual Studio Code"\n"pid"=18910\n"CFBundleIdentifier"="com.microsoft.VSCode"\n'
    mac = MacActivity(run=runner({'front': 'ASN:0x0-0x9ac9ac0:\n', 'info': info}, calls), idle=lambda: 12.5, exists=lambda path: True)
    assert mac.sample(idle=True) == {'idle_seconds': 12.5}
    assert mac.sample(active_app=True) == {'active_app': {'app_name': 'Visual Studio Code', 'bundle_id': 'com.microsoft.VSCode', 'process_id': 18910}}
    assert calls == [['/usr/bin/lsappinfo', 'front'], ['/usr/bin/lsappinfo', 'info', '-only', 'name', '-only', 'pid', '-only', 'bundleid', 'ASN:0x0-0x9ac9ac0:']]
    capabilities = mac.capabilities()
    assert capabilities['idle']['supported'] and capabilities['active_app']['fields'] == ['app_name', 'bundle_id', 'process_id']
    assert 'window_title' not in mac.sample(active_app=True)['active_app'] and 'Screen Recording' in capabilities['active_app']['description']
    with pytest.raises(RuntimeError, match='named no application'):
        MacActivity(run=runner({'front': '\n'}), idle=lambda: 0, exists=lambda path: True).sample(active_app=True)
    with pytest.raises(RuntimeError, match='no idle time'): MacActivity(idle=lambda: float('nan'), exists=lambda path: True).sample(idle=True)
    missing = MacActivity(exists=lambda path: False).capabilities()
    assert missing['active_app'] == {**missing['active_app'], 'supported': False, 'reason': 'lsappinfo is missing'} and not missing['idle']['supported']


@pytest.mark.skipif(sys.platform != 'darwin', reason='CoreGraphics and lsappinfo exist only on macOS')
def test_mac_adapter_reads_this_mac_for_real():
    mac = MacActivity()
    assert mac.capabilities()['idle']['supported'] and mac.sample(idle=True)['idle_seconds'] >= 0  # no permission prompt
    if mac.capabilities()['active_app']['supported']:
        try: app = mac.sample(active_app=True)['active_app']
        except RuntimeError as exc: assert 'named no application' in str(exc)  # a session with nothing in front (a CI runner)
        else: assert set(app) == {'app_name', 'bundle_id', 'process_id'}


def linux(environ, answers, which=('gdbus', 'xprop'), calls=None, clock=None):
    return LinuxActivity(environ=environ, which=lambda name: f'/usr/bin/{name}' if name in which else None, run=runner(answers, calls),
        **({'clock': clock} if clock else {}))


GNOME_X11 = {'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus', 'DISPLAY': ':0', 'XDG_SESSION_TYPE': 'x11', 'XDG_CURRENT_DESKTOP': 'GNOME'}


def test_linux_idle_prefers_mutter_and_falls_back_to_the_screensaver_service():
    calls = []
    gnome = linux(GNOME_X11, {'org.gnome.Mutter.IdleMonitor': '(uint64 42500,)\n'}, calls=calls)
    assert gnome.sample(idle=True) == {'idle_seconds': 42.5} and gnome.idle_method == 'mutter'
    assert calls[0][:3] == ['/usr/bin/gdbus', 'call', '--session'] and calls[0][-1] == 'org.gnome.Mutter.IdleMonitor.GetIdletime'
    xfce = linux({**GNOME_X11, 'XDG_CURRENT_DESKTOP': 'XFCE'}, {'org.freedesktop.ScreenSaver': '(uint32 30,)\n'})
    assert xfce.sample(idle=True) == {'idle_seconds': 30} and xfce.idle_method == 'screensaver'  # seconds, per the spec
    kde = linux({**GNOME_X11, 'XDG_CURRENT_DESKTOP': 'KDE'}, {'org.freedesktop.ScreenSaver': '(uint32 30000,)\n'})
    assert kde.sample(idle=True) == {'idle_seconds': 30}  # KDE answers in milliseconds


def test_linux_without_an_idle_service_turns_unsupported_but_a_service_that_answered_before_only_fails():
    bare = linux({**GNOME_X11, 'XDG_CURRENT_DESKTOP': 'i3'}, {})
    assert bare.capabilities()['idle']['supported']
    with pytest.raises(ActivityUnsupported, match='no idle-time service'): bare.sample(idle=True)
    assert bare.capabilities()['idle'] == {'supported': False, 'reason': bare.no_idle}
    answers = {'org.gnome.Mutter.IdleMonitor': '(uint64 1000,)'}
    gnome = linux(GNOME_X11, answers)
    gnome.sample(idle=True)
    answers['org.gnome.Mutter.IdleMonitor'] = subprocess.TimeoutExpired(['gdbus'], 2)
    with pytest.raises(RuntimeError, match='mutter') as failure: gnome.sample(idle=True)
    assert not isinstance(failure.value, ActivityUnsupported) and gnome.capabilities()['idle']['supported']
    assert not linux({'DISPLAY': ':0'}, {}).capabilities()['idle']['supported']  # no session bus
    assert linux(GNOME_X11, {}, which=('xprop',)).capabilities()['idle']['reason'] == 'gdbus is missing (GLib tools)'


def test_linux_idle_turns_unsupported_only_when_dbus_says_no_service_and_asks_again_later():
    """A slow session bus after login or a gnome-shell restart (a timeout, any other failure) is passing; only gdbus's
    ServiceUnknown/UnknownMethod for both services is unsupported, and even that is asked again after RETRY_SECONDS."""
    now = [0.0]
    answers = {'org.gnome.Mutter.IdleMonitor': subprocess.TimeoutExpired(['gdbus'], 2),
               'org.freedesktop.ScreenSaver': subprocess.CalledProcessError(1, ['gdbus'], stderr=MISSING)}
    gnome = linux(GNOME_X11, answers, clock=lambda: now[0])
    with pytest.raises(RuntimeError, match='mutter') as failure: gnome.sample(idle=True)
    assert not isinstance(failure.value, ActivityUnsupported) and gnome.capabilities()['idle']['supported'] and not gnome.no_idle
    answers['org.gnome.Mutter.IdleMonitor'] = '(uint64 2000,)'  # the next poll: Mutter is up
    assert gnome.sample(idle=True) == {'idle_seconds': 2.0}
    later = {'org.gnome.Mutter.IdleMonitor': subprocess.CalledProcessError(1, ['gdbus'], stderr=MISSING.replace('ServiceUnknown', 'UnknownMethod'))}
    bare = linux({**GNOME_X11, 'XDG_CURRENT_DESKTOP': 'i3'}, later, clock=lambda: now[0])
    with pytest.raises(ActivityUnsupported): bare.sample(idle=True)
    now[0] = LinuxActivity.RETRY_SECONDS - 1
    assert not bare.capabilities()['idle']['supported']
    later['org.gnome.Mutter.IdleMonitor'] = '(uint64 5000,)'  # gnome-shell came up meanwhile
    now[0] = LinuxActivity.RETRY_SECONDS
    assert bare.capabilities()['idle']['supported'] and bare.sample(idle=True) == {'idle_seconds': 5.0}


def test_linux_active_window_on_x11_and_never_on_wayland():
    window = ('_NET_WM_PID(CARDINAL) = 4242\nWM_CLASS(STRING) = "gnome-terminal-server", "Gnome-terminal"\n'
              '_NET_WM_NAME(UTF8_STRING) = "build \\"riko\\" ~/code"\n')
    x11 = linux(GNOME_X11, {'-root': '_NET_ACTIVE_WINDOW(WINDOW): window id # 0x3a00007\n', '-id': window})
    assert x11.capabilities()['active_app']['supported']
    assert x11.sample(active_app=True) == {'active_app': {'window_title': 'build "riko" ~/code', 'app_name': 'Gnome-terminal', 'process_id': 4242}}
    nothing = linux(GNOME_X11, {'-root': '_NET_ACTIVE_WINDOW(WINDOW): window id # 0x0\n'})
    assert nothing.sample(active_app=True) == {'active_app': {'window_title': '', 'app_name': '', 'process_id': None}}
    bare = linux(GNOME_X11, {'-root': '_NET_ACTIVE_WINDOW(WINDOW): window id # 0x1\n', '-id': '_NET_WM_PID:  not found.\nWM_CLASS:  not found.\n'})
    assert bare.sample(active_app=True)['active_app'] == {'window_title': '', 'app_name': '', 'process_id': None}
    for environ in ({**GNOME_X11, 'XDG_SESSION_TYPE': 'wayland'}, {**GNOME_X11, 'WAYLAND_DISPLAY': 'wayland-0'}):
        assert linux(environ, {}).capabilities()['active_app']['reason'] == 'Wayland does not tell apps which window is focused'
    assert linux({**GNOME_X11, 'DISPLAY': ''}, {}).capabilities()['active_app']['reason'] == 'no X11 display'
    assert linux(GNOME_X11, {}, which=('gdbus',)).capabilities()['active_app']['reason'] == 'xprop is missing (x11-utils)'
