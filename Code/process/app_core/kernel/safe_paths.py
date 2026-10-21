"""Paths from outside (the renderer, the model, a settings form) checked as text before any filesystem call. On Windows,
resolving or even stat-ing a UNC path (\\\\host\\share\\x) contacts that host and offers the user's credentials, so a check
that resolves first has already leaked them: lexical() refuses network and device paths and Windows' ambiguous forms on the
text alone, within() checks containment the same way, and only then do resolve_within() and local_path()'s callers touch
the filesystem (resolve_within re-checks the real path, so a symlink cannot point out). flavour is os.path, or ntpath or
posixpath in tests, so every case runs on every OS. Only the value is checked, never base or the roots: a data root on a
share or a mapped drive keeps working."""
import ntpath
import os
from pathlib import Path
import re

NETWORK = 'Network and device paths are not supported; copy the file to this computer'
OUTSIDE = 'Media path is outside approved asset directories'
INVALID = 'Invalid path'
# Matched against a component's name before its first '.', trailing spaces removed, as Win32 (and ntpath.isreserved in
# Python 3.13+) does: 'COM1 .png' and 'nul.tar.gz' open the device.
RESERVED = re.compile(r'(?i)con|prn|aux|nul|com[0-9\u00b9\u00b2\u00b3]|lpt[0-9\u00b9\u00b2\u00b3]|conin\$|conout\$')
SEPARATORS = re.compile(r'[\\/]')


class Outside(ValueError):
    """A well-formed path outside every allowed root."""


def lexical(value, base, *, flavour=os.path, network=False):
    """value joined to base and normalised, as text, or ValueError: not text, empty, over 4096 characters or holding a control
    character; a device path (\\\\?\\, \\\\.\\); a network path unless network; and with ntpath, Windows' invalid characters,
    a drive that is not one letter, a drive-relative path (C:x), any other ':' (an alternate data stream), a component
    ending in '.' or a space, or a reserved device name (CON, NUL, COM1, ...). No filesystem call."""
    if not isinstance(value, str) or not value or len(value) > 4096 or any(ord(c) < 32 or c == '\x7f' for c in value): raise ValueError(INVALID)
    lead = value[:4].replace('\\', '/')
    if lead in ('//?/', '//./') or not network and lead[:2] == '//': raise ValueError(NETWORK)
    if flavour is ntpath:
        if any(c in '<>"|?*' for c in value): raise ValueError(INVALID)
        drive, rest = ntpath.splitdrive(value)
        if drive and drive[:2].replace('\\', '/') != '//':
            if not re.fullmatch(r'[A-Za-z]:', drive): raise ValueError(INVALID)
            if rest[:1] not in ('\\', '/'): raise ValueError('Use a full path such as C:\\folder\\file')
        if ':' in rest: raise ValueError(INVALID)
        for part in SEPARATORS.split(rest):
            if part not in ('', '.', '..') and (part[-1] in '. ' or RESERVED.fullmatch(part.partition('.')[0].rstrip(' '))): raise ValueError(INVALID)
    return flavour.normpath(flavour.join(os.fspath(base), value))


def within(value, roots, base, *, flavour=os.path):
    """lexical(value) when it lies inside one of roots (compared as text, case-insensitively on Windows), else Outside."""
    path = lexical(value, base, flavour=flavour, network=True)
    for root in roots:
        root = flavour.normpath(os.fspath(root))
        try:
            if flavour.normcase(flavour.commonpath([root, path])) == flavour.normcase(root): return path
        except ValueError: continue  # another drive or share, or relative against absolute
    raise Outside(OUTSIDE)


def resolve_within(value, roots, base):
    """within(), then the real path, which must still lie inside one of the real roots (a symlink cannot point out)."""
    path = Path(within(value, roots, base)).resolve()
    if not any(path.is_relative_to(Path(root).resolve()) for root in roots): raise Outside(OUTSIDE)
    return path


def local_path(value, base):
    """A file the user chose anywhere on this computer: ~ expanded, then lexical() without network or device paths. The
    caller resolves it."""
    value = os.fspath(value) if isinstance(value, os.PathLike) else value
    if isinstance(value, str): value = os.path.expanduser(value)
    return Path(lexical(value, base))
