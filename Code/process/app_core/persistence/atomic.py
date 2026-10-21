"""Replace a user file durably: readers see the old contents or the new, never a torn or empty file."""
import glob
import os
from pathlib import Path
import stat
import tempfile
import time

try: from fcntl import F_FULLFSYNC, fcntl  # macOS only: there fsync stops at the drive's cache
except ImportError: F_FULLFSYNC = fcntl = None
# Windows refuses to replace a file (WinError 5/32) while antivirus, the search indexer, OneDrive or an editor briefly
# holds it, or the new temporary file, without FILE_SHARE_DELETE. Retry for about 0.8 s in all, then fail.
REPLACE_ATTEMPTS, REPLACE_BACKOFF_SECONDS = 6, .025
STALE_TEMPORARY_SECONDS = 600  # older than any write in progress: left by a killed process


def sync_file(descriptor):
    """Flush a file to stable storage, through the drive's cache where the OS allows (F_FULLFSYNC on macOS)."""
    if F_FULLFSYNC is not None:
        try:
            fcntl(descriptor, F_FULLFSYNC)
            return
        except OSError: pass  # a volume without it (network, FUSE): plain fsync
    os.fsync(descriptor)


def sync_directory(directory):
    """Best effort: make a rename itself durable on POSIX. Windows needs (and allows) no directory fsync."""
    if os.name == 'nt': return
    try: descriptor = os.open(directory, os.O_RDONLY)
    except OSError: return
    try: os.fsync(descriptor)
    except OSError: pass
    finally: os.close(descriptor)


def replace_file(source, target):
    """os.replace, retried with backoff while another program briefly holds either file."""
    for attempt in range(REPLACE_ATTEMPTS):
        try: return os.replace(source, target)
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1: raise
            time.sleep(REPLACE_BACKOFF_SECONDS * 2 ** attempt)


def atomic_write(path, data, encoding='utf-8', *, mode=None):
    """Replace path with data (str or bytes) through a unique temporary file beside it, synced before the rename.

    Raises OSError with the original file untouched and no temporary file left when any step fails. An existing
    file keeps its permission bits; a new one is owner-only. mode (POSIX permission bits, such as 0o600) sets them
    whatever the existing file had.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.encode(encoding) if isinstance(data, str) else data
    if os.name == 'nt': mode = None
    elif mode is None:
        try: mode = stat.S_IMODE(os.stat(path).st_mode)
        except OSError: pass
    for leftover in path.parent.glob(f'.{glob.escape(path.name)}.*.tmp'):  # a process killed mid-write leaves its copy
        try:
            if time.time() - leftover.stat().st_mtime > STALE_TEMPORARY_SECONDS: leftover.unlink()
        except OSError: pass
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            if mode is not None: os.fchmod(stream.fileno(), mode)
            stream.write(payload)
            stream.flush()
            sync_file(stream.fileno())
        replace_file(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except OSError: pass
        raise
    sync_directory(path.parent)
