"""Shim: moved to kernel/workers.py. Old imports keep working, with the same objects, for one release; import from kernel."""
from ..kernel.workers import DaemonExecutor  # noqa: F401
