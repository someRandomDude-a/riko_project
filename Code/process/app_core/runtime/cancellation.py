"""Shim: moved to kernel/cancellation.py. Old imports keep working, with the same objects, for one release; import from kernel."""
from ..kernel.cancellation import TurnCancelled  # noqa: F401
