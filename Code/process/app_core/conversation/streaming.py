"""Shim: moved to kernel/streaming.py. Old imports keep working, with the same objects, for one release; import from kernel."""
from ..kernel.streaming import WordDeltas  # noqa: F401
