"""Shim: moved to kernel/lifecycle.py. Old imports keep working, with the same objects, for one release; import from kernel."""
from ..kernel.lifecycle import close_bounded, run_bounded  # noqa: F401
