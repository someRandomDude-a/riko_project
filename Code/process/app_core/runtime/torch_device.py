"""Shim: moved to kernel/torch_device.py. Old imports keep working, with the same objects, for one release; import from kernel."""
from ..kernel.torch_device import preserve_torch_globals, resolve, validate  # noqa: F401
