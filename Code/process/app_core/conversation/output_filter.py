"""Shim: moved to kernel/output_filter.py. Old imports keep working, with the same objects, for one release; import from kernel."""
from ..kernel.output_filter import OutputFilter, clean_output  # noqa: F401
