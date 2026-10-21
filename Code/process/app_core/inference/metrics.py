"""Shim: moved to kernel/metrics.py. Old imports keep working, with the same objects, for one release; import from kernel."""
from ..kernel.metrics import InferenceMetrics  # noqa: F401
