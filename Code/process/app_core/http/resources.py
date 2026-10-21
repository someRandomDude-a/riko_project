"""/api/resources and /ws/resources/gpu: GPU telemetry, Electron's process list and the VRAM estimate for a Settings draft."""
from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..configuration.config import load_config
from ..events.stream import stream_events
from .backend import Services

router = APIRouter()


@router.get('/api/resources/gpu')
def gpu_status(backend: Services):
    return backend.gpu_monitor.sample(backend.provider())

class ElectronProcesses(BaseModel):
    processes: list[dict]

@router.post('/api/resources/electron')
def electron_processes(request: ElectronProcesses, backend: Services):
    try: backend.gpu_monitor.register_electron(request.processes)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'ok': True}

class ResourceEstimate(BaseModel):
    changes: dict = Field(default_factory=dict)

PATH_SUFFIXES = ('_path', '_dir', '_directory', '_folder', '_file', '_root')

@router.post('/api/resources/estimate')
def resource_estimate(request: ResourceEstimate, backend: Services):
    import os
    import tempfile
    from pathlib import Path
    from ..resources.vram_estimate import draft_telemetry, estimate
    from ..configuration.settings_store import LOCK
    from ..kernel.safe_paths import local_path
    store = backend.settings_store()
    # The estimate stats and reads a draft's model files, so its paths are checked as text first (kernel/safe_paths.py: on
    # Windows, stat-ing a network path contacts that host with the user's credentials).
    unsafe = {}
    for key, value in request.changes.items():
        if isinstance(key, str) and key.endswith(PATH_SUFFIXES) and isinstance(value, str) and value:
            try: local_path(value, store.path.parent)
            except ValueError as exc: unsafe[key] = str(exc)
    if unsafe: raise HTTPException(400, 'Invalid estimate settings: ' + '; '.join(f'{key}: {value}' for key, value in unsafe.items()))
    with LOCK:
        _, output, errors = store.prepare(request.changes)
        # A prompt+output mismatch does not prevent estimating the selected KV
        # allocation. Show feedback without silently changing either parameter;
        # normal settings save still rejects this draft.
        budget_only = bool(errors) and set(errors) == {'runtime.n_ctx'} and errors['runtime.n_ctx'].startswith('Context must fit')
        if errors and not budget_only: raise HTTPException(400, 'Invalid estimate settings: ' + '; '.join(f'{key}: {value}' for key, value in errors.items()))
        descriptor, temporary = tempfile.mkstemp(prefix='.resource-estimate-', suffix='.yaml', dir=store.path.parent)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as file: file.write(output)
            candidate = load_config(temporary)
        finally: Path(temporary).unlink(missing_ok=True)
    if 'initiative.context_window_tokens' in request.changes:
        candidate.runtime.initiative_n_ctx = request.changes['initiative.context_window_tokens']
    projected = estimate(candidate, draft_telemetry(backend.gpu_monitor.sample(backend.provider()), backend.config.runtime, candidate.runtime))
    if errors: projected.setdefault('warnings', []).extend('Unsavable draft: ' + value for value in errors.values())
    return {'estimate': projected, 'validation_errors':errors, 'draft': bool(request.changes)}

@router.websocket('/ws/resources/gpu')
async def gpu_events(websocket: WebSocket, backend: Services):
    monitor = backend.gpu_monitor
    unsubscribe = await run_in_threadpool(monitor.subscribe, backend.provider)
    try:
        await stream_events(websocket, backend.bus, lambda: {},
            initial=lambda: {'gpu': monitor.sample(backend.provider())},
            event_filter=lambda event: event.type == 'resource.gpu')
    finally: await run_in_threadpool(unsubscribe)
