"""Read-only GPU telemetry: one row per physical GPU, joined from the sources in gpu_sources.py. Never load a model to
measure it, and never invent an attribution the driver does not report."""
from copy import deepcopy
import json
import os
import threading
import time

from .gpu_sources import MIB, BridgeSource, platform_sources, process_identity  # noqa: F401 (MIB and process_identity: their old home)

OWNED = 'Python (LLM, ASR and runtime)'
FIELDS = ('index', 'name', 'uuid', 'pci_bus_id', 'luid', 'backend', 'ggml_device', 'ggml_index', 'llm', 'main', 'memory_kind',
          'memory_scope', 'total_mib', 'used_mib', 'free_mib')


def reconcile(gpus, processes, owned, *, complete=False):
    """Do not fabricate attribution when the driver hides per-process memory. processes: {'gpu': a row's key, 'pid',
    'used_mib'}; complete: the list holds every process on those GPUs (Windows PDH), not only some (nvidia-smi compute)."""
    result = deepcopy(gpus)
    for gpu in result:
        rows = [p for p in processes if p['gpu'] == gpu['key']]
        unknown = any(row['used_mib'] is None for row in rows)
        ours = [row for row in rows if row['pid'] in owned]
        known = sum(row['used_mib'] or 0 for row in ours)
        attributable = complete and not unknown and gpu['used_mib'] is not None and sum(row['used_mib'] or 0 for row in rows) <= gpu['used_mib'] + 64
        gpu.update(owned_mib=known if attributable else None,
            other_mib=max(0, gpu['used_mib'] - known) if attributable else None,
            attribution_complete=attributable,
            owned_processes=[{'pid': row['pid'], 'used_mib': row['used_mib'], 'category': owned[row['pid']]} for row in ours])
    return result


def row_key(row):
    if row.get('pci_bus_id'): return 'pci:' + row['pci_bus_id']
    if row.get('uuid'): return 'uuid:' + row['uuid']
    if row.get('luid'): return 'luid:' + row['luid']
    if row.get('ggml_device'): return 'ggml:' + row['ggml_device']
    return f'gpu:{row["sources"][0]}:{row.get("index")}'


def matching(rows, device, label):
    """The row this source's device is: the same PCI address, UUID or LUID; failing those (D3DKMT could not say), the one
    row of the same name that this source has not filled yet and that has no different PCI address."""
    for name in ('pci_bus_id', 'uuid', 'luid'):
        value = device.get(name)
        if value is not None:
            for row in rows:
                if row.get(name) == value: return row
    title = (device.get('name') or '').casefold()
    candidates = [row for row in rows if title and (row.get('name') or '').casefold() == title and label not in row['sources']
        and not (row.get('pci_bus_id') and device.get('pci_bus_id'))]
    return candidates[0] if len(candidates) == 1 else None


def merge(readings, owned, errors=None):
    """The telemetry payload from the sources' readings (vendor sources first, the bridge last): a field takes the first
    source's value that is not None, so device-wide vendor figures win and ggml's fill the gaps, except an integrated GPU's
    memory, where ggml's shared pool (what llama.cpp budgets) replaces the vendor's carve-out. Attribution comes from the
    first source with a complete process list for that GPU, else from the incomplete ones (and stays unknown). Metal's
    figures are this process's own allocations: owned is all of used, and nobody else's use can be seen."""
    rows, lists, ggml = [], {}, None
    for reading in readings:
        if reading.ggml_devices is not None: ggml = reading.ggml_devices
        local = {}
        for device in reading.devices:
            row = matching(rows, device, reading.label)
            if row is None:
                row = {'sources': []}
                rows.append(row)
            for name in FIELDS:
                if row.get(name) is None and device.get(name) is not None: row[name] = device[name]
            for name in device.get('replace', ()): row[name], row['shared_pool'] = device[name], True  # an iGPU: ggml's pool
            if reading.label not in row['sources']: row['sources'].append(reading.label)
            local[device['id']] = row
        for device_id, row in local.items():
            processes = [p for p in reading.processes if p['device'] == device_id]
            if processes or device_id in reading.complete: lists.setdefault(id(row), []).append((device_id in reading.complete, processes))
    rows = [row for row in rows if row.get('total_mib') is not None or row.get('used_mib') is not None]  # not only processes
    gpus = []
    for row in rows:
        row = {name: row.get(name) for name in FIELDS} | {'sources': row['sources']}
        row.update(key=row_key(row), memory_kind=row['memory_kind'] or 'dedicated', memory_scope=row['memory_scope'] or 'device', main=bool(row['main']))
        gpus.append(row)
    result = []
    for source_row, gpu in zip(rows, gpus):
        found = lists.get(id(source_row), [])
        if gpu['memory_scope'] == 'process':  # Metal: this process's working set; Electron's lives in other processes
            pid = os.getpid()
            gpu.update(owned_mib=gpu['used_mib'], other_mib=None, attribution_complete=False,
                owned_processes=[{'pid': pid, 'used_mib': gpu['used_mib'], 'category': owned.get(pid, OWNED)}])
            result.append(gpu)
            continue
        # A vendor's process list counts an iGPU's dedicated carve-out only, never its share of the pool: incomplete there.
        complete = None if source_row.get('shared_pool') else next((processes for whole, processes in found if whole), None)
        processes = complete if complete is not None else [p for whole, processes in found for p in processes]
        result += reconcile([gpu], [{**p, 'gpu': gpu['key']} for p in processes], owned, complete=complete is not None)
    # The GPU the model runs on: llama.cpp's main device, else its only model device, else the only GPU there is.
    llm = [gpu for gpu in result if gpu['llm']]
    main = next((gpu for gpu in result if gpu['main']), None) or (llm[0] if len(llm) == 1 else None) or (result[0] if len(result) == 1 else None)
    main_key = main['key'] if main else None
    payload = {'available': bool(result), 'gpus': result, 'main_key': main_key, 'ggml_devices': deepcopy(ggml),
        'source': ' + '.join(reading.label for reading in readings), 'errors': dict(errors or {}), 'sampled_at': time.time(),
        'note': 'Other includes external providers/TTS, other applications and driver/unattributed allocations. Unknown attribution is not treated as zero.'}
    if not result:
        payload['error'] = '; '.join(f'{name}: {message}' for name, message in (errors or {}).items()) or (
            'No GPU telemetry here: no nvidia-smi, no driver counters this OS offers, and the native library reports no GPU '
            '(an older riko-native without riko_devices, the model not loaded yet, or another provider)')
    return payload


class GPUMonitor:
    def __init__(self, sources=None):
        self.sources = platform_sources() if sources is None else list(sources)  # the bridge joins per sample (sample's provider)
        self.lock = threading.RLock()
        self.cached = None
        self.last = -float('inf')
        self.electron = {}
        self.electron_identity = {}
        self.electron_updated = -float('inf')
        self.watchers = 0
        self.watch_stop = None

    def register_electron(self, rows):
        if not isinstance(rows, list) or len(rows) > 64: raise ValueError('Invalid Electron process list')
        mapped = {}
        for row in rows:
            if not isinstance(row, dict) or type(row.get('pid')) is not int or row['pid'] <= 0 or row.get('kind') not in {'Browser', 'GPU', 'Renderer', 'Utility', 'Zygote', 'Sandbox helper'}:
                raise ValueError('Invalid Electron process')
            mapped[row['pid']] = 'Electron ' + row['kind']
        with self.lock:
            self.electron = mapped
            self.electron_identity = {pid: process_identity(pid) for pid in mapped}
            self.electron_updated = time.monotonic()

    def sample(self, provider=None):
        with self.lock:
            if self.cached is not None and time.monotonic() - self.last < 3: return deepcopy(self.cached)
            owned = {os.getpid(): OWNED}
            for pid, category in self.electron.items():
                identity = self.electron_identity.get(pid)
                if identity is not None and process_identity(pid) == identity: owned[pid] = category
            readings, errors = [], {}
            for source in [*self.sources, *([BridgeSource(provider)] if provider is not None else [])]:  # the bridge fills gaps
                try: reading = source.read(owned)
                except Exception as exc:  # a source that exists here but failed: shown, and the others still count
                    errors[source.name] = str(exc) or type(exc).__name__
                    continue
                if reading is not None: readings.append(reading)
            self.cached = merge(readings, owned, errors)
            self.last = time.monotonic()
            return deepcopy(self.cached)

    def subscribe(self, provider):
        with self.lock:
            self.watchers += 1
            if self.watch_stop is None:
                stop = self.watch_stop = threading.Event()
                def watch():
                    from ..events.bus import event_bus
                    previous = None
                    while not stop.is_set():
                        value = self.sample(provider())
                        key = json.dumps({k:v for k,v in value.items() if k != 'sampled_at'}, sort_keys=True)
                        if key != previous:
                            previous = key
                            event_bus.publish('resource.gpu', **value)
                        stop.wait(3)
                threading.Thread(target=watch, daemon=True, name='gpu-telemetry').start()
        def unsubscribe():
            with self.lock:
                self.watchers -= 1
                if self.watchers == 0:
                    self.watch_stop.set()
                    self.watch_stop = None
        return unsubscribe
