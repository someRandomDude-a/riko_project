"""Read-only GPU telemetry. Never load a model to measure it."""
import csv
from copy import deepcopy
import io
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time

MIB = 1024 ** 2


def run(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=5,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True).stdout


def number(value):
    try:
        result = float(value.strip())
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, AttributeError): return None


def process_identity(pid):
    """Reject stale Electron PID registrations after exit/PID reuse."""
    if os.name != 'nt':
        try: return os.stat(f'/proc/{pid}').st_ctime_ns
        except OSError: return None
    import ctypes as c
    kernel = c.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [c.c_uint32, c.c_int, c.c_uint32], c.c_void_p
    kernel.CloseHandle.argtypes = [c.c_void_p]
    kernel.GetProcessTimes.argtypes = [c.c_void_p, *([c.POINTER(c.c_uint64)] * 4)]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle: return None
    try:
        created, exited, system, user = [c.c_uint64() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, c.byref(created), c.byref(exited), c.byref(system), c.byref(user)): return None
        return created.value
    finally: kernel.CloseHandle(handle)


def nvidia_adapter_luids():
    """DXGI adapter IDs let Windows process counters exclude integrated GPUs."""
    import ctypes as c
    from ctypes import wintypes as w
    class GUID(c.Structure):
        _fields_ = [('a', w.DWORD), ('b', w.WORD), ('d', w.WORD), ('tail', c.c_ubyte * 8)]
    class LUID(c.Structure):
        _fields_ = [('low', w.DWORD), ('high', w.LONG)]
    class DESC(c.Structure):
        _fields_ = [('description', w.WCHAR * 128), ('vendor', w.UINT), ('device', w.UINT), ('subsystem', w.UINT), ('revision', w.UINT),
            ('video', c.c_size_t), ('system', c.c_size_t), ('shared', c.c_size_t), ('luid', LUID), ('flags', w.UINT)]
    def method(pointer, index, args):
        table = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(c.c_long, c.c_void_p, *args)(table[index])
    factory = c.c_void_p()
    iid = GUID(0x770aae78, 0xf26f, 0x4dba, (c.c_ubyte * 8)(0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87))
    create = c.WinDLL('dxgi').CreateDXGIFactory1
    create.argtypes, create.restype = [c.POINTER(GUID), c.POINTER(c.c_void_p)], c.c_long
    if create(c.byref(iid), c.byref(factory)) < 0: return []
    result = []
    try:
        for index in range(32):
            adapter = c.c_void_p()
            if method(factory, 12, [w.UINT, c.POINTER(c.c_void_p)])(factory, index, c.byref(adapter)) < 0: break
            try:
                desc = DESC()
                if method(adapter, 10, [c.POINTER(DESC)])(adapter, c.byref(desc)) >= 0 and desc.vendor == 0x10de:
                    result.append((desc.luid.high & 0xffffffff, desc.luid.low))
            finally: method(adapter, 2, [])(adapter)
    finally: method(factory, 2, [])(factory)
    return result


def reconcile(gpus, processes, owned, *, complete=False):
    """Do not fabricate attribution when the driver hides per-process memory."""
    result = deepcopy(gpus)
    for gpu in result:
        rows = [p for p in processes if p['gpu_uuid'] == gpu['uuid']]
        unknown = any(row['used_mib'] is None for row in rows)
        ours = [row for row in rows if row['pid'] in owned]
        known = sum(row['used_mib'] or 0 for row in ours)
        attributable = complete and not unknown and sum(row['used_mib'] or 0 for row in rows) <= gpu['used_mib'] + 64
        gpu.update(owned_mib=known if attributable else None,
            other_mib=max(0, gpu['used_mib'] - known) if attributable else None,
            attribution_complete=attributable,
            owned_processes=[{**row, 'category': owned[row['pid']]} for row in ours])
    return result


class GPUMonitor:
    def __init__(self):
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
            owned = {os.getpid(): 'Python (ASR and runtime)'}
            process = getattr(provider, 'process', None)
            if process and process.poll() is None: owned[process.pid] = 'Managed llama-server'
            for pid, category in self.electron.items():
                identity = self.electron_identity.get(pid)
                if identity is not None and process_identity(pid) == identity: owned[pid] = category
            try:
                executable = shutil.which('nvidia-smi')
                if not executable: raise RuntimeError('nvidia-smi unavailable; live NVIDIA telemetry cannot be read')
                data = run([executable, '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'])
                gpus = []
                for row in csv.reader(io.StringIO(data)):
                    if len(row) != 6: continue
                    index, uuid, name, total, used, free = [v.strip() for v in row]
                    if any(number(v) is None for v in (total, used, free)): continue
                    gpus.append({'index': int(index), 'uuid': uuid, 'name': name,
                        'total_mib': number(total), 'used_mib': number(used), 'free_mib': number(free)})
                processes, complete, source = [], False, 'nvidia-smi (compute processes only)'
                try:
                    data = run([executable, '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'])
                    for row in csv.reader(io.StringIO(data)):
                        if len(row) == 3 and row[1].strip().isdigit():
                            processes.append({'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'used_mib': number(row[2])})
                except (OSError, subprocess.SubprocessError): pass
                # WDDM compute memory is often N/A. Use LOCAL (resident video)
                # usage, not Dedicated Usage (which can exceed physical residency).
                # Match DXGI NVIDIA LUIDs to avoid counting integrated GPU memory.
                if os.name == 'nt' and len(gpus) == 1:
                    try:
                        luids = nvidia_adapter_luids()
                        if len(luids) != 1: raise ValueError('NVIDIA adapter mapping unavailable')
                        script = r"$ErrorActionPreference='Stop'; @( (Get-Counter '\GPU Process Memory(*)\Local Usage').CounterSamples | ForEach-Object { @{instance=$_.InstanceName;bytes=$_.CookedValue} } ) | ConvertTo-Json -Compress"
                        rows = json.loads(run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script]))
                        if isinstance(rows, dict): rows = [rows]
                        totals = {}
                        for row in rows:
                            match = re.search(r'pid_(\d+)_', row['instance'], re.IGNORECASE)
                            luid = re.search(r'luid_0x([0-9a-f]+)_0x([0-9a-f]+)_', row['instance'], re.IGNORECASE)
                            if match and luid and (int(luid[1], 16), int(luid[2], 16)) == luids[0]:
                                value = float(row['bytes'])
                                if not math.isfinite(value) or value < 0: raise ValueError('Invalid process memory counter')
                                totals[int(match[1])] = totals.get(int(match[1]), 0) + value / MIB
                        if totals:
                            processes = [{'gpu_uuid': gpus[0]['uuid'], 'pid': pid, 'used_mib': value} for pid, value in totals.items()]
                            complete, source = True, 'Windows local resident GPU process counters (DXGI adapter-matched)'
                    except (OSError, ValueError, KeyError, subprocess.SubprocessError): pass
                self.cached = {'available': bool(gpus), 'gpus': reconcile(gpus, processes, owned, complete=complete),
                    'source': source, 'sampled_at': time.time(),
                    'note': 'Other includes external providers/TTS, other applications and driver/unattributed allocations. Unknown attribution is not treated as zero.'}
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                self.cached = {'available': False, 'gpus': [], 'error': str(exc), 'sampled_at': time.time()}
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
