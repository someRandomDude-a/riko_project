"""Where GPU telemetry comes from. Each source reads what one interface reports and nothing more; gpu_memory.merge joins
their readings into one row per physical GPU. Read-only: never load a model or allocate GPU memory to measure.

- BridgeSource: llama.cpp's own devices (provider.devices(), riko_devices): CUDA, ROCm, Vulkan and Metal, free to ask.
  Metal's figures are this process's working set (memory_scope 'process'), not the GPU's.
- NvidiaSmi: nvidia-smi on PATH (any OS); its compute-process list misses graphics processes, so attribution is incomplete.
- AmdgpuSysfs and DrmFdinfo (Linux): the amdgpu driver's VRAM counters, and the DRM memory of Riko's own processes.
- WindowsCounters (Windows): every DXGI adapter, its PCI address, and the per-process GPU counters (PDH, English names).
"""
from dataclasses import dataclass, field
import csv
import io
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

MIB = 1024 ** 2
logger = logging.getLogger(__name__)


@dataclass
class Reading:
    """One source's answer. devices: rows of index, name, uuid, pci_bus_id, luid, backend, ggml_device, ggml_index, llm,
    main, memory_kind, memory_scope, total_mib, used_mib, free_mib (None when it does not say), and its own `id`.
    processes: {'device': a device id, 'pid', 'used_mib'}. complete: ids of the devices whose process list is whole."""
    label: str
    devices: list = field(default_factory=list)
    processes: list = field(default_factory=list)
    complete: set = field(default_factory=set)
    ggml_devices: list | None = None


def run(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=5,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True).stdout


def number(value):
    try:
        result = float(value.strip())
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, AttributeError): return None


def pci_address(text):
    """dddd:bb:dd.f, lowercase, from nvidia-smi's 00000000:01:00.0, ggml's 0000:01:00.0 or sysfs's; None otherwise."""
    match = re.fullmatch(r'(?:([0-9a-fA-F]{1,8}):)?([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\.([0-7])', (text or '').strip())
    if not match: return None
    return f'{int(match[1] or "0", 16) & 0xffff:04x}:{int(match[2], 16):02x}:{int(match[3], 16):02x}.{match[4]}'


class BridgeSource:
    """provider.devices(): ggml's view. GPU and integrated-GPU devices only (the CPU device reports physical RAM as free)."""
    name = 'llama.cpp'
    label = 'llama.cpp devices (riko_devices)'

    def __init__(self, provider): self.provider = provider

    def read(self, owned):
        devices = self.provider.devices()
        if devices is None: return None
        rows = []
        for device in devices:
            if device['type'] not in ('gpu', 'igpu'): continue
            metal, shared = device['backend'] == 'MTL', device['type'] == 'igpu'
            # Vulkan without VK_EXT_memory_budget reports the heap size as free, and its unsigned budget - usage wraps past
            # the total when a heap is over budget (Metal clamps that to 0): no usable "used" figure from either.
            unknown = (device['backend'] == 'Vulkan' and device['memory_free'] == device['memory_total']) or device['memory_free'] > device['memory_total']
            total = device['memory_total'] / MIB or None
            row = {'id': device['index'], 'index': device['index'], 'name': device['description'] or device['name'],
                'pci_bus_id': pci_address(device['device_id']), 'backend': device['backend'], 'ggml_device': device['name'],
                'ggml_index': device['index'], 'llm': device.get('model'), 'main': bool(device.get('main')),
                'memory_kind': 'unified' if metal or shared else None, 'memory_scope': 'process' if metal else 'device',
                'total_mib': total, 'free_mib': None if unknown or not total else device['memory_free'] / MIB,
                'used_mib': None if unknown or not total else (device['memory_total'] - device['memory_free']) / MIB}
            # An integrated GPU's vendor figures are its small dedicated carve-out (amdgpu mem_info_vram_*, DXGI
            # DedicatedVideoMemory); llama.cpp budgets the shared pool ggml reports, so ggml's memory figures replace theirs.
            if shared and total: row['replace'] = ('total_mib', 'used_mib', 'free_mib')
            rows.append(row)
        return Reading(self.label, rows, ggml_devices=devices)


class NvidiaSmi:
    name = 'nvidia-smi'
    label = 'nvidia-smi (compute processes only)'

    def __init__(self, which=shutil.which, run=run): self.which, self.run = which, run

    def read(self, owned):
        executable = self.which('nvidia-smi')
        if not executable: return None
        data = self.run([executable, '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,pci.bus_id', '--format=csv,noheader,nounits'])
        reading = Reading(self.label)
        for row in csv.reader(io.StringIO(data)):
            if len(row) != 7: continue
            index, uuid, name, total, used, free, bus = [value.strip() for value in row]
            if not index.isdigit() or any(number(value) is None for value in (total, used, free)): continue
            reading.devices.append({'id': uuid, 'index': int(index), 'uuid': uuid, 'name': name, 'pci_bus_id': pci_address(bus),
                'memory_scope': 'device', 'total_mib': number(total), 'used_mib': number(used), 'free_mib': number(free)})
        try:  # WDDM often reports N/A for compute memory, and graphics processes (Electron) are never listed
            data = self.run([executable, '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'])
            for row in csv.reader(io.StringIO(data)):
                if len(row) == 3 and row[1].strip().isdigit():
                    reading.processes.append({'device': row[0].strip(), 'pid': int(row[1]), 'used_mib': number(row[2])})
        except (OSError, subprocess.SubprocessError): pass
        return reading if reading.devices else None


class AmdgpuSysfs:
    """The amdgpu driver's VRAM counters (no ROCm, no subprocess): /sys/class/drm/card<N>/device/mem_info_vram_*."""
    name = 'amdgpu'
    label = 'amdgpu sysfs'

    def __init__(self, root='/sys'): self.root = Path(root)

    def read(self, owned):
        reading = Reading(self.label)
        for card in sorted(self.root.glob('class/drm/card*'), key=lambda path: path.name):
            if not re.fullmatch(r'card\d+', card.name): continue  # card0-DP-1 is a connector
            device = card / 'device'
            try:
                if (device / 'vendor').read_text().strip().lower() != '0x1002': continue
                total, used = (int((device / f'mem_info_vram_{key}').read_text().strip()) for key in ('total', 'used'))
            except (OSError, ValueError): continue
            try: name = (device / 'product_name').read_text().strip()
            except OSError: name = ''
            reading.devices.append({'id': card.name, 'index': int(card.name[4:]), 'name': name or 'AMD GPU',
                'pci_bus_id': pci_slot(device), 'memory_scope': 'device',
                'total_mib': total / MIB, 'used_mib': used / MIB, 'free_mib': max(0, total - used) / MIB})
        return reading if reading.devices else None


def pci_slot(device):
    """A DRM card's PCI address: PCI_SLOT_NAME in its uevent, else the name its `device` link resolves to."""
    try:
        for line in (device / 'uevent').read_text().splitlines():
            if line.startswith('PCI_SLOT_NAME='): return pci_address(line.split('=', 1)[1])
    except OSError: pass
    return pci_address(device.resolve().name)


class DrmFdinfo:
    """Riko's own processes' GPU memory from /proc/<pid>/fdinfo (drm-pdev and drm-memory-vram or drm-resident-*),
    counted once per DRM client. Complete only when every owned process could be read: Chromium's sandboxed helpers
    deny it, which leaves their usage unknown rather than zero."""
    name = 'drm-fdinfo'
    label = 'DRM fdinfo (owned processes)'
    KEYS = ('drm-memory-vram', 'drm-resident-vram', 'drm-resident-vram0', 'drm-resident-local0')
    UNITS = {'': 1, 'kib': 1024, 'mib': MIB, 'gib': 1024 * MIB}

    def __init__(self, proc='/proc'): self.proc = Path(proc)

    def read(self, owned):
        totals, clients, readable = {}, set(), True
        for pid in owned:
            try: entries = list((self.proc / str(pid) / 'fdinfo').iterdir())
            except OSError: readable = False; continue
            for entry in entries:
                try: text = entry.read_text(errors='replace')
                except OSError: continue
                fields = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
                fields = {key.strip(): value.strip() for key, value in fields.items()}
                device, client = pci_address(fields.get('drm-pdev')), fields.get('drm-client-id')
                key = next((key for key in self.KEYS if key in fields), None)
                if not device or key is None or (device, client) in clients: continue
                clients.add((device, client))
                match = re.fullmatch(r'(\d+)\s*([KMG]iB)?', fields[key], re.IGNORECASE)
                if not match: continue
                value = int(match[1]) * self.UNITS[(match[2] or '').lower()] / MIB
                totals[device, pid] = totals.get((device, pid), 0) + value
        if not totals: return None
        reading = Reading(self.label, [{'id': device, 'pci_bus_id': device} for device in sorted({device for device, _ in totals})])
        reading.processes = [{'device': device, 'pid': pid, 'used_mib': value} for (device, pid), value in totals.items()]
        if readable: reading.complete = {device for device, _ in totals}
        return reading


class WindowsCounters:
    """Every hardware DXGI adapter (any vendor) with its PCI address (D3DKMT), and the GPU Adapter Memory and GPU Process
    Memory performance counters through PDH by their English names, so a localized Windows reads them too. One PDH query
    is opened on first use and kept: no PowerShell, no process per sample."""
    name = 'windows'
    label = 'Windows GPU counters (DXGI + PDH)'

    def __init__(self, adapters=None):
        self.adapters = adapters or windows_adapters
        self.query = self.counters = self.pdh = None
        self.unavailable = ''

    def read(self, owned):
        adapters = self.adapters()
        if not adapters: return None
        reading = Reading(self.label)
        for index, adapter in enumerate(adapters):
            reading.devices.append({'id': adapter['luid'], 'index': index, 'name': adapter['name'], 'luid': adapter['luid'],
                'pci_bus_id': adapter['pci_bus_id'], 'memory_scope': 'device', 'total_mib': adapter['dedicated'] / MIB or None})
        values = self.sample()
        if values is None: return reading
        adapter_usage, process_usage = values
        luids = {adapter['luid'] for adapter in adapters}
        for row in reading.devices:
            used = adapter_usage.get(row['luid'])
            if used is not None and row['total_mib']: row.update(used_mib=used / MIB, free_mib=max(0, row['total_mib'] - used / MIB))
        if process_usage is None: return reading  # the process counter failed this time: usage, but no attribution
        totals = {}
        for (pid, luid), value in process_usage.items():
            if luid in luids: totals[luid, pid] = totals.get((luid, pid), 0) + value / MIB
        reading.processes = [{'device': luid, 'pid': pid, 'used_mib': value} for (luid, pid), value in totals.items()]
        reading.complete = {row['luid'] for row in reading.devices if row['luid'] in adapter_usage}  # PDH lists every process
        return reading

    def sample(self):
        """({luid: dedicated bytes in use}, {(pid, luid): local bytes} or None when that counter could not be read) from
        PDH, or None when the counters are missing or the adapter counter could not be read."""
        if self.unavailable: return None
        import ctypes as c
        if self.pdh is None:
            try: pdh = c.WinDLL('pdh')
            except OSError as exc: return self.disable(f'pdh.dll: {exc}')
            pdh.PdhOpenQueryW.argtypes = [c.c_wchar_p, c.c_size_t, c.POINTER(c.c_void_p)]
            pdh.PdhAddEnglishCounterW.argtypes = [c.c_void_p, c.c_wchar_p, c.c_size_t, c.POINTER(c.c_void_p)]
            pdh.PdhCollectQueryData.argtypes = pdh.PdhCloseQuery.argtypes = [c.c_void_p]
            for function in (pdh.PdhOpenQueryW, pdh.PdhAddEnglishCounterW, pdh.PdhCollectQueryData, pdh.PdhCloseQuery): function.restype = c.c_uint32
            self.pdh = pdh
        pdh = self.pdh
        if self.query is None:
            query = c.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, c.byref(query)) != 0: return self.disable('PdhOpenQueryW failed')
            counters = []
            for path in (r'\GPU Adapter Memory(*)\Dedicated Usage', r'\GPU Process Memory(*)\Local Usage'):
                counter = c.c_void_p()
                status = pdh.PdhAddEnglishCounterW(query, path, 0, c.byref(counter))
                if status != 0:  # PDH_CSTATUS_NO_OBJECT before WDDM 2.0, or the counters are disabled
                    pdh.PdhCloseQuery(query)
                    return self.disable(f'{path} unavailable (PDH status 0x{status:08x})')
                counters.append(counter)
            self.query, self.counters = query, counters
        if pdh.PdhCollectQueryData(self.query) != 0: return None
        adapter, process = pdh_values(pdh, self.counters[0]), pdh_values(pdh, self.counters[1])
        if adapter is None: return None
        usage, owners = counter_totals(adapter, process or [])
        return usage, owners if process is not None else None

    def disable(self, reason):
        self.unavailable = reason
        logger.debug('Windows GPU process counters unavailable: %s', reason)
        return None


def counter_totals(adapter_values, process_values):
    """({luid: bytes}, {(pid, luid): bytes}) from PDH instances: GPU Adapter Memory's luid_0x<high>_0x<low>_phys_<n>, and
    GPU Process Memory's pid_<pid>_luid_0x<high>_0x<low>_phys_<n> (summed over the adapter's physical parts)."""
    adapter, process = {}, {}
    for values, target in ((adapter_values, adapter), (process_values, process)):
        for name, value in values:
            luid = re.search(r'luid_0x([0-9a-f]+)_0x([0-9a-f]+)', name, re.IGNORECASE)
            if not luid or not math.isfinite(value) or value < 0: continue
            key = f'{int(luid[1], 16):08x}:{int(luid[2], 16):08x}'
            if target is process:
                pid = re.search(r'pid_(\d+)_', name, re.IGNORECASE)
                if pid: process[int(pid[1]), key] = process.get((int(pid[1]), key), 0) + value
            else: adapter[key] = adapter.get(key, 0) + value
    return adapter, process


def pdh_values(pdh, counter):
    """(instance name, value) of a wildcard counter, as PDH_FMT_LARGE: [] when it has no instances, None when PDH failed
    (unknown, never read as nobody using the GPU)."""
    import ctypes as c
    from ctypes import wintypes as w
    class Value(c.Structure):
        class Union(c.Union): _fields_ = [('large', c.c_longlong), ('double', c.c_double)]
        _fields_ = [('status', w.DWORD), ('value', Union)]
    class Item(c.Structure): _fields_ = [('name', c.c_wchar_p), ('value', Value)]
    pdh.PdhGetFormattedCounterArrayW.argtypes = [c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.POINTER(w.DWORD), c.c_void_p]
    pdh.PdhGetFormattedCounterArrayW.restype = c.c_uint32
    size, count = w.DWORD(0), w.DWORD(0)
    status = pdh.PdhGetFormattedCounterArrayW(counter, 0x400, c.byref(size), c.byref(count), None)
    if (status == 0 and not count.value) or status in (0x800007D1, 0x800007D5): return []  # no instances (PDH_CSTATUS_NO_INSTANCE, PDH_NO_DATA)
    if status != 0x800007D2 or not size.value: return None  # PDH_MORE_DATA tells the size; anything else failed
    buffer = (c.c_byte * size.value)()
    if pdh.PdhGetFormattedCounterArrayW(counter, 0x400, c.byref(size), c.byref(count), buffer) != 0: return None
    items = c.cast(buffer, c.POINTER(Item))
    return [(items[i].name or '', float(items[i].value.value.large)) for i in range(count.value) if items[i].value.status in (0, 1)]


def windows_adapters():
    """Hardware DXGI adapters: name, vendor, dedicated bytes, LUID ('high:low' in hex) and PCI address (D3DKMT), skipping
    software adapters (DXGI_ADAPTER_FLAG_SOFTWARE, Microsoft's 0x1414 Basic Render Driver)."""
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
    iid = GUID(0x770aae78, 0xf26f, 0x4dba, (c.c_ubyte * 8)(0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87))  # IDXGIFactory1
    create = c.WinDLL('dxgi').CreateDXGIFactory1
    create.argtypes, create.restype = [c.POINTER(GUID), c.POINTER(c.c_void_p)], c.c_long
    if create(c.byref(iid), c.byref(factory)) < 0: return []
    result = []
    try:
        for index in range(32):
            adapter = c.c_void_p()
            if method(factory, 12, [w.UINT, c.POINTER(c.c_void_p)])(factory, index, c.byref(adapter)) < 0: break  # EnumAdapters1
            try:
                desc = DESC()
                if method(adapter, 10, [c.POINTER(DESC)])(adapter, c.byref(desc)) < 0 or desc.flags & 2 or desc.vendor == 0x1414: continue  # GetDesc1
                luid = (desc.luid.high & 0xffffffff, desc.luid.low)
                result.append({'name': desc.description, 'vendor': desc.vendor, 'dedicated': desc.video, 'luid': f'{luid[0]:08x}:{luid[1]:08x}',
                    'pci_bus_id': adapter_address(desc.luid)})
            finally: method(adapter, 2, [])(adapter)
    finally: method(factory, 2, [])(factory)
    return result


def adapter_address(luid):
    """The adapter's PCI bus/device/function (D3DKMTQueryAdapterInfo KMTQAITYPE_ADAPTERADDRESS); None when the kernel
    thunk is missing or refuses. D3DKMT does not report the PCI domain: 0000 is assumed (multi-domain PCs are servers)."""
    import ctypes as c
    from ctypes import wintypes as w
    class Open(c.Structure): _fields_ = [('luid', type(luid)), ('adapter', w.UINT)]
    class Address(c.Structure): _fields_ = [('bus', w.UINT), ('device', w.UINT), ('function', w.UINT)]
    class Query(c.Structure): _fields_ = [('adapter', w.UINT), ('type', c.c_int), ('data', c.c_void_p), ('size', w.UINT)]
    class Close(c.Structure): _fields_ = [('adapter', w.UINT)]
    try:
        gdi = c.WinDLL('gdi32')
        thunks = gdi.D3DKMTOpenAdapterFromLuid, gdi.D3DKMTQueryAdapterInfo, gdi.D3DKMTCloseAdapter
    except (OSError, AttributeError): return None
    for thunk in thunks: thunk.argtypes, thunk.restype = [c.c_void_p], c.c_long
    opened, address = Open(luid, 0), Address()
    if thunks[0](c.byref(opened)) != 0: return None
    try:
        if thunks[1](c.byref(Query(opened.adapter, 6, c.addressof(address), c.sizeof(address)))) != 0: return None
        return adapter_pci(address.bus, address.device, address.function)
    finally: thunks[2](c.byref(Close(opened.adapter)))


def adapter_pci(bus, device, function):
    return f'0000:{bus & 0xff:02x}:{device & 0x1f:02x}.{function & 7}'


def platform_sources(platform=sys.platform):
    """The sources this OS can have, besides the bridge (GPUMonitor adds that per sample, from the provider)."""
    sources = [NvidiaSmi()]
    if platform.startswith('linux'): sources += [AmdgpuSysfs(), DrmFdinfo()]
    if platform == 'win32': sources.append(WindowsCounters())
    return sources


def process_identity(pid, platform=None):
    """When this pid's process started, to reject a stale Electron PID registration after exit or PID reuse; None when
    it cannot be read (gone, another user's, or an OS without a way to ask)."""
    import ctypes as c
    platform = platform or sys.platform
    if platform == 'win32':
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
    if platform == 'darwin':  # libproc's proc_bsdinfo (136 bytes): pbi_pid at 12, pbi_start_tvsec/usec at 120/128
        try: libproc = c.CDLL('/usr/lib/libproc.dylib')
        except OSError: return None
        libproc.proc_pidinfo.argtypes, libproc.proc_pidinfo.restype = [c.c_int, c.c_int, c.c_uint64, c.c_void_p, c.c_int], c.c_int
        info = (c.c_ubyte * 136)()
        if libproc.proc_pidinfo(pid, 3, 0, info, 136) != 136: return None  # PROC_PIDTBSDINFO
        started = int.from_bytes(bytes(info[120:128]), 'little') * 1_000_000 + int.from_bytes(bytes(info[128:136]), 'little')
        return started if int.from_bytes(bytes(info[12:16]), 'little') == pid else None
    if not platform.startswith('linux'): return None
    try: return os.stat(f'/proc/{pid}').st_ctime_ns
    except OSError: return None
