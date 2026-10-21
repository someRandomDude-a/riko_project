"""Which ggml devices llama.cpp puts a model on, from riko_devices' list: its default selection with no --device list
(llama_prepare_model_devices in src/llama.cpp, lines 185-300 at the pinned commit). Re-check it whenever the pin moves."""


def model_devices(devices, *, split_mode, main_gpu):
    """The devices (riko_devices entries, in ggml's order) the model loads onto: RPC servers first, then discrete GPUs (a
    device_id seen twice counts once), integrated GPUs only without a discrete one (those of the first iGPU's backend);
    split_mode none keeps only the main_gpu'th of them. [] means the CPU only; None when main_gpu names no device, which
    llama.cpp refuses to load. CPU and accelerator devices (BLAS) are never on the list."""
    rpc, gpus, igpus = [], [], []
    for device in devices:
        if device.get('type') == 'gpu':
            if device.get('backend') == 'RPC': rpc.append(device)
            elif not (device.get('device_id') and any(gpu.get('device_id') == device['device_id'] for gpu in gpus)): gpus.append(device)
        elif device.get('type') == 'igpu' and (not igpus or device.get('backend') == igpus[-1].get('backend')): igpus.append(device)
    chosen = rpc + gpus + ([] if gpus else igpus)
    if split_mode != 'none' or not chosen: return chosen
    if main_gpu < 0: return []
    return [chosen[main_gpu]] if main_gpu < len(chosen) else None
