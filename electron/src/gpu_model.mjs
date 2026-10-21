// GPU telemetry rows (/ws/resources/gpu, app_core/resources/gpu_memory.py) for the GPU memory panel: a row's stable key,
// the GPU the model runs on, and how to describe memory that is unified (Apple, integrated GPUs) or counted per process (Metal).
const mib=value=>value==null?'unknown':`${Math.round(value).toLocaleString('en-US')} MiB`;

export const gpuKey=gpu=>gpu?.key||gpu?.uuid||gpu?.pci_bus_id||`gpu-${gpu?.index??'unknown'}`;

export function mainGpu(live){
  const gpus=live?.gpus||[];
  return gpus.find(gpu=>live.main_key&&gpuKey(gpu)===live.main_key)||(gpus.filter(gpu=>gpu.llm).length===1?gpus.find(gpu=>gpu.llm):null)
    ||(gpus.length===1?gpus[0]:null);
}

export function gpuTitle(gpu){
  const device=[gpu.backend,gpu.ggml_device].filter(Boolean).join(' ');
  return `${gpu.name||'GPU'} · ${device||`GPU ${gpu.index??'?'}`}`;
}

export function memoryCaption(gpu){
  if(gpu?.memory_kind==='unified'&&gpu.memory_scope==='process')
    return `Metal working-set budget ${mib(gpu.total_mib)} · this app ${mib(gpu.used_mib)} · other apps share this RAM and macOS does not report their GPU use`;
  if(gpu?.memory_kind==='unified')return `Integrated GPU: shares system RAM · budget ${mib(gpu.total_mib)} · used ${mib(gpu.used_mib)}`;
  return `Total: ${mib(gpu?.total_mib)} · Used: ${mib(gpu?.used_mib)} · Free: ${mib(gpu?.free_mib)}`;
}

export const telemetryErrors=live=>Object.entries(live?.errors||{}).map(([source,message])=>`${source}: ${message}`);
