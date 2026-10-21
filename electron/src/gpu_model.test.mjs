import assert from 'node:assert/strict';
import {test} from 'node:test';
import {gpuKey,mainGpu,gpuTitle,memoryCaption,telemetryErrors} from './gpu_model.mjs';

const metal={key:'ggml:MTL0',index:0,name:'Apple M5 Pro',backend:'MTL',ggml_device:'MTL0',llm:true,main:true,memory_kind:'unified',memory_scope:'process',total_mib:38339,used_mib:548,free_mib:37791};
const cuda=(index,extra={})=>({key:`pci:0000:0${index+1}:00.0`,index,name:'NVIDIA GeForce RTX 4090',backend:'CUDA',ggml_device:`CUDA${index}`,memory_kind:'dedicated',memory_scope:'device',total_mib:24564,used_mib:4000,free_mib:20564,...extra});

test('a row keeps a stable key from whatever identity its sources gave it',()=>{
  assert.equal(gpuKey(metal),'ggml:MTL0');
  assert.equal(gpuKey({uuid:'GPU-1',index:0}),'GPU-1');  // an older backend's NVML row
  assert.equal(gpuKey({pci_bus_id:'0000:01:00.0',index:1}),'0000:01:00.0');
  assert.equal(gpuKey({index:2}),'gpu-2');
});

test('the model GPU is the main_key row, else the only model row, else the only row, never simply the first',()=>{
  const gpus=[cuda(0),cuda(1,{llm:true})];
  assert.equal(mainGpu({gpus,main_key:'pci:0000:01:00.0'}),gpus[0]);
  assert.equal(mainGpu({gpus,main_key:null}),gpus[1]);
  assert.equal(mainGpu({gpus:[cuda(0),cuda(1)],main_key:null}),null);  // two GPUs and no word on which runs the model
  assert.equal(mainGpu({gpus:[metal]}),metal);
  assert.equal(mainGpu(null),null);
});

test('unified and per-process memory say what the numbers mean',()=>{
  assert.equal(memoryCaption(metal),'Metal working-set budget 38,339 MiB · this app 548 MiB · other apps share this RAM and macOS does not report their GPU use');
  assert.match(memoryCaption({...metal,memory_scope:'device',backend:'Vulkan'}),/^Integrated GPU: shares system RAM/);
  assert.equal(memoryCaption({...cuda(0),free_mib:null}),'Total: 24,564 MiB · Used: 4,000 MiB · Free: unknown');
  assert.equal(gpuTitle(metal),'Apple M5 Pro · MTL MTL0');
  assert.equal(gpuTitle({name:'AMD Radeon',index:1}),'AMD Radeon · GPU 1');
  assert.deepEqual(telemetryErrors({errors:{'nvidia-smi':'driver mismatch'}}),['nvidia-smi: driver mismatch']);
  assert.deepEqual(telemetryErrors({}),[]);
});
