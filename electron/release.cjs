const path=require('path');
const fs=require('fs');
const os=require('os');
const {execFile,spawn}=require('child_process');
const YAML=require('yaml');
const run=(file,args)=>new Promise(resolve=>execFile(file,args,{timeout:5000,windowsHide:true},(error,stdout)=>resolve(error?'':stdout.trim())));

async function hardware(){
 const gpu=await run('nvidia-smi',['--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits']);
 const vulkan=await run('vulkaninfo',['--summary']);
 return {cpu:os.cpus()[0]?.model||'Unknown',threads:os.cpus().length,ramGB:Math.round(os.totalmem()/2**30),nvidia:gpu,vulkan:vulkan||'Vulkan enumeration unavailable; driver support must be checked',preferred:gpu?'cuda':'vulkan'};
}
function nativeLibrary(resources,backend){return path.join(resources,'native',backend,process.platform==='win32'?'riko-native.dll':'libriko-native.so');}
function configuration(input,resources){
 if(input.sovitsAuto&&(!path.isAbsolute(input.sovitsExecutable||'')||!fs.existsSync(input.sovitsExecutable)))throw new Error('Choose an existing absolute GPT-SoVITS executable');
 if(!['cuda','vulkan'].includes(input.backend))throw new Error('Choose CUDA or Vulkan');
 const library=nativeLibrary(resources,input.backend);
 if(!fs.existsSync(library))throw new Error('Packaged native backend is missing: '+library);
 const context=Number(input.context),output=Number(input.output),threads=Number(input.threads);
 if(!Number.isInteger(context)||context<2048||context>131072||!Number.isInteger(output)||output<64||output>=context)throw new Error('Invalid context/output budget');
 if(!Number.isInteger(threads)||threads<1||threads>1024)throw new Error('Invalid CPU thread count');
 if(!input.modelPath&&(!input.repo||!input.filename||!input.filename.endsWith('.gguf')||input.filename.includes('..')||input.filename.startsWith('/')))throw new Error('Choose a local GGUF or exact Hugging Face repository/file');
 if(input.modelPath&&(!path.isAbsolute(input.modelPath)||!input.modelPath.toLowerCase().endsWith('.gguf')||!fs.existsSync(input.modelPath)))throw new Error('Local GGUF does not exist');
 return {runtime:{provider:'llama_cpp',native_library:'bundled:'+input.backend,model_path:input.modelPath||null,hf_repo_id:input.repo||null,hf_filename:input.filename||null,hf_revision:input.revision||'main',n_ctx:context,max_output_tokens:output,n_threads:threads,n_gpu_layers:input.cpuOnly?0:-1,parallel_slots:2,flash_attn:false,type_k:'f16',type_v:'f16',warmup:false},
  presets:{default:{name:input.name||'Riko',system_prompt:input.prompt||'You are a helpful local companion.'}},
  memory:{context_window_tokens:context,default_memories:String(input.memories||'').split('\n').filter(t=>t.trim()).map(text=>({text,memory_type:'factual',importance:.8})),embeddings_enabled:!!input.embeddings,system1_enabled:!!input.julia,reflection_enabled:!!input.reflection},
  emotion:{enabled:!!input.julia,device:'cpu',probe:{enabled:false}},voice:{asr_device:'cpu'},tools:{require_approval:true},initiative:{enabled:false},desktop:{setup_on_startup_error:true},
  sovits_ping_config:{auto_start:!!input.sovitsAuto,executable:input.sovitsExecutable||null,arguments:[],url:input.sovitsUrl||'http://127.0.0.1:9880/tts',ref_audio_path:input.referenceAudio||'',prompt_text:input.referenceText||'',text_lang:'en',prompt_lang:'en',sample_rate:32000}};
}
function saveSetup(directory,input,resources){
 if(!path.isAbsolute(directory))throw new Error('Choose an absolute data directory');
 const relative=path.relative(path.resolve(resources),path.resolve(directory));
 if(relative===''||relative!=='..'&&!relative.startsWith('..'+path.sep)&&!path.isAbsolute(relative))throw new Error('Choose a data folder outside installed application resources');
 const config=configuration(input,resources);
 fs.mkdirSync(directory,{recursive:true});
 for(const folder of ['models','persistent_memories','logs'])fs.mkdirSync(path.join(directory,folder),{recursive:true});
 // Never overwrite an existing user configuration, even after a failed first run.
 fs.writeFileSync(path.join(directory,'character_config.yaml'),YAML.stringify(config),{flag:'wx',mode:0o600});
 return directory;
}
function startBackend(directory,resources){
 const executable=path.join(resources,'backend',process.platform==='win32'?'riko-backend.exe':'riko-backend');
 const log=fs.openSync(path.join(directory,'logs','backend-launch.log'),'a');
 const env={...process.env,RIKO_MANAGED:'1',RIKO_DATA_DIR:directory,RIKO_CONFIG:path.join(directory,'character_config.yaml'),HF_HOME:path.join(directory,'models','huggingface'),TORCH_HOME:path.join(directory,'models','torch'),XDG_CACHE_HOME:path.join(directory,'models','cache'),RIKO_BUNDLE_ROOT:resources};
 let child;
 try{child=spawn(executable,[],{cwd:directory,env,stdio:['pipe',log,log],windowsHide:true});}finally{fs.closeSync(log);}
 child.stdin.on('error',()=>{});
 return child;
}
function startSovits(settings){
 if(!settings?.auto_start)return null;
 const executable=settings.executable,args=settings.arguments||[];
 if(!executable||!path.isAbsolute(executable)||!fs.statSync(executable).isFile())throw new Error('GPT-SoVITS executable is missing');
 if(!Array.isArray(args)||args.some(a=>typeof a!=='string'))throw new Error('GPT-SoVITS arguments must be a list of strings');
 return spawn(executable,args,{cwd:path.dirname(executable),stdio:'ignore',windowsHide:true,shell:false});
}
function stopBackend(child){return new Promise(resolve=>{if(!child||child.exitCode!==null){resolve();return;}const timeout=setTimeout(()=>{child.kill();resolve();},15000);child.once('exit',()=>{clearTimeout(timeout);resolve();});child.stdin.end('shutdown\n');});}
module.exports={hardware,configuration,saveSetup,startBackend,nativeLibrary,startSovits,stopBackend};
