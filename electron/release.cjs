const path=require('path');
const fs=require('fs');
const os=require('os');
const {execFile,spawn}=require('child_process');
const native=require('./native_backends.cjs');
const run=(file,args)=>new Promise(resolve=>execFile(file,args,{timeout:5000,windowsHide:true},(error,stdout)=>resolve(error?'':stdout.trim())));

// One backend's detection (native_backends.json): its command, then each fallback argv until one answers; with a pattern,
// only the matching output lines count (rocminfo lists the CPU too) and are what setup shows. '' when nothing found hardware.
// The targets a backend was built for, from its -DGPU_TARGETS= flag (detect.targets names the flag).
function builtTargets(backend){const flag=(backend.cmake_flags||[]).find(item=>item.startsWith(`-D${backend.detect.targets}=`));return new Set(flag?flag.split('=')[1].split(';'):[]);}
async function detect(backend,probe,exists){
 const spec=backend.detect,pattern=spec.pattern&&new RegExp(spec.pattern);
 // A runtime the bundle links (ROCm 7's libamdhip64.so.7) must be installed, or its module cannot load.
 if(spec.runtime&&!spec.runtime.some(file=>exists(file)))return '';
 for(const [command,...args] of [[spec.command,...(spec.args||[])],...(spec.fallbacks||[])]){
  const output=await probe(command,args);
  const found=pattern?[...new Set(output.split(/\r?\n/).map(line=>line.trim()).filter(line=>pattern.test(line)))]:output?[output]:[];
  if(!found.length)continue;
  // ggml-hip uses every GPU it finds: one it has no code for (an RX 6700 XT, a Ryzen iGPU) aborts the backend.
  if(spec.targets){const built=builtTargets(backend);if(!found.every(line=>built.has(line.match(/gfx[0-9a-f]+/)?.[0])))return '';}
  return found.join('\n');
 }
 return '';
}
// Runs only the detection tools of the bundles this installation holds (resources/native/<id>: macOS ships Metal and has
// neither nvidia-smi nor vulkaninfo; rocminfo runs only beside a hip bundle). An optional backend is offered only when its
// required detection found hardware: HIP needs an AMD GPU with ROCm, and Vulkan stays the choice for AMD otherwise.
async function hardware(resources,platform=process.platform,probe=run,exists=fs.existsSync){
 const installed=native.installedBackends(resources,platform),probed=installed.filter(backend=>backend.detect);
 const outputs=await Promise.all(probed.map(backend=>detect(backend,probe,exists)));
 const detected=Object.fromEntries(probed.map((backend,index)=>[backend.id,outputs[index]]));
 const hidden=backend=>backend.optional&&backend.detect?.required&&!detected[backend.id],offered=installed.filter(backend=>!hidden(backend));
 return {cpu:os.cpus()[0]?.model||'Unknown',threads:os.cpus().length,ramGB:Math.round(os.totalmem()/2**30),platform,
  backends:offered.map(({id,label,description})=>({id,label,description})),
  unavailable:installed.filter(hidden).map(backend=>({id:backend.id,label:backend.label,reason:backend.detect.missing})),
  detections:probed.map(backend=>({backend:backend.id,label:backend.detect.label,required:!!backend.detect.required,output:detected[backend.id]||backend.detect.missing})),
  preferred:native.preferredBackend(platform,detected,native.MANIFEST,offered)};
}
// The folder an update replaces and an uninstall deletes (NSIS ends with RMDir /r $INSTDIR): the .app bundle on
// macOS, the executable's folder elsewhere.
function installRoot(executable,platform=process.platform){return platform==='darwin'?path.resolve(executable,'..','..','..'):path.dirname(executable);}
function inside(parent,child){const relative=path.relative(path.resolve(parent),path.resolve(child));return relative===''||relative!=='..'&&!relative.startsWith('..'+path.sep)&&!path.isAbsolute(relative);}
// The packaged backend. Main runs it to serve the app (startBackend) and, on the first run, to turn the setup page's choices
// into its configuration (saveSetup).
function backendExecutable(resources,platform=process.platform){return path.join(resources,'backend',platform==='win32'?'riko-backend.exe':'riko-backend');}
// One answer from the backend's command line (Code/run_server.py): input as JSON on stdin, one line of JSON on stdout.
// command is [file, ...arguments]; development and tests pass Python and Code/run_server.py.
function backendAnswer(command,args,input,env,timeout=120000){
 return new Promise((resolve,reject)=>{
  let child,output='',errors='';
  try{child=spawn(command[0],[...command.slice(1),...args],{env,stdio:['pipe','pipe','pipe'],windowsHide:true});}catch(error){reject(error);return;}
  const timer=setTimeout(()=>{child.kill();reject(new Error('The backend did not answer in time'));},timeout);
  child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
  child.stdout.on('data',chunk=>{output+=chunk;});child.stderr.on('data',chunk=>{errors=(errors+chunk).slice(-2000);});
  child.on('error',error=>{clearTimeout(timer);reject(new Error('The backend could not start: '+error.message));});
  child.on('close',code=>{
   clearTimeout(timer);let answer=null;
   try{answer=JSON.parse(output.trim().split(/\r?\n/).at(-1));}catch{}
   if(!answer||typeof answer!=='object')reject(new Error('The backend gave no answer (exit '+code+')'+(errors.trim()?': '+errors.trim():'')));
   else if(answer.error)reject(new Error(answer.error));else resolve(answer);
  });
  child.stdin.on('error',()=>{});child.stdin.end(JSON.stringify(input));
 });
}
// First run: the backend turns the choices into the configuration it will load, over what a packaged build starts with, and
// checks it as Settings checks an edit (Code/process/app_core/configuration/first_run.py). Main keeps the data folder's
// rules and writes the file, never over an existing one, even after a failed first run.
async function saveSetup(directory,input,resources,install=resources,platform=process.platform,command=[backendExecutable(resources,platform)]){
 if(!path.isAbsolute(directory))throw new Error('Choose an absolute data directory');
 if([install,resources].some(root=>inside(root,directory)))throw new Error('Choose a data folder outside the application folder ('+path.resolve(install)+'); updates and uninstall delete everything inside it');
 const file=path.join(directory,'character_config.yaml');
 if(fs.existsSync(file))throw Object.assign(new Error(`EEXIST: file already exists, open '${file}'`),{code:'EEXIST'});
 fs.mkdirSync(directory,{recursive:true});
 const {config}=await backendAnswer(command,['--setup-config',directory],input,{...process.env,RIKO_BUNDLE_ROOT:resources});
 if(typeof config!=='string'||!config.trim())throw new Error('The backend returned no configuration');
 for(const folder of ['models','persistent_memories','logs'])fs.mkdirSync(path.join(directory,folder),{recursive:true});
 fs.writeFileSync(file,config,{flag:'wx',mode:0o600});
 return directory;
}
// The backend (Code/run_server.py) prints listeningLine(port) once it holds 127.0.0.1:<port>. Main sends the API token only
// after the line names the port it gave the backend, so a process that grabbed the port while the backend was still
// starting never sees this start's token.
const LISTENING='RIKO_BACKEND_LISTENING';
const listeningLine=port=>LISTENING+' port='+port;
function watchListening(port,onListening){
 let tail='';const line=listeningLine(port);
 return chunk=>{
  if(tail===null)return;
  tail=(tail+String(chunk)).slice(-4096);
  if(tail.split(/\r?\n/).slice(0,-1).includes(line)){tail=null;onListening();}
 };
}
// A free loopback port for the packaged backend. Should another process take it before the backend binds it, the backend
// exits after its retries without ever naming it, and main shows 'Backend stopped'.
function freePort(net=require('net')){return new Promise((resolve,reject)=>{const server=net.createServer();server.unref();server.once('error',reject);server.listen(0,'127.0.0.1',()=>{const {port}=server.address();server.close(()=>resolve(port));});});}
// Finder and the Dock start apps with launchd's PATH (/usr/bin:/bin:/usr/sbin:/sbin), which hides Homebrew's node,
// npx, uvx and ffmpeg from MCP servers and the Discord worker; put its prefixes first, as a Terminal does.
const HOMEBREW_PATHS=['/opt/homebrew/bin','/opt/homebrew/sbin','/usr/local/bin'];
function finderPath(current){const parts=String(current||'/usr/bin:/bin:/usr/sbin:/sbin').split(':').filter(Boolean);return [...HOMEBREW_PATHS.filter(dir=>!parts.includes(dir)),...parts].join(':');}
// macOS attributes the packaged backend's microphone use to this app: the hardened runtime needs the audio-input
// entitlement, and the prompt shows NSMicrophoneUsageDescription. Ask before the backend starts, so the prompt names Riko
// at launch instead of interrupting the first voice turn. Resolves true/false once answered, null when nothing was asked.
// 'blocked' when macOS already denies the microphone: the backend's input then opens but records silence.
function askMicrophone(preferences,platform=process.platform){
 if(platform!=='darwin')return Promise.resolve(null);
 const status=preferences.getMediaAccessStatus('microphone');
 if(status==='denied'||status==='restricted')return Promise.resolve('blocked');
 if(status!=='not-determined')return Promise.resolve(null);
 return Promise.resolve().then(()=>preferences.askForMediaAccess('microphone')).catch(()=>false);
}
// logs/backend-launch.log: the packaged backend's stdout and stderr, all a startup crash leaves ('Backend stopped' points
// here), written whatever logging.file_enabled says. Capped like debug.log: past maxBytes (and at open, when an older build
// left it larger) .log becomes .log.1, which becomes .log.2. Writes are synchronous (little volume; closing and renaming
// between writes behave the same on every OS) and never throw. Each stream is buffered by line, at most 64 KiB, so a secret
// split across chunks is still found: this start's API token and confirmation key are written as [REDACTED].
const LAUNCH_LOG_BYTES=5*1024*1024,LINE_BUFFER=64*1024;
function launchLog(file,{maxBytes=LAUNCH_LOG_BYTES,backups=2,secrets=[]}={}){
 const values=[...new Set(secrets)].filter(value=>typeof value==='string'&&value.length>=8).sort((a,b)=>b.length-a.length);
 const keep=Math.max(0,...values.map(value=>value.length-1)),redact=text=>values.reduce((result,value)=>result.split(value).join('[REDACTED]'),text);
 let fd=null,size=0;
 function open(){try{fs.mkdirSync(path.dirname(file),{recursive:true});fd=fs.openSync(file,'a');size=fs.fstatSync(fd).size;}catch{fd=null;}}
 function rotate(){
  if(fd!==null){try{fs.closeSync(fd);}catch{}fd=null;}
  try{if(backups<1)fs.rmSync(file,{force:true});for(let index=backups;index>=1;index--){const from=index===1?file:`${file}.${index-1}`;if(fs.existsSync(from))fs.renameSync(from,`${file}.${index}`);}}catch{}
  open();
 }
 function write(text){
  if(!text)return;const bytes=Buffer.from(redact(text),'utf8');
  if(fd!==null&&size>0&&size+bytes.length>maxBytes)rotate();
  if(fd===null)return;
  try{fs.writeSync(fd,bytes);size+=bytes.length;}catch{}
 }
 open();if(fd!==null&&size>=maxBytes)rotate();
 const {StringDecoder}=require('string_decoder');
 return {
  stream(){
   const decoder=new StringDecoder('utf8');let pending='';
   // A line too long to hold is cut short of its last characters, which may begin a secret still arriving (redacted first,
   // so a whole one already there cannot straddle the cut).
   const flush=all=>{
    let cut=all?pending.length:pending.lastIndexOf('\n')+1;
    if(!cut&&pending.length>=LINE_BUFFER){pending=redact(pending);cut=pending.length-keep;}
    if(cut>0){write(pending.slice(0,cut));pending=pending.slice(cut);}
   };
   return {write(chunk){pending+=typeof chunk==='string'?chunk:decoder.write(chunk);flush(false);},end(){pending+=decoder.end();flush(true);}};
  },
  close(){if(fd!==null){try{fs.closeSync(fd);}catch{}fd=null;}},
 };
}
function startBackend(directory,resources,secrets={},port,onListening=()=>{}){
 const executable=backendExecutable(resources);
 // This start's API token and confirmation key go to the backend only; it removes them from its own environment.
 const own=secrets.api_token&&secrets.confirm_key?{RIKO_API_TOKEN:secrets.api_token,RIKO_CONFIRM_KEY:secrets.confirm_key}:{};
 const env={...process.env,...own,...(process.platform==='darwin'?{PATH:finderPath(process.env.PATH)}:{}),RIKO_MANAGED:'1',RIKO_PORT:String(port),RIKO_DATA_DIR:directory,RIKO_CONFIG:path.join(directory,'character_config.yaml'),HF_HOME:path.join(directory,'models','huggingface'),TORCH_HOME:path.join(directory,'models','torch'),XDG_CACHE_HOME:path.join(directory,'models','cache'),RIKO_BUNDLE_ROOT:resources};
 const log=launchLog(path.join(directory,'logs','backend-launch.log'),{secrets:[secrets.api_token,secrets.confirm_key]});
 let child;
 try{child=spawn(executable,[],{cwd:directory,env,stdio:['pipe','pipe','pipe'],windowsHide:true});}catch(error){log.close();throw error;}
 child.stdin.on('error',()=>{});
 const output=log.stream(),errors=log.stream(),listening=watchListening(port,onListening);
 let open=2;const closed=stream=>()=>{stream.end();if(--open===0)log.close();};
 child.stdout.on('data',chunk=>{output.write(chunk);listening(chunk);});child.stdout.once('close',closed(output));
 child.stderr.on('data',chunk=>errors.write(chunk));child.stderr.once('close',closed(errors));
 return child;
}
// Only a YAML boolean true starts it (launch_config.cjs reads the file as the backend does): a quoted 'no' is not consent.
function startSovits(settings){
 if(settings?.auto_start!==true)return null;
 const executable=settings.executable,args=settings.arguments||[];
 if(!executable||!path.isAbsolute(executable)||!fs.statSync(executable).isFile())throw new Error('GPT-SoVITS executable is missing');
 if(!Array.isArray(args)||args.some(a=>typeof a!=='string'))throw new Error('GPT-SoVITS arguments must be a list of strings');
 return spawn(executable,args,{cwd:path.dirname(executable),stdio:'ignore',windowsHide:true,shell:false});
}
// Ask first; the backend gives itself 14 s. uvicorn ignores SIGTERM while already shutting down, so escalate to SIGKILL.
function stopBackend(child,grace=15000,force=3000){return new Promise(resolve=>{if(!child||child.exitCode!==null||child.signalCode!==null){resolve();return;}let timer=setTimeout(()=>{child.kill();timer=setTimeout(()=>{child.kill('SIGKILL');resolve();},force);},grace);child.once('exit',()=>{clearTimeout(timer);resolve();});child.stdin.end('shutdown\n');});}
module.exports={hardware,saveSetup,backendExecutable,backendAnswer,installRoot,askMicrophone,startBackend,launchLog,LAUNCH_LOG_BYTES,nativeLibrary:native.nativeLibrary,startSovits,stopBackend,watchListening,listeningLine,freePort,finderPath,LISTENING};
