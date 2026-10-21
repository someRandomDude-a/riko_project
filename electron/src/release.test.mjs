import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {EventEmitter} from 'node:events';
const require=createRequire(import.meta.url);
const {hardware,saveSetup,backendExecutable,backendAnswer,installRoot,askMicrophone,startBackend,launchLog,LAUNCH_LOG_BYTES,startSovits,stopBackend,watchListening,listeningLine,freePort,finderPath,LISTENING}=require('../release.cjs');
const {MANIFEST}=require('../native_backends.cjs');
const YAML=require('yaml');
const form={backend:'cuda',repo:'owner/model',filename:'model.gguf',context:8192,output:1024,threads:4,memories:'I like tea.\nMy name is Alex.',sovitsAuto:false};
// A stand-in for riko-backend --setup-config (the real one: tests/test_settings_store.py and test_first_run.py). It logs each
// call beside itself, then answers as its first argument says.
const FAKE=String.raw`const fs=require('fs'),path=require('path');const [mode,...args]=process.argv.slice(2);const input=JSON.parse(fs.readFileSync(0,'utf8'));
fs.appendFileSync(path.join(__dirname,'calls.jsonl'),JSON.stringify({args,bundle:process.env.RIKO_BUNDLE_ROOT,input})+'\n');
if(mode==='ok'){console.log('a warning first');console.log(JSON.stringify({config:'presets:\n  default:\n    name: '+JSON.stringify(input.name||'Riko')+'\n'}));}
else if(mode==='refuse'){console.log(JSON.stringify({error:'Invalid context/output budget'}));process.exitCode=1;}
else if(mode==='empty')console.log(JSON.stringify({config:''}));
else if(mode==='crash'){console.log('not json');console.error('Traceback: boom');process.exitCode=3;}
else setTimeout(()=>{},60000);`;
function fakeBackend(){
  const folder=fs.mkdtempSync(path.join(os.tmpdir(),'riko-backend-')),script=path.join(folder,'riko-backend.cjs');fs.writeFileSync(script,FAKE);
  const calls=()=>fs.existsSync(path.join(folder,'calls.jsonl'))?fs.readFileSync(path.join(folder,'calls.jsonl'),'utf8').trim().split('\n').map(line=>JSON.parse(line)):[];
  return {command:mode=>[process.execPath,script,mode],calls,remove:()=>fs.rmSync(folder,{recursive:true,force:true})};
}
const temporary=prefix=>fs.mkdtempSync(path.join(os.tmpdir(),prefix));
test('setup asks the backend for the configuration and writes its answer in a data folder outside the application, never twice',async()=>{
  const backend=fakeBackend(),resources=temporary('riko-release-'),parent=temporary('riko-data-'),directory=path.join(parent,'Riko');
  try{
    assert.equal(await saveSetup(directory,{...form,name:'Mika'},resources,resources,'linux',backend.command('ok')),directory);
    const file=path.join(directory,'character_config.yaml');
    assert.equal(fs.readFileSync(file,'utf8'),'presets:\n  default:\n    name: "Mika"\n','the backend\'s text, as it wrote it');
    if(process.platform!=='win32')assert.equal(fs.statSync(file).mode&0o777,0o600);
    for(const folder of ['models','persistent_memories','logs'])assert.ok(fs.statSync(path.join(directory,folder)).isDirectory(),folder);
    assert.deepEqual(backend.calls(),[{args:['--setup-config',directory],bundle:resources,input:{...form,name:'Mika'}}],'the choices on stdin, the bundle to resolve bundled:<backend>');
    await assert.rejects(saveSetup(directory,form,resources,resources,'linux',backend.command('ok')),/EEXIST/);
    await assert.rejects(saveSetup(path.join(resources,'data'),form,resources,resources,'linux',backend.command('ok')),/outside the application folder/);
    await assert.rejects(saveSetup('relative/data',form,resources,resources,'linux',backend.command('ok')),/absolute data directory/);
    assert.equal(backend.calls().length,1,'refused before the backend is asked');assert.equal(fs.readFileSync(file,'utf8'),'presets:\n  default:\n    name: "Mika"\n');
  }finally{backend.remove();fs.rmSync(resources,{recursive:true,force:true});fs.rmSync(parent,{recursive:true,force:true});}
});
test('setup shows the backend\'s refusal, or why it gave none, and writes no configuration',async()=>{
  const backend=fakeBackend(),resources=temporary('riko-release-'),directory=temporary('riko-data-');
  try{
    await assert.rejects(saveSetup(directory,form,resources,resources,'linux',backend.command('refuse')),/^Error: Invalid context\/output budget$/);
    await assert.rejects(saveSetup(directory,form,resources,resources,'linux',backend.command('crash')),/gave no answer \(exit 3\): Traceback: boom/);
    await assert.rejects(saveSetup(directory,form,resources,resources,'linux',backend.command('empty')),/returned no configuration/);
    await assert.rejects(saveSetup(directory,form,resources,resources,'linux',[path.join(directory,'missing-backend')]),/could not start/);
    await assert.rejects(backendAnswer(backend.command('silent'),[],{},process.env,300),/did not answer in time/);
    assert.ok(!fs.existsSync(path.join(directory,'character_config.yaml')));
  }finally{backend.remove();fs.rmSync(resources,{recursive:true,force:true});fs.rmSync(directory,{recursive:true,force:true});}
});
test('the packaged backend is resources/backend/riko-backend, with .exe on Windows',()=>{
  assert.equal(backendExecutable('/app/resources','darwin'),path.join('/app/resources','backend','riko-backend'));
  assert.equal(backendExecutable('/app/resources','win32'),path.join('/app/resources','backend','riko-backend.exe'));
});
test('GPT-SoVITS starts only for auto_start: true, with an absolute executable',()=>{for(const auto_start of [false,undefined,'no','yes',1])assert.equal(startSovits({auto_start,executable:'relative.exe'}),null,String(auto_start));assert.equal(startSovits(undefined),null);assert.throws(()=>startSovits({auto_start:true,executable:'relative.exe'}),/missing/);});
const probe=outputs=>{const calls=[];return {calls,run:(file,args)=>{calls.push([file,...args]);return Promise.resolve(outputs[file]||'');}};};
test('hardware detection runs only the tools for this OS\'s bundles and preselects from them',async()=>{
  const mac=probe({'nvidia-smi':'NVIDIA','vulkaninfo':'MoltenVK'}),macHw=await hardware('/app/resources','darwin',mac.run);
  assert.deepEqual(mac.calls,[]);
  assert.deepEqual(macHw.backends,[{id:'metal',label:'Metal',description:'Apple silicon GPU'}]);
  assert.deepEqual(macHw.detections,[]);assert.equal(macHw.preferred,'metal');assert.equal(macHw.platform,'darwin');
  const linux=probe({'vulkaninfo':'GPU0: AMD Radeon RX 7900','rocminfo':'  Name:  gfx1100'}),linuxHw=await hardware('/app/resources','linux',linux.run);
  assert.deepEqual(linux.calls.map(call=>call[0]),['nvidia-smi','vulkaninfo'],'no hip bundle, so no rocminfo');
  assert.equal(linuxHw.preferred,'vulkan');assert.deepEqual(linuxHw.backends.map(backend=>backend.id),['cuda','vulkan']);assert.deepEqual(linuxHw.unavailable,[]);
  assert.deepEqual(linuxHw.detections.map(item=>[item.backend,item.required,item.output]),[['cuda',true,'No NVIDIA GPU detected via nvidia-smi'],['vulkan',false,'GPU0: AMD Radeon RX 7900']]);
  const windows=await hardware('/app/resources','win32',probe({'nvidia-smi':'NVIDIA GeForce RTX 4090, 24564, 551.86'}).run);
  assert.equal(windows.preferred,'cuda');assert.deepEqual(windows.backends.map(backend=>backend.id),['cuda','vulkan']);
});
test('a release with the opt-in hip bundle offers ROCm only for GPUs it was built for, with ROCm 7 installed, and Vulkan otherwise',async()=>{
  const resources=temporary('riko-resources-'),library=path.join(resources,'native','hip','libriko-native.so'),rocm=()=>true,noRocm=()=>false;
  const agents=(...names)=>['ROCk module is loaded','  Name:                    AMD Ryzen 9 7950X',...names.flatMap(name=>[`  Name:                    ${name}`,'  Marketing Name:          Radeon',`      Name:                    amdgcn-amd-amdhsa--${name}`])].join('\n');
  try{
    fs.mkdirSync(path.dirname(library),{recursive:true});fs.writeFileSync(library,'');
    const amd=probe({rocminfo:agents('gfx1100'),'vulkaninfo':'GPU0: AMD Radeon RX 7900'}),hw=await hardware(resources,'linux',amd.run,rocm);
    assert.deepEqual(amd.calls.map(call=>call[0]),['nvidia-smi','rocminfo','vulkaninfo']);
    assert.deepEqual(hw.backends.map(backend=>backend.id),['cuda','hip','vulkan']);assert.equal(hw.preferred,'hip');assert.deepEqual(hw.unavailable,[]);
    assert.equal(hw.detections.find(item=>item.backend==='hip').output,'Name:                    gfx1100','only GPU agent names');
    // AMD's packages keep rocminfo in /opt/rocm/bin, off a desktop session's PATH.
    const fallback=probe({'/opt/rocm/bin/rocminfo':agents('gfx1201')}),viaRocm=await hardware(resources,'linux',fallback.run,rocm);
    assert.deepEqual(fallback.calls.map(call=>call[0]),['nvidia-smi','rocminfo','vulkaninfo','/opt/rocm/bin/rocminfo']);assert.equal(viaRocm.preferred,'hip');
    // Never preselect a bundle that would abort the backend: a GPU it has no code for (RX 6700 XT gfx1031, a 780M gfx1103),
    // a supported card beside an unsupported iGPU (ggml-hip uses both), only amd-smi (no proof of /dev/kfd), no AMD GPU,
    // or no ROCm 7 runtime (the module would not load).
    for(const [outputs,exists] of [[{rocminfo:agents('gfx1031')},rocm],[{rocminfo:agents('gfx1103')},rocm],[{rocminfo:agents('gfx1100','gfx1036')},rocm],
      [{'amd-smi':'TARGET_GRAPHICS_VERSION: gfx1100'},rocm],[{rocminfo:'  Name:                    AMD Ryzen 9 7950X'},rocm],[{},rocm],[{rocminfo:agents('gfx1100')},noRocm]]){
      const none=probe(outputs),noHip=await hardware(resources,'linux',none.run,exists);
      assert.deepEqual(noHip.backends.map(backend=>backend.id),['cuda','vulkan'],JSON.stringify(outputs));assert.equal(noHip.preferred,'vulkan');
      assert.deepEqual(noHip.unavailable,[{id:'hip',label:'ROCm (HIP)',reason:MANIFEST.backends.find(backend=>backend.id==='hip').detect.missing}]);
      assert.ok(!none.calls.some(call=>call[0].includes('amd-smi')));
      if(exists===noRocm)assert.ok(!none.calls.some(call=>call[0].includes('rocminfo')),'no runtime: rocminfo never runs');
    }
    const nvidia=await hardware(resources,'linux',probe({'nvidia-smi':'NVIDIA GeForce RTX 4090, 24564, 550.54.15',rocminfo:agents('gfx1100')}).run,rocm);
    assert.equal(nvidia.preferred,'cuda');assert.deepEqual(nvidia.backends.map(backend=>backend.id),['cuda','hip','vulkan']);
    const windows=await hardware(resources,'win32',probe({rocminfo:agents('gfx1100')}).run,rocm);  // hip ships only for Linux
    assert.deepEqual(windows.backends.map(backend=>backend.id),['cuda','vulkan']);
  }finally{fs.rmSync(resources,{recursive:true,force:true});}
});
test('the setup page offers the backends hardware() reports instead of a fixed list',()=>{
  const page=fs.readFileSync(new URL('./first_setup.jsx',import.meta.url),'utf8');
  assert.doesNotMatch(page,/value="(cuda|vulkan|metal)"|backend:'(cuda|vulkan|metal)'|hw\.(nvidia|vulkan)\b/);
  assert.match(page,/\(hw\?\.backends\|\|\[\]\)\.map\(backend=><option key=\{backend\.id\} value=\{backend\.id\}>/);
  assert.match(page,/backend:h\.preferred/);
  assert.match(page,/\(hw\?\.unavailable\|\|\[\]\)\.map\(item=><p key=\{item\.id\}>\{item\.label\} is bundled but not offered\. \{item\.reason\}/);
});
test('setup refuses a data folder that an update or uninstall would delete',async()=>{
  const install=fs.mkdtempSync(path.join(os.tmpdir(),'riko-install-')),resources=path.join(install,'resources'),sibling=install+'-data',backend=fakeBackend();
  fs.mkdirSync(resources,{recursive:true});
  try{
    for(const directory of [install,path.join(install,'data'),path.join(install,'data','nested'),resources])await assert.rejects(saveSetup(directory,form,resources,install,'linux',backend.command('ok')),/outside the application folder/);
    assert.ok(!fs.existsSync(path.join(install,'data')));assert.deepEqual(backend.calls(),[]);
    assert.equal(await saveSetup(sibling,form,resources,install,'linux',backend.command('ok')),sibling);assert.ok(fs.existsSync(path.join(sibling,'character_config.yaml')));
  }finally{backend.remove();fs.rmSync(install,{recursive:true,force:true});fs.rmSync(sibling,{recursive:true,force:true});}
  assert.equal(installRoot('/Applications/Riko.app/Contents/MacOS/Riko','darwin'),'/Applications/Riko.app');
  assert.equal(installRoot('/opt/Riko/riko-companion','linux'),'/opt/Riko');
  // setup-finish passes installRoot(app.getPath('exe')): ipc_routes.test.mjs and main_wiring.test.mjs.
});
test('main learns the backend holds its port only from a complete marker line naming that port',()=>{
  assert.equal(listeningLine(9123),'RIKO_BACKEND_LISTENING port=9123');  // Code/run_server.py prints exactly this
  let calls=0;const watch=watchListening(9123,()=>calls++);
  watch('Riko AI server: http://127.0.0.1:9123\n'+LISTENING+'\n'+listeningLine(9124)+'\n'+LISTENING+' port=91234\nRIKO_BACKEND_');assert.equal(calls,0);
  watch('LISTENING port=9123_NOT\n');assert.equal(calls,0);watch(listeningLine(9123)+'\r\nINFO later\n');assert.equal(calls,1);watch(listeningLine(9123)+'\n');assert.equal(calls,1);
  const split=watchListening(9123,()=>calls++);for(const part of ['RIKO_BACK','END_LISTEN','ING por','t=9123','\n'])split(Buffer.from(part));assert.equal(calls,2);
});
test('the packaged backend gets a free loopback port of its own',async()=>{
  const port=await freePort();assert.ok(Number.isInteger(port)&&port>=1024&&port<=65535,String(port));
  const net=require('node:net');await new Promise((resolve,reject)=>{const server=net.createServer();server.once('error',reject);server.listen(port,'127.0.0.1',()=>server.close(resolve));});
  await assert.rejects(freePort({createServer:()=>Object.assign(new EventEmitter(),{unref(){},listen(){setImmediate(()=>this.emit('error',new Error('EACCES')));}})}),/EACCES/);
});
test('startBackend hands the backend its port and hears it only once the backend names that port',{skip:process.platform==='win32'&&'the fixture backend is a shell script'},async()=>{
  const resources=fs.mkdtempSync(path.join(os.tmpdir(),'riko-resources-')),directory=fs.mkdtempSync(path.join(os.tmpdir(),'riko-data-'));
  try{
    fs.mkdirSync(path.join(resources,'backend'));fs.mkdirSync(path.join(directory,'logs'));
    // Names whatever RIKO_PORT it was given, then waits for stdin to close, as the managed backend does.
    // Its stderr (logging's console) goes to the launch log too, with this start's secrets redacted.
    fs.writeFileSync(path.join(resources,'backend','riko-backend'),'#!/bin/sh\necho "token=$RIKO_API_TOKEN key=$RIKO_CONFIRM_KEY" >&2\necho "RIKO_BACKEND_LISTENING port=$RIKO_PORT managed=$RIKO_MANAGED"\necho "RIKO_BACKEND_LISTENING port=$RIKO_PORT"\ncat >/dev/null\necho "stopped" >&2\n',{mode:0o755});
    let heard;const listening=new Promise(resolve=>{heard=resolve;});
    const child=startBackend(directory,resources,{api_token:'t'.repeat(43),confirm_key:'k'.repeat(43)},9123,heard);
    const closed=new Promise(resolve=>child.once('close',resolve));
    await listening;await stopBackend(child,5000,1000);await closed;
    assert.equal(child.exitCode,0);assert.deepEqual(child.signalCode,null);
    const log=fs.readFileSync(path.join(directory,'logs','backend-launch.log'),'utf8');
    assert.match(log,/token=\[REDACTED\] key=\[REDACTED\]\n/);assert.match(log,/RIKO_BACKEND_LISTENING port=9123\n/);assert.match(log,/stopped\n/);
    assert.doesNotMatch(log,/tttttttt|kkkkkkkk/);
  }finally{fs.rmSync(resources,{recursive:true,force:true});fs.rmSync(directory,{recursive:true,force:true});}
});
test('the launch log is capped with two backups and never holds this start\'s secrets, even split across chunks',()=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'riko-launch-')),file=path.join(directory,'logs','backend-launch.log');
  try{
    assert.equal(LAUNCH_LOG_BYTES,5*1024*1024);
    fs.mkdirSync(path.dirname(file));fs.writeFileSync(file,'x'.repeat(300));  // an older build's uncapped log: rotated at open
    const secret='s3cr3t-token-value-0123456789',log=launchLog(file,{maxBytes:200,secrets:[secret,'short',undefined]}),out=log.stream(),err=log.stream();
    assert.equal(fs.readFileSync(file+'.1','utf8'),'x'.repeat(300));assert.equal(fs.readFileSync(file,'utf8'),'');
    out.write(Buffer.from('token='+secret.slice(0,10)));err.write('error: ');out.write(Buffer.from(secret.slice(10)+' ready\n'));  // split mid-secret
    const accent=Buffer.from('é\n');err.write(accent.subarray(0,1));err.write(accent.subarray(1));  // a character split across chunks
    assert.equal(fs.readFileSync(file,'utf8'),'token=[REDACTED] ready\nerror: é\n');
    for(let index=0;index<12;index++)out.write(`line ${index} `+'y'.repeat(40)+'\n');  // past 200 bytes, twice over
    out.write('tail without newline '+secret.slice(0,5));err.end();out.end();log.close();
    const files=fs.readdirSync(path.dirname(file)).sort();
    assert.deepEqual(files,['backend-launch.log','backend-launch.log.1','backend-launch.log.2']);  // two backups, no more
    for(const name of files)assert.ok(fs.statSync(path.join(path.dirname(file),name)).size<=200,name);
    const text=files.map(name=>fs.readFileSync(path.join(path.dirname(file),name),'utf8')).join('');
    assert.ok(!text.includes(secret)&&text.includes('line 11 ')&&fs.readFileSync(file,'utf8').endsWith('tail without newline '+secret.slice(0,5)));
    // A line longer than the buffer is written in pieces, a secret straddling a piece still redacted.
    const long=launchLog(path.join(directory,'long.log'),{secrets:[secret]}),stream=long.stream();
    stream.write('z'.repeat(64*1024-10)+secret+'z'.repeat(10));stream.write('\n');stream.end();long.close();
    const written=fs.readFileSync(path.join(directory,'long.log'),'utf8');
    assert.ok(!written.includes(secret.slice(0,8))&&written.includes('[REDACTED]')&&written.endsWith('z'.repeat(10)+'\n'));
  }finally{fs.rmSync(directory,{recursive:true,force:true});}
});
function fakeChild({exitOnShutdown=false,signalCode=null}={}){const child=Object.assign(new EventEmitter(),{exitCode:null,signalCode,signals:[],written:'',kill(signal='SIGTERM'){child.signals.push(signal);return true;}});child.stdin={end(text){child.written+=text;if(exitOnShutdown)setImmediate(()=>child.emit('exit',0,null));}};return child;}
test('stopBackend asks for shutdown, then escalates to SIGTERM and SIGKILL',async()=>{const hung=fakeChild();await stopBackend(hung,20,20);assert.equal(hung.written,'shutdown\n');assert.deepEqual(hung.signals,['SIGTERM','SIGKILL']);const quick=fakeChild({exitOnShutdown:true});await stopBackend(quick,20,20);assert.deepEqual(quick.signals,[]);const crashed=fakeChild({signalCode:'SIGABRT'}),started=Date.now();await stopBackend(crashed,1000,1000);assert.ok(Date.now()-started<500);assert.equal(crashed.written,'');});
test('a Finder-launched macOS backend still finds Homebrew tools for MCP servers and FFmpeg',()=>{
  assert.equal(finderPath('/usr/bin:/bin:/usr/sbin:/sbin'),'/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin');
  assert.equal(finderPath('/usr/local/bin:/opt/homebrew/bin:/usr/bin'),'/opt/homebrew/sbin:/usr/local/bin:/opt/homebrew/bin:/usr/bin');
  assert.equal(finderPath(undefined),'/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin');
  assert.match(fs.readFileSync(new URL('../release.cjs',import.meta.url),'utf8'),/process\.platform==='darwin'\?\{PATH:finderPath\(process\.env\.PATH\)\}/);
});
test('Linux launches keep the AppImage sandbox flag and run under XWayland',()=>{const builder=YAML.parse(fs.readFileSync(new URL('../electron-builder.yml',import.meta.url),'utf8'));assert.deepEqual(builder.appImage.executableArgs,['--no-sandbox','--ozone-platform=x11']);assert.ok(builder.linux.executableArgs.includes('--ozone-platform=x11'));});  // a Wayland launch relaunches itself: window_platform.test.mjs, main_wiring.test.mjs
test('installers carry every local file main-process code requires, including the native-backend manifest',()=>{
  const builder=YAML.parse(fs.readFileSync(new URL('../electron-builder.yml',import.meta.url),'utf8')),dir=new URL('..',import.meta.url);
  const shipped=file=>builder.files.some(pattern=>pattern===file||(pattern.startsWith('*.')&&!file.includes('/')&&file.endsWith(pattern.slice(1)))||(pattern.endsWith('/**')&&file.startsWith(pattern.slice(0,-2))));
  const local=fs.readdirSync(dir).filter(f=>f.endsWith('.cjs')).flatMap(f=>[...fs.readFileSync(new URL(f,dir),'utf8').matchAll(/require\(\s*['"](\.\/[^'"]+)['"]\s*\)/g)].map(m=>path.posix.normalize(m[1])));
  assert.ok(local.includes('native_backends.json')&&local.includes('native_backends.cjs'));
  for(const file of local)assert.ok(shipped(file),file+' is required by main-process code but not packaged');
});
test('installers ship only what main-process files require, and every npm version is exact',()=>{
  // electron-builder packs each production dependency into app.asar and refuses electron or itself there; the renderer is bundled into dist/.
  const pkg=JSON.parse(fs.readFileSync(new URL('../package.json',import.meta.url),'utf8')),dir=new URL('..',import.meta.url),{builtinModules}=require('node:module');
  const required=new Set(fs.readdirSync(dir).filter(f=>f.endsWith('.cjs')).flatMap(f=>[...fs.readFileSync(new URL(f,dir),'utf8').matchAll(/require\(\s*['"]([^'"]+)['"]\s*\)/g)].map(m=>m[1])).filter(n=>!n.startsWith('.')&&n!=='electron'&&!n.startsWith('node:')&&!builtinModules.includes(n)));
  assert.deepEqual(Object.keys(pkg.dependencies).sort(),[...required].sort());
  for(const name of ['electron','electron-builder','vite'])assert.ok(pkg.devDependencies[name],name);
  for(const [name,version] of Object.entries({...pkg.dependencies,...pkg.devDependencies}))assert.match(version,/^\d+\.\d+\.\d+$/,name);
  assert.equal(pkg.devDependencies['electron-builder'],'26.0.12');
});
test('macOS asks for the microphone before the packaged backend starts, without holding it up',async()=>{
  const preferences=(status,answer=()=>Promise.resolve(true))=>{const calls=[];return {calls,getMediaAccessStatus:media=>{calls.push(['status',media]);return status;},askForMediaAccess:media=>{calls.push(['ask',media]);return answer();}};};
  const fresh=preferences('not-determined');assert.equal(await askMicrophone(fresh,'darwin'),true);assert.deepEqual(fresh.calls,[['status','microphone'],['ask','microphone']]);
  const granted=preferences('granted');assert.equal(await askMicrophone(granted,'darwin'),null);assert.deepEqual(granted.calls,[['status','microphone']]);
  for(const status of ['denied','restricted']){const known=preferences(status);assert.equal(await askMicrophone(known,'darwin'),'blocked');assert.deepEqual(known.calls,[['status','microphone']]);}
  for(const platform of ['win32','linux']){const other=preferences('not-determined');assert.equal(await askMicrophone(other,platform),null);assert.deepEqual(other.calls,[]);}
  assert.equal(await askMicrophone(preferences('not-determined',()=>Promise.resolve(false)),'darwin'),false);
  for(const fails of [()=>{throw new Error('no usage description');},()=>Promise.reject(new Error('denied'))])assert.equal(await askMicrophone(preferences('not-determined',fails),'darwin'),false);
  // Main asks before it starts the backend, without waiting for the answer: main_wiring.test.mjs.
});
test('the macOS app is a hardened arm64 DMG whose processes may use the microphone and load the backend libraries',()=>{
  const builder=YAML.parse(fs.readFileSync(new URL('../electron-builder.yml',import.meta.url),'utf8')),mac=builder.mac;
  assert.deepEqual(mac.target,[{target:'dmg',arch:['arm64']}]);assert.equal(mac.hardenedRuntime,true);
  // One file for the app and everything inside it: the backend child opens the microphone and loads ad hoc-signed libraries.
  assert.equal(mac.entitlements,'build/entitlements.mac.plist');assert.equal(mac.entitlementsInherit,mac.entitlements);
  const entitlements=fs.readFileSync(new URL('../'+mac.entitlements,import.meta.url),'utf8');
  assert.deepEqual([...entitlements.matchAll(/<key>([^<]+)<\/key>\s*<true\/>/g)].map(match=>match[1]).sort(),['com.apple.security.cs.allow-jit','com.apple.security.cs.disable-library-validation','com.apple.security.device.audio-input']);
  assert.equal([...entitlements.matchAll(/<key>/g)].length,3);
  assert.match(mac.extendInfo.NSMicrophoneUsageDescription,/microphone/);
  assert.equal(mac.sign,'./build/mac_sign.cjs');assert.ok(fs.existsSync(new URL('.'+mac.sign.slice(1),new URL('../',import.meta.url))));
  assert.equal(mac.icon,'../release-stage/mac/icon.icns');
  assert.ok(MANIFEST.backends.find(backend=>backend.id==='metal').cmake_flags.includes('-DCMAKE_OSX_DEPLOYMENT_TARGET='+mac.minimumSystemVersion));
  assert.ok(!builder.files.some(pattern=>pattern.startsWith('build')),'build resources stay out of app.asar');
});
test('electron-builder copies the staged backend and native bundles with their library links kept as links',{skip:process.platform==='win32'&&'Windows bundles hold no links'},async()=>{
  // tools/release/build.py stages each versioned library once (libggml.so -> .so.0 -> .so.0.25.3) and PyInstaller links
  // libraries into _internal; extraResources goes through app-builder-lib's copyFiles, which must not turn them into copies.
  const builder=YAML.parse(fs.readFileSync(new URL('../electron-builder.yml',import.meta.url),'utf8'));
  assert.deepEqual(builder.extraResources.filter(item=>item.from.startsWith('../release-stage/')).map(item=>item.to),['backend','native','notices']);
  const {copyFiles,FileMatcher}=require('app-builder-lib/out/fileMatcher'),stage=temporary('riko-stage-'),resources=temporary('riko-resources-');
  try{
    const bundle=path.join(stage,'native','cuda');fs.mkdirSync(bundle,{recursive:true});
    fs.writeFileSync(path.join(bundle,'libggml.so.0.25.3'),'ggml');fs.symlinkSync('libggml.so.0.25.3',path.join(bundle,'libggml.so.0'));fs.symlinkSync('libggml.so.0',path.join(bundle,'libggml.so'));
    await copyFiles([new FileMatcher(path.join(stage,'native'),path.join(resources,'native'),value=>value)]);
    const copied=path.join(resources,'native','cuda');
    assert.equal(fs.readlinkSync(path.join(copied,'libggml.so')),'libggml.so.0');assert.equal(fs.readlinkSync(path.join(copied,'libggml.so.0')),'libggml.so.0.25.3');
    assert.ok(!fs.lstatSync(path.join(copied,'libggml.so.0.25.3')).isSymbolicLink());assert.equal(fs.readFileSync(path.join(copied,'libggml.so'),'utf8'),'ggml');
  }finally{fs.rmSync(stage,{recursive:true,force:true});fs.rmSync(resources,{recursive:true,force:true});}
});
test('packaged apps cannot be relaunched as plain Node to borrow their microphone permission',()=>{const builder=YAML.parse(fs.readFileSync(new URL('../electron-builder.yml',import.meta.url),'utf8'));assert.deepEqual(builder.electronFuses,{runAsNode:false,enableNodeOptionsEnvironmentVariable:false,enableNodeCliInspectArguments:false});});
