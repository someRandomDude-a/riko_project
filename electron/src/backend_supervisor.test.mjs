import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fakeChild,tempDir} from './electron_fakes.mjs';
const require=createRequire(import.meta.url);
const {backendSupervisor,microphoneNotice}=require('../backend_supervisor.cjs');
const release=require('../release.cjs');
const {resolveBackend}=require('../backend_origin.cjs');
function packaged({quitting=false,stopTimes}={}){
 const errors=[],started=[],child=fakeChild();
 const fake={...release,freePort:async()=>9123,startBackend:(...args)=>{started.push(args);return child;}};
 const backend=backendSupervisor({release:fake,dialog:{showErrorBox:(...args)=>errors.push(args)},quitting:()=>quitting,stopTimes});
 return {backend,errors,started,child};
}
function inject(backend){
 let listener;backend.injectToken({onBeforeSendHeaders:(filter,fn)=>{listener=fn;listener.urls=filter.urls;}});
 return Object.assign(url=>{let headers;listener({url,requestHeaders:{Accept:'*/*'}},result=>{headers=result.requestHeaders;});return headers.Authorization;},{urls:listener.urls});
}
test('packaged: a fresh port and secrets per start, and the token only between the backend naming its port and exiting',async()=>{
 const {backend,errors,started,child}=packaged(),heard=[];
 assert.equal(await backend.spawn('/data','/resources',()=>heard.push('listening')),child);
 const [directory,resources,secrets,port,onListening]=started[0];
 assert.deepEqual([directory,resources,port],['/data','/resources',9123]);assert.equal(backend.origin.origin,'http://127.0.0.1:9123');
 assert.match(secrets.api_token,/^[A-Za-z0-9_-]{43}$/);assert.match(secrets.confirm_key,/^[A-Za-z0-9_-]{43}$/);assert.notEqual(secrets.api_token,secrets.confirm_key);
 const authorization=inject(backend);
 assert.deepEqual(authorization.urls,['http://127.0.0.1:9123/*','ws://127.0.0.1:9123/*']);
 assert.equal(authorization('http://127.0.0.1:9123/api/status'),undefined);assert.equal(backend.secret('confirm_key'),'');
 onListening();assert.deepEqual(heard,['listening']);
 assert.equal(authorization('http://127.0.0.1:9123/api/status'),'Bearer '+secrets.api_token);assert.equal(backend.secret('confirm_key'),secrets.confirm_key);
 for(const url of ['http://127.0.0.1:9124/','http://localhost:9123/','https://127.0.0.1:9123/'])assert.equal(authorization(url),undefined,url);
 child.emit('exit',0);assert.equal(authorization('http://127.0.0.1:9123/api/status'),undefined);assert.deepEqual(errors,[]);
 child.emit('exit',1);child.emit('error',new Error('ENOENT'));
 assert.deepEqual(errors.map(([title])=>title),['Backend stopped','Backend failed to launch']);assert.equal(errors[1][1],'ENOENT');
 const quiet=packaged({quitting:true});await quiet.backend.spawn('/data','/resources');quiet.child.emit('exit',1);assert.deepEqual(quiet.errors,[],'no alarm while quitting');
});
test('quitting stops the backend first: shutdown, then SIGTERM, then SIGKILL, and holds the quit until it has gone',async()=>{
 const {backend,child}=packaged({stopTimes:[20,20]});await backend.spawn('/data','/resources');
 let quit;const done=new Promise(resolve=>{quit=resolve;});
 assert.equal(backend.shutdown(quit),true);assert.equal(backend.shutdown(()=>assert.fail('stopped twice')),false,'a second quit passes');
 await done;assert.equal(child.written,'shutdown\n');assert.deepEqual(child.signals,['SIGTERM','SIGKILL']);
 const dev=backendSupervisor({release,dialog:{}});assert.equal(dev.shutdown(()=>assert.fail('nothing to stop')),false);
});
test('GPT-SoVITS starts only when asked, its failures are shown, and it is killed on quit',()=>{
 const errors=[],sovits=fakeChild(),dialog={showErrorBox:(...args)=>errors.push(args)};
 const start=startSovits=>backendSupervisor({release:{...release,startSovits},dialog});
 const backend=start(()=>sovits);backend.startSovits({auto_start:true});sovits.emit('error',new Error('EACCES'));
 assert.equal(backend.shutdown(()=>{}),false);assert.deepEqual(sovits.signals,['SIGTERM']);
 start(()=>{throw new Error('GPT-SoVITS executable is missing');}).startSovits({auto_start:true});
 start(()=>null).startSovits({auto_start:false});
 assert.deepEqual(errors,[['GPT-SoVITS could not start','EACCES'],['GPT-SoVITS could not start','GPT-SoVITS executable is missing']]);
});
test('development: the secrets are the files the backend writes, re-read whenever they change',()=>{
 const dir=tempDir('riko-secrets-'),fetched=[];
 try{
  const backend=backendSupervisor({release,dialog:{},fetch:(...args)=>{fetched.push(args);return Promise.resolve('ok');}});
  assert.equal(backend.token(),'','no folder yet');
  backend.watchSecrets(dir);assert.equal(backend.token(),'','no file yet');
  fs.writeFileSync(path.join(dir,'api_token'),'first\n');assert.equal(backend.token(),'first');
  fs.writeFileSync(path.join(dir,'api_token'),'second\n');fs.utimesSync(path.join(dir,'api_token'),new Date(),new Date(Date.now()+5000));assert.equal(backend.token(),'second');
  fs.writeFileSync(path.join(dir,'confirm_key'),'key');assert.equal(backend.secret('confirm_key'),'key');
  fs.rmSync(path.join(dir,'api_token'));assert.equal(backend.token(),'');
  fs.writeFileSync(path.join(dir,'api_token'),'third');
  return (async()=>{
   await assert.rejects(backend.request('/api/status'),/No backend yet/);
   backend.connect(resolveBackend({RIKO_BACKEND_URL:'http://127.0.0.1:9124'}));
   assert.equal(await backend.request('/api/mic/toggle',{method:'POST',headers:{'Content-Type':'application/json'}}),'ok');
   assert.deepEqual(fetched,[['http://127.0.0.1:9124/api/mic/toggle',{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer third'}}]]);
  })().finally(()=>fs.rmSync(dir,{recursive:true,force:true}));
 }catch(error){fs.rmSync(dir,{recursive:true,force:true});throw error;}
});
test('macOS: a blocked microphone is explained once, until access is granted',async()=>{
 const dir=tempDir('riko-mic-'),marker=path.join(dir,'microphone-blocked-notice'),shown=[],opened=[];
 const dialog={showMessageBox:options=>{shown.push(options);return Promise.resolve({response:0});}},shell={openExternal:url=>{opened.push(url);return Promise.resolve();}};
 try{
  microphoneNotice(true,{marker,dialog,shell});microphoneNotice(true,{marker,dialog,shell});await new Promise(resolve=>setImmediate(resolve));
  assert.equal(shown.length,1);assert.equal(shown[0].message,'Riko cannot hear you');assert.ok(fs.existsSync(marker));
  assert.deepEqual(opened,['x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone']);
  microphoneNotice(false,{marker,dialog,shell});assert.ok(!fs.existsSync(marker));
  microphoneNotice(true,{marker,dialog,shell});assert.equal(shown.length,2,'blocked again after being granted');
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
