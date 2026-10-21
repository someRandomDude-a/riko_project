import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {loadMain,loadPreload,settle} from './electron_fakes.mjs';
const require=createRequire(import.meta.url);
const {ARGUMENT}=require('../backend_origin.cjs');
const {desktopRoutes}=require('../ipc_routes.cjs');
// main.cjs and preload.cjs, evaluated against recording fakes of Electron (electron_fakes.mjs).
const OWNERS=Object.fromEntries(desktopRoutes({windows:{pointer:{}}}).map(route=>[route.channel,route.from]));
const BACKEND='http://127.0.0.1:9123';
async function started(options){const h=loadMain(options);try{await h.start();}catch(error){h.restore();throw error;}return h;}
const quit=h=>{const event={prevented:false,preventDefault(){this.prevented=true;}};h.module.app.emit('before-quit',event);return event.prevented;};
const index=(h,predicate)=>h.log.findIndex(predicate);
function preloadChannels(platform){
 const saved={document:globalThis.document,window:globalThis.window};
 globalThis.document={querySelector:()=>null};globalThis.window={matchMedia:()=>({matches:false}),screenX:0,screenY:0};
 try{
  const {exposed,calls}=loadPreload({platform});
  for(const api of Object.values(exposed))for(const value of Object.values(api))if(typeof value==='function'){const off=value(()=>{});if(typeof off==='function')off();}
  return [...new Set(calls.filter(call=>call.kind!=='on'&&call.kind!=='off').map(call=>call.channel))].sort();
 }finally{Object.assign(globalThis,saved);}
}

test('a Wayland launch relaunches under XWayland before anything else runs',async()=>{
 const h=loadMain({platform:'linux',env:{WAYLAND_DISPLAY:'wayland-0',APPIMAGE:'/apps/Riko.AppImage'},argv:['/apps/riko','.','--flag'],packaged:true});
 try{
  assert.deepEqual(h.calls(h.module.app,'relaunch').map(e=>e.args),[[{execPath:'/apps/Riko.AppImage',args:['.','--flag','--ozone-platform=x11']}]]);
  assert.deepEqual(h.calls(h.module.app,'exit').map(e=>e.args),[[0]]);
  assert.deepEqual(h.calls(h.module.app,'requestSingleInstanceLock'),[],'before the single-instance lock');
  assert.equal(h.ipc.handlers.size+h.ipc.listeners.size,0);assert.ok(!h.requested.has('./release.cjs'));
 }finally{h.restore();}
 for(const options of [{platform:'linux',env:{WAYLAND_DISPLAY:'wayland-0'},electron:{switches:['ozone-platform']}},{platform:'linux'},{platform:'darwin',env:{WAYLAND_DISPLAY:'wayland-0'}}]){
  const other=loadMain(options);
  try{assert.deepEqual(other.calls(other.module.app,'relaunch'),[],JSON.stringify(options));assert.ok(other.ipc.handlers.size>0);}finally{other.restore();}
 }
});

test('a second packaged launch only wakes the first: no backend, GPT-SoVITS, IPC or windows',async()=>{
 const h=loadMain({packaged:true,lock:false});
 try{
  await h.start();
  assert.equal(h.calls(h.module.app,'quit').length,1);
  assert.equal(h.ipc.handlers.size+h.ipc.listeners.size,0);assert.equal(h.windows.length,0);assert.ok(!h.requested.has('./release.cjs'));
  for(const event of ['second-instance','before-quit','will-quit'])assert.equal(h.module.app.listenerCount(event),0,event);
 }finally{h.restore();}
});

test('packaged: the backend gets a port of its own before any window, and the token flows only while it holds it',async()=>{
 const h=await started({packaged:true,release:{microphone:()=>new Promise(()=>{})}});  // an unanswered microphone prompt holds nothing up
 try{
  const step=method=>index(h,e=>e.method===method);
  assert.ok(step('askMicrophone')<step('freePort')&&step('freePort')<step('startBackend')&&step('startBackend')<step('onBeforeSendHeaders')&&step('onBeforeSendHeaders')<step('new'),'microphone, port, backend, token scope, then windows');
  const [directory,resources,secrets,port]=h.release.started[0];
  assert.deepEqual([directory,resources,port],[h.data,h.resources,9123]);assert.equal(process.env.RIKO_CONFIG,path.join(h.data,'character_config.yaml'));
  assert.deepEqual(h.windows.map(w=>w.page),['overlay','control','whiteboard','effects']);
  for(const window of h.windows)assert.deepEqual(window.options.webPreferences,{preload:require.resolve('../preload.cjs'),contextIsolation:true,nodeIntegration:false,additionalArguments:[ARGUMENT+BACKEND],...(['overlay','control'].includes(window.page)?{backgroundThrottling:false}:{})},window.page);
  assert.deepEqual(h.webRequest.filter.urls,[BACKEND+'/*','ws://127.0.0.1:9123/*']);
  for(const url of [BACKEND+'/api/status','ws://127.0.0.1:9123/ws/events'])assert.equal(h.authorization(url),undefined,'not before the backend names its port');
  const published=h.fetches.length;
  h.release.started[0][4]();  // release.watchListening heard RIKO_BACKEND_LISTENING port=9123
  for(const url of [BACKEND+'/api/status','ws://127.0.0.1:9123/ws/events'])assert.equal(h.authorization(url),'Bearer '+secrets.api_token,url);
  for(const url of ['http://127.0.0.1:9124/api/status','https://127.0.0.1:9123/','http://localhost:9123/'])assert.equal(h.authorization(url),undefined,url);
  await settle();assert.ok(h.fetches.slice(published).some(([url,init])=>url===BACKEND+'/api/resources/electron'&&init.headers.Authorization==='Bearer '+secrets.api_token),'Electron processes reach the backend once it listens');
  h.release.children[0].emit('exit',0);assert.equal(h.authorization(BACKEND+'/api/status'),undefined,'nor after it exits');
  assert.deepEqual(h.calls(h.module.dialog,'showErrorBox'),[]);
  h.release.children[0].emit('exit',1);assert.equal(h.calls(h.module.dialog,'showErrorBox')[0].args[0],'Backend stopped');
  // GPU rasterization stays on and nothing disables acceleration.
  assert.deepEqual(h.calls(h.module.app.commandLine,'appendSwitch').map(e=>e.args),[['enable-gpu-rasterization']]);assert.deepEqual(h.calls(h.module.app,'disableHardwareAcceleration'),[]);
 }finally{h.restore();}
});

test('every channel the preload uses is served, and refused to each window that does not own it',async()=>{
 const h=await started({packaged:true});
 try{
  const control=h.window('control');control.webContents.emit('did-finish-load');
  assert.equal(await h.invoke('neural-data-open',control),true);  // its page loads the same preload
  const channels=[...h.ipc.handlers.keys(),...h.ipc.listeners.keys()].sort();
  assert.deepEqual(channels,Object.keys(OWNERS).sort(),'main registers exactly the routes table');
  for(const platform of ['darwin','win32','linux'])assert.deepEqual(preloadChannels(platform),channels,'preload uses every channel and no other ('+platform+')');
  const senders={overlay:h.window('overlay'),control,whiteboard:h.window('whiteboard'),effects:h.window('effects'),'neural-data':h.window('neural-data'),stranger:9999};
  const args={'setup-finish':[{directory:'/d',settings:{}}],'settings-dirty':[true],'window-action':['maximize'],'chat-mode':['full'],'window-material':[true],'window-gesture':[{id:'g',phase:'begin',kind:'move'}],
   'compact-interactive':[true],'compact-scale':[{width:200,height:200}],'capture-whiteboard':[{x:0,y:0,width:10,height:10}],'confirm-security-change':['{}'],'pick-path':[{}],
   'popup-interactive':[true],'avatar-interactive':[true],'approval-interactive':[true],'overlay-drag':[{source:'avatar',active:true}],'sync-surfaces':[{has_displays:false,whiteboard_visible:true,effect:true}]};
  for(const channel of channels)for(const [name,sender] of Object.entries(senders)){
   if(OWNERS[channel].includes(name))continue;
   const calls=h.log.length,fetches=h.fetches.length,label=`${channel} from ${name}`;
   if(h.ipc.handlers.has(channel))await assert.rejects(h.invoke(channel,sender,...(args[channel]||[])),/required|limited to/,label);
   else h.send(channel,sender,...(args[channel]||[]));
   await settle(2);assert.equal(h.log.length,calls,label);assert.equal(h.fetches.length,fetches,label);
  }
  assert.equal(quit(h),true);assert.deepEqual(h.calls(h.module.dialog,'showMessageBoxSync'),[],'no foreign settings-dirty reached main');
  h.release.finishStop?.();
 }finally{h.restore();}
});

test('first run: only the setup page, with no backend, and only it can finish setup',async()=>{
 const h=await started({packaged:true,dataLocation:false});
 try{
  const setup=h.window('setup');
  assert.deepEqual(h.windows.map(w=>w.page),['setup']);assert.deepEqual(setup.options.webPreferences.additionalArguments,[],'no backend exists yet');
  assert.deepEqual([h.release.started.length,h.trays.length,h.webRequest.listener],[0,0,undefined]);assert.deepEqual(h.log.filter(e=>e.method==='freePort'),[]);
  await assert.rejects(h.invoke('setup-finish',9999,{directory:'/data/Riko',settings:{}}),/Setup window required/);
  assert.deepEqual(await h.invoke('setup-hardware',setup),{cpu:'Test CPU',gpus:['Test GPU']});
  assert.deepEqual(h.log.find(e=>e.method==='hardware').args,[h.resources],'detects only the bundles these resources hold');
  assert.equal(await h.invoke('setup-model',setup),'/picked');
  assert.deepEqual(h.calls(h.module.dialog,'showOpenDialog').at(-1).args,[setup,{properties:['openFile'],filters:[{name:'GGUF models',extensions:['gguf']}]}]);
  h.state.openDialog={canceled:true,filePaths:[]};assert.equal(await h.invoke('setup-directory',setup),null);
  assert.equal(await h.invoke('setup-finish',setup,{directory:'/data/Riko',settings:{name:'Mika'}}),true);
  assert.deepEqual(h.log.find(e=>e.method==='saveSetup').args,['/data/Riko',{name:'Mika'},h.resources,path.dirname('/install/Riko/riko')]);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(h.userData,'data-location.json'),'utf8')),{directory:'/data/Riko'});
  assert.ok(index(h,e=>e.method==='relaunch')<index(h,e=>e.method==='quit'));
  h.module.app.emit('second-instance');assert.equal(h.calls(setup,'focus').length,1);
 }finally{h.restore();}
});

test('quitting asks about unsaved Settings first, then holds the quit until the backend has stopped',async()=>{
 const h=await started({packaged:true});
 try{
  const control=h.window('control');control.webContents.emit('did-finish-load');
  h.send('settings-dirty',control,true);h.state.messageBox=1;  // Keep editing
  assert.equal(quit(h),true);assert.equal(h.module.app.isQuitting,undefined);assert.deepEqual(h.log.filter(e=>e.method==='stopBackend'),[]);
  assert.equal(h.calls(h.module.dialog,'showMessageBoxSync')[0].args[0],control);assert.equal(control.bounds.width,1050,'Settings is shown full size to decide');
  h.state.messageBox=0;  // Quit and discard
  assert.equal(quit(h),true);assert.equal(h.module.app.isQuitting,true);
  assert.deepEqual(h.log.filter(e=>e.method==='stopBackend').map(e=>e.args[0]),[h.release.children[0]]);assert.deepEqual(h.calls(h.module.app,'quit'),[]);
  assert.equal(quit(h),false,'a later quit passes');assert.equal(h.log.filter(e=>e.method==='stopBackend').length,1);
  h.release.finishStop();await settle();assert.equal(h.calls(h.module.app,'quit').length,1,'and main quits once the backend has gone');
  h.module.app.emit('will-quit');await settle();
  assert.equal(h.calls(h.module.globalShortcut,'unregisterAll').length,1);assert.deepEqual(JSON.parse(h.fetches.at(-1)[1].body),{processes:[]});
 }finally{h.restore();}
});

test('development: the origin comes from RIKO_BACKEND_URL and the token from the files the backend writes beside its config',async()=>{
 const h=loadMain({env:{RIKO_BACKEND_URL:'http://127.0.0.1:9124'}});
 try{
  const token=path.join(h.data,'persistent_memories','api_token');fs.mkdirSync(path.dirname(token));fs.writeFileSync(token,'dev-token\n');
  await h.start();
  assert.deepEqual(h.windows.map(w=>w.options.webPreferences.additionalArguments[0]),Array(4).fill(ARGUMENT+'http://127.0.0.1:9124'));
  assert.deepEqual(h.release.started,[],'development never spawns a backend');
  assert.equal(h.log.find(e=>e.method==='prepareAvatarAssets').args[0],path.join(path.dirname(require.resolve('../main.cjs')),'..','character_files'));
  assert.equal(h.authorization('http://127.0.0.1:9124/api/status'),'Bearer dev-token');
  fs.writeFileSync(token,'next-token\n');fs.utimesSync(token,new Date(),new Date(Date.now()+5000));
  assert.equal(h.authorization('http://127.0.0.1:9124/api/status'),'Bearer next-token','re-read when the backend restarts');
  await h.trays[0].menu.template.find(item=>item.label==='Start Discord client').click();
  assert.deepEqual(h.fetches.at(-1),['http://127.0.0.1:9124/api/discord/start',{method:'POST',headers:{Authorization:'Bearer next-token'}}]);
  assert.equal(quit(h),false,'nothing to stop');
 }finally{h.restore();}
 const bad=await started({env:{RIKO_BACKEND_URL:'http://localhost:9124'}});
 try{
  const [title,detail]=bad.calls(bad.module.dialog,'showErrorBox')[0].args;
  assert.equal(title,'Desktop startup failed');assert.match(detail,/no localhost/);assert.equal(bad.calls(bad.module.app,'quit').length,1);assert.equal(bad.windows.length,0);
 }finally{bad.restore();}
});

test('the tray, window icons and global shortcuts',async()=>{
 const h=await started({packaged:true});
 try{
  const icon=path.join(h.resources,'assets','tray.png');  // the root assets folder, shipped as extraResources
  assert.equal(h.trays[0].image.file,icon);assert.deepEqual(h.calls(h.trays[0],'setToolTip')[0].args,['Mika']);
  for(const window of h.windows)assert.deepEqual(h.calls(window,'setIcon').map(e=>e.args),[[icon]]);
  assert.deepEqual(h.trays[0].menu.template.map(item=>item.label||item.type),['Open chat','Start Discord client','Settings','Appearance','Whiteboard','separator','Quit']);
  assert.deepEqual(h.calls(h.module.globalShortcut,'register').map(e=>e.args[0]),['CommandOrControl+Shift+Space'],'only the summon shortcut by default');
  h.window('control').webContents.emit('did-finish-load');h.calls(h.module.globalShortcut,'register')[0].args[1]();
  assert.equal(h.window('control').bounds.width>88,true,'it opens the mini chat');
 }finally{h.restore();}
 const custom=await started({packaged:true,config:'desktop:\n  shortcuts:\n    popup: null\n    mic: Alt+M\n'});
 try{
  assert.deepEqual(custom.calls(custom.module.globalShortcut,'register').map(e=>e.args[0]),['Alt+M']);
  custom.release.started[0][4]();custom.calls(custom.module.globalShortcut,'register')[0].args[1]();await settle();
  assert.deepEqual(custom.fetches.filter(([url])=>url.endsWith('/toggle')).map(([url,init])=>[url,init.method]),[[BACKEND+'/api/mic/toggle','POST']]);
 }finally{custom.restore();}
 // Read as the backend's PyYAML reads it (launch_config.cjs): yes and off are booleans, not text.
 const yaml11=await started({packaged:true,config:'desktop:\n  debug: yes\n  shortcuts:\n    popup: off\n    mic: Alt+M\npresets:\n  default:\n    name: Yes\n'});
 try{
  assert.deepEqual(yaml11.calls(yaml11.module.globalShortcut,'register').map(e=>e.args[0]),['Alt+M']);
  assert.equal(yaml11.calls(yaml11.window('control'),'show').length>0,true,'debug: yes opens the control window');
  assert.deepEqual(yaml11.calls(yaml11.trays[0],'setToolTip')[0].args,['True']);
 }finally{yaml11.restore();}
});

test('the sandboxed preload requires only electron and hands the page only the origin main passed, and strict values',()=>{
 const preload=loadPreload({argv:['--other',ARGUMENT+BACKEND]});
 assert.deepEqual(preload.requested,['electron']);assert.equal(preload.exposed.rikoConfig.backend,BACKEND);
 for(const value of ['http://localhost:9123','http://127.0.0.1:9123/x','https://127.0.0.1:9123'])assert.equal(loadPreload({argv:[ARGUMENT+value]}).exposed.rikoConfig.backend,'',value);
 assert.equal(loadPreload().exposed.rikoConfig.backend,'');
 const {exposed,calls}=loadPreload();
 exposed.riko.settingsDirty('yes');exposed.riko.settingsDirty(true);exposed.compactBridge.interactive(0);
 assert.deepEqual(calls.map(call=>[call.channel,...call.args]),[['settings-dirty',false],['settings-dirty',true],['compact-interactive',false]]);
 assert.equal(loadPreload({platform:'win32'}).exposed.overlayInputBridge.native,true);assert.equal(exposed.overlayInputBridge.native,false);
});

test('on Linux the preload replays forwarded cursor positions as the pointermove hit tests expect',()=>{
 assert.ok(!loadPreload({platform:'darwin'}).listeners.has('forwarded-pointer'));
 const {listeners}=loadPreload({platform:'linux'}),dispatched=[],saved={document:globalThis.document,PointerEvent:globalThis.PointerEvent};
 const target={dispatchEvent:event=>dispatched.push(event)};
 globalThis.document={elementFromPoint:(x,y)=>x===5?null:target,documentElement:{dispatchEvent:event=>dispatched.push({root:true,...event})}};
 globalThis.PointerEvent=class{constructor(type,init){Object.assign(this,{type},init);}};
 try{
  listeners.get('forwarded-pointer')(null,{x:10,y:20});listeners.get('forwarded-pointer')(null,{x:5,y:6});listeners.get('forwarded-pointer')(null,{x:'1',y:2});
  assert.deepEqual(dispatched.map(event=>[event.root||false,event.type,event.clientX,event.clientY,event.pointerType,event.bubbles]),[[false,'pointermove',10,20,'mouse',true],[true,'pointermove',5,6,'mouse',true]]);
 }finally{Object.assign(globalThis,saved);}
});
