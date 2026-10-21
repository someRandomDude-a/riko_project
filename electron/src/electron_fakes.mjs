// Test fixture (no sibling test of its own): recording fakes of Electron's main-process API, and loaders that evaluate the real
// main.cjs and preload.cjs against them. Every fake call lands in `log` as {target, method, args}, in order.
import {EventEmitter} from 'node:events';
import Module,{createRequire} from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const require=createRequire(import.meta.url);
export const MAIN=require.resolve('../main.cjs'),PRELOAD=require.resolve('../preload.cjs');
export const settle=async(rounds=20)=>{for(let i=0;i<rounds;i++)await new Promise(resolve=>setImmediate(resolve));};
export const PRIMARY={id:1,label:'Built-in',bounds:{x:0,y:0,width:1920,height:1080},workArea:{x:0,y:0,width:1920,height:1040},scaleFactor:2};
// Manual timers for modules that take {set, clear}: run() fires what is due, in time order.
export function fakeTimers(){
 let now=0,id=0;const pending=new Map();
 return {set:(fn,ms)=>{pending.set(++id,{fn,at:now+ms});return id;},clear:handle=>{pending.delete(handle);},get pending(){return pending.size;},
  run(ms){const until=now+ms;for(;;){const due=[...pending].filter(([,t])=>t.at<=until).sort((a,b)=>a[1].at-b[1].at)[0];if(!due)break;pending.delete(due[0]);now=due[1].at;due[1].fn();}now=until;},
  now:()=>now};
}
export function fakeElectron({packaged=false,lock=true,userData,exe='/install/Riko/riko',switches=[],displays=[PRIMARY],messageBox=0,openDialog={canceled:false,filePaths:['/picked']}}={}){
 const log=[],windows=[];let nextId=1,resolveReady;
 const record=(target,method,args)=>{log.push({target,method,args});};
 const recorder=(target,methods)=>{for(const [method,fn] of Object.entries(methods))target[method]=(...args)=>{record(target,method,args);return fn?.(...args);};return target;};
 const state={cursor:{x:0,y:0},messageBox,openDialog,displays,shortcuts:true};
 class WebContents extends EventEmitter{
  constructor(owner){super();this.id=nextId++;this.owner=owner;this.sent=[];
   recorder(this,{send:(channel,value)=>{this.sent.push([channel,value]);},setWindowOpenHandler:fn=>{this.openHandler=fn;},
    capturePage:rect=>Promise.resolve({rect,getSize:()=>({width:rect.width,height:rect.height}),resize:size=>({getSize:()=>size,toPNG:()=>Buffer.from('png')}),toPNG:()=>Buffer.from('png')})});}
  sentOn(channel){return this.sent.filter(([name])=>name===channel).map(([,value])=>value);}
 }
 class BrowserWindow extends EventEmitter{
  constructor(options){
   super();this.options=options;this.bounds={x:options.x??0,y:options.y??0,width:options.width??800,height:options.height??600};
   this.visible=false;this.minimized=false;this.maximized=false;this.destroyed=false;this.hooks={};this.contentView={children:[]};
   this.webContents=new WebContents(this);windows.push(this);record(this,'new',[options]);
   recorder(this,{loadURL:url=>{this.url=url;return Promise.resolve();},setIgnoreMouseEvents:(ignore,opts)=>{this.ignoring=ignore;this.forward=!!opts?.forward;},
    setFocusable:v=>{this.focusable=v;},showInactive:()=>{this.visible=true;},show:()=>{this.visible=true;},hide:()=>{this.visible=false;},focus:null,
    restore:()=>{this.minimized=false;},minimize:()=>{this.minimized=true;},maximize:()=>{this.maximized=true;},unmaximize:()=>{this.maximized=false;},
    setBounds:b=>{this.bounds={...this.bounds,...b};},setMinimumSize:null,setResizable:null,setAspectRatio:null,setAlwaysOnTop:null,setBackgroundColor:null,
    setBackgroundMaterial:null,setMenuBarVisibility:null,setIcon:null,hookWindowMessage:(message,fn)=>{this.hooks[message]=fn;},close:()=>{const event={defaultPrevented:false,preventDefault(){this.defaultPrevented=true;}};this.emit('close',event);return event;}});
  }
  get page(){return this.url?.split('#/')[1]||'';}
  isVisible(){return this.visible;} isMinimized(){return this.minimized;} isMaximized(){return this.maximized;} isDestroyed(){return this.destroyed;}
  getBounds(){return {...this.bounds};} getNormalBounds(){return {...this.bounds};} getContentBounds(){return {...this.bounds};}
  destroy(){this.destroyed=true;}
 }
 const app=Object.assign(new EventEmitter(),{isPackaged:packaged,isQuitting:undefined,
  commandLine:recorder({},{appendSwitch:null,hasSwitch:name=>switches.includes(name)}),
  whenReady:()=>ready,isReady:()=>true,getAppMetrics:()=>[{pid:10,type:'Browser'},{pid:11,type:'Tab'}],getGPUInfo:()=>Promise.resolve({gpuDevice:[{deviceString:'Test GPU'}]}),
  getPath:name=>({userData,exe}[name])});
 const ready=new Promise(resolve=>{resolveReady=resolve;});
 recorder(app,{quit:null,exit:null,relaunch:null,requestSingleInstanceLock:()=>lock,disableHardwareAcceleration:null});
 const ipc={handlers:new Map(),listeners:new Map()};
 const ipcMain={handle:(channel,fn)=>{ipc.handlers.set(channel,fn);},on:(channel,fn)=>{ipc.listeners.set(channel,fn);}};
 const event=sender=>({sender:{id:typeof sender==='number'?sender:sender.webContents.id}});
 // As Electron does, a handler's synchronous throw rejects the invoke.
 const invoke=async(channel,sender,...args)=>ipc.handlers.get(channel)(event(sender),...args);
 const send=(channel,sender,...args)=>ipc.listeners.get(channel)(event(sender),...args);
 const dialog=recorder({},{showErrorBox:null,showOpenDialog:()=>Promise.resolve(state.openDialog),showMessageBox:()=>Promise.resolve({response:state.messageBox}),showMessageBoxSync:()=>state.messageBox});
 const screen=Object.assign(new EventEmitter(),{getPrimaryDisplay:()=>state.displays[0],getAllDisplays:()=>state.displays,getCursorScreenPoint:()=>({...state.cursor}),
  getDisplayMatching:b=>state.displays.find(d=>b.x>=d.bounds.x&&b.x<d.bounds.x+d.bounds.width)||state.displays[0]});
 const webRequest=recorder({},{onBeforeSendHeaders:(filter,listener)=>{webRequest.filter=filter;webRequest.listener=listener;}});
 const shell=recorder({},{openExternal:()=>Promise.resolve()});
 const trays=[];
 class Tray{constructor(image){this.image=image;trays.push(this);record(this,'new',[image]);recorder(this,{setToolTip:null,setContextMenu:menu=>{this.menu=menu;},on:(name,fn)=>{this[name]=fn;}});}}
 const Menu={buildFromTemplate:template=>({template})};
 const nativeImage={createFromPath:file=>({file,isEmpty:()=>false,resize:size=>({file,size})})};
 const globalShortcut=recorder({},{register:()=>state.shortcuts,unregisterAll:null});
 const systemPreferences={getMediaAccessStatus:()=>'granted',askForMediaAccess:()=>Promise.resolve(true)};
 const module={app,BrowserWindow,ipcMain,dialog,screen,session:{defaultSession:{webRequest}},shell,Tray,Menu,nativeImage,globalShortcut,systemPreferences};
 // The header main would add to a renderer request for url, or undefined.
 const authorization=url=>{let headers;webRequest.listener({url,requestHeaders:{}},result=>{headers=result.requestHeaders;});return headers.Authorization;};
 return {module,log,windows,trays,ipc,invoke,send,state,webRequest,authorization,ready:()=>resolveReady(),
  window:page=>windows.find(w=>w.page===page&&!w.destroyed),calls:(target,method)=>log.filter(entry=>entry.target===target&&(!method||entry.method===method))};
}
export function fakeChild(){
 const child=Object.assign(new EventEmitter(),{exitCode:null,signalCode:null,signals:[],written:''});
 child.kill=(signal='SIGTERM')=>{child.signals.push(signal);return true;};child.stdin={end:text=>{child.written+=text;}};
 return child;
}
// release.cjs as main uses it, recording into the same log.
export function fakeRelease(log,{port=9123,microphone=()=>Promise.resolve(null)}={}){
 const release={children:[],started:[]};const record=(method,args)=>log.push({target:release,method,args});
 Object.assign(release,{
  hardware:async(...args)=>{record('hardware',args);return {cpu:'Test CPU'};},
  saveSetup:(...args)=>{record('saveSetup',args);return args[0];},installRoot:exe=>path.dirname(exe),
  askMicrophone:preferences=>{record('askMicrophone',[preferences]);return microphone();},
  freePort:async()=>{record('freePort',[]);return port;},
  startBackend:(...args)=>{record('startBackend',args);const child=fakeChild();release.children.push(child);release.started.push(args);return child;},
  startSovits:settings=>{record('startSovits',[settings]);return null;},
  stopBackend:child=>{record('stopBackend',[child]);return new Promise(resolve=>{release.finishStop=resolve;});}});
 return release;
}
export function tempDir(prefix='riko-main-'){return fs.mkdtempSync(path.join(os.tmpdir(),prefix));}
// Evaluates main.cjs against the fakes (on darwin unless platform says otherwise): process.platform, the environment, process.resourcesPath and fetch are swapped for
// the run (restore() puts them back), and require('electron'), main's './release.cjs' and './avatar_assets.cjs' are faked.
export function loadMain({file=MAIN,platform='darwin',env={},packaged=false,lock=true,dataLocation,config='presets:\n  default:\n    name: Mika\n',electron:options={},release:releaseOptions,argv}={}){
 const home=tempDir(),userData=path.join(home,'userData'),resources=path.join(home,'resources'),data=path.join(home,'data');
 for(const dir of [userData,resources,data,path.join(resources,'assets')])fs.mkdirSync(dir,{recursive:true});
 if(config!==null)fs.writeFileSync(path.join(data,'character_config.yaml'),config);
 if(packaged&&dataLocation!==false)fs.writeFileSync(path.join(userData,'data-location.json'),JSON.stringify({directory:dataLocation||data}));
 const electron=fakeElectron({packaged,lock,userData,...options}),release=fakeRelease(electron.log,releaseOptions),fetches=[],requested=new Set();
 const saved={platform:Object.getOwnPropertyDescriptor(process,'platform'),env:{...process.env},resourcesPath:process.resourcesPath,fetch:globalThis.fetch,argv:process.argv};
 Object.defineProperty(process,'platform',{...saved.platform,value:platform});
 for(const key of Object.keys(process.env))if(/^RIKO_|^WAYLAND_DISPLAY$|^APPIMAGE$/.test(key))delete process.env[key];
 Object.assign(process.env,packaged?{}:{RIKO_CONFIG:path.join(data,'character_config.yaml')},env);
 if(argv)process.argv=argv;
 process.resourcesPath=resources;
 globalThis.fetch=(url,init)=>{fetches.push([url,init]);return Promise.resolve({ok:true,json:async()=>({})});};
 const restore=()=>{Object.defineProperty(process,'platform',saved.platform);for(const key of Object.keys(process.env))if(!(key in saved.env))delete process.env[key];Object.assign(process.env,saved.env);
  process.argv=saved.argv;if(saved.resourcesPath===undefined)delete process.resourcesPath;else process.resourcesPath=saved.resourcesPath;globalThis.fetch=saved.fetch;fs.rmSync(home,{recursive:true,force:true});};
 const load=Module._load;
 Module._load=function(request,parent,isMain){
  if(request==='electron')return electron.module;
  if(parent?.filename===file&&request.startsWith('./'))requested.add(request);
  if(parent?.filename===file&&request==='./release.cjs')return release;
  if(parent?.filename===file&&request==='./avatar_assets.cjs')return {prepareAvatarAssets:(...args)=>electron.log.push({target:'avatar_assets',method:'prepareAvatarAssets',args})};
  return load.apply(this,arguments);
 };
 try{delete require.cache[file];require(file);}catch(error){restore();throw error;}finally{Module._load=load;}
 return {...electron,release,fetches,requested,restore,home,userData,resources,data,start:async()=>{electron.ready();await settle();}};
}
// Evaluates preload.cjs against a fake contextBridge and ipcRenderer: `exposed` holds each bridge, `calls` each renderer call,
// `requested` each module the preload requires.
export function loadPreload({platform='darwin',argv=[]}={}){  // explicit, never the host's: tests pass 'linux' or 'win32' when they mean it
 const exposed={},calls=[],listeners=new Map(),requested=[];
 const ipcRenderer={invoke:(channel,...args)=>{calls.push({kind:'invoke',channel,args});return Promise.resolve();},send:(channel,...args)=>{calls.push({kind:'send',channel,args});},
  on:(channel,fn)=>{calls.push({kind:'on',channel});listeners.set(channel,fn);},removeListener:(channel)=>{calls.push({kind:'off',channel});}};
 const electron={contextBridge:{exposeInMainWorld:(name,api)=>{exposed[name]=api;}},ipcRenderer};
 const saved={platform:Object.getOwnPropertyDescriptor(process,'platform'),argv:process.argv},load=Module._load;
 Object.defineProperty(process,'platform',{...saved.platform,value:platform});process.argv=[...process.argv.slice(0,1),...argv];
 Module._load=function(request,parent){if(parent?.filename===PRELOAD)requested.push(request);return request==='electron'?electron:load.apply(this,arguments);};
 try{delete require.cache[PRELOAD];require(PRELOAD);}finally{Module._load=load;Object.defineProperty(process,'platform',saved.platform);process.argv=saved.argv;}
 return {exposed,calls,listeners,requested};
}
