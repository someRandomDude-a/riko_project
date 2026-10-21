const crypto=require('crypto');
const fs=require('fs');
const path=require('path');
const {originForPort,ownsRequest}=require('./backend_origin.cjs');
// The Python backend as main sees it: its origin, the secrets that authenticate main and every renderer to it, the packaged
// process main spawns and stops, and GPT-SoVITS when the YAML asks main to start it.
// The backend requires the install's API token (Code/process/app_core/desktop/api_guard.py). Main adds it to every request to
// the backend at the network layer (injectToken), so renderer JavaScript never sees it. Both secrets are fresh on every
// backend start. Packaged builds generate them here, hand them to the backend they spawn (no files) and send the token only
// once it reports that it holds the port; in development the backend writes them beside its config after it holds the port,
// and main re-reads them whenever the files change (watchSecrets).
function backendSupervisor({release,dialog,quitting=()=>false,fetch=globalThis.fetch,files=fs,randomBytes=crypto.randomBytes,stopTimes=[]}){
 let origin=null,child=null,sovits=null,own=null,directory='',stopping=null;const cache={};
 function secret(name){
  if(own)return own.listening?own[name]:'';
  if(!directory)return '';
  const file=path.join(directory,name),cached=cache[name]||{mtime:0,value:''};
  try{const stat=files.statSync(file);if(stat.mtimeMs!==cached.mtime)cache[name]={mtime:stat.mtimeMs,value:files.readFileSync(file,'ascii').trim()};}
  catch{cache[name]={mtime:0,value:''};}
  return cache[name].value;
 }
 const token=()=>secret('api_token');
 return {
  get origin(){return origin;},
  // Development: the origin resolveBackend gave (RIKO_BACKEND_URL or the default); that backend is started by hand.
  connect(value){origin=value;},
  // Packaged: a free port of its own, so a development backend (or anything else) on the default port is never taken for it.
  // Nothing carries the token until the backend says it holds that port (release.watchListening), and nothing does after it exits.
  async spawn(root,resources,onListening=()=>{}){
   own={api_token:randomBytes(32).toString('base64url'),confirm_key:randomBytes(32).toString('base64url'),listening:false};
   origin=originForPort(await release.freePort());
   child=release.startBackend(root,resources,own,origin.port,()=>{own.listening=true;onListening();});
   child.on('error',error=>dialog.showErrorBox('Backend failed to launch',error.message));
   child.on('exit',code=>{own.listening=false;if(!quitting()&&code)dialog.showErrorBox('Backend stopped','Review logs/backend-launch.log in your data folder. Open Settings to correct the model or backend configuration.');});
   return child;
  },
  watchSecrets(folder){directory=folder;},
  secret,token,
  // This origin only: 'localhost' may resolve to ::1, where another account could listen.
  injectToken(webRequest){
   webRequest.onBeforeSendHeaders({urls:[...origin.patterns]},(details,callback)=>{
    const value=token();
    if(value&&ownsRequest(origin,details.url))details.requestHeaders.Authorization='Bearer '+value;
    callback({requestHeaders:details.requestHeaders});
   });
  },
  // Main's own requests (tray, shortcuts, whiteboard geometry, telemetry): the only fetch in main-process code.
  request:(route,init={})=>origin?fetch(origin.origin+route,{...init,headers:{...init.headers,Authorization:'Bearer '+token()}}):Promise.reject(new Error('No backend yet')),
  startSovits(settings){
   try{sovits=release.startSovits(settings);sovits?.on('error',error=>dialog.showErrorBox('GPT-SoVITS could not start',error.message));}
   catch(error){dialog.showErrorBox('GPT-SoVITS could not start',error.message);}
  },
  // before-quit: the backend stops first (release.stopBackend asks, then escalates to SIGTERM and SIGKILL) and then() quits again
  // once it has gone. True while main must hold this quit; the second before-quit passes.
  shutdown(then){
   let hold=false;
   if(child&&!stopping){hold=true;stopping=release.stopBackend(child,...stopTimes).finally(then);}
   if(sovits)sovits.kill();
   return hold;
  }};
}
// macOS hands a process without microphone access silent buffers, not an error, so say so once (until access is granted).
function microphoneNotice(blocked,{marker,dialog,shell,files=fs}){
 if(!blocked){files.rmSync(marker,{force:true});return;}
 if(files.existsSync(marker))return;
 try{files.writeFileSync(marker,'');}catch{}
 dialog.showMessageBox({type:'info',buttons:['Open Settings','Not now'],defaultId:0,cancelId:1,message:'Riko cannot hear you',
  detail:'Microphone access for Riko is off, so voice input records silence. Turn it on in System Settings → Privacy & Security → Microphone.'})
  .then(({response})=>{if(response===0)shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone');}).catch(()=>{});
}
module.exports={backendSupervisor,microphoneNotice};
