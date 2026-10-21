const crypto=require('crypto');
const fs=require('fs');
const path=require('path');
const {handle,on}=require('./ipc_router.cjs');
// Every channel main serves (preload.cjs is the only caller), with the windows allowed to use it; ipc_router.cjs enforces
// those. src/main_wiring.test.mjs checks that the preload's channels and this table match and that a foreign sender is
// refused on every one. Names: overlay, control (chat and Settings), whiteboard, effects, setup, neural-data.
const CHROME=['control','whiteboard'],TITLES={settings:'Change security-sensitive settings?',tool_approvals:'Let these tools run without asking first?',discord_access:'Change who can use Riko from Discord?'};
// Show control, invisible and direction-changing characters as escapes, so the text shown is what gets signed.
const visible=text=>String(text).replace(/[\p{C}\p{Zl}\p{Zp}]/gu,c=>`\\u{${c.codePointAt(0).toString(16)}}`);
// Security-sensitive changes (native library, programs Riko starts, unattended tools, Discord access) need the user's approval
// here. The signature uses the confirmation key, which never leaves main and the backend, over the exact change the backend
// asked to confirm, so a compromised page cannot approve it silently. key() is read once the user has answered.
async function confirmChange(dialog,parent,challenge,key){
 let change;
 try{change=JSON.parse(challenge);}catch{throw new Error('Invalid confirmation request');}
 const lines=Object.entries(change.changes||{}).map(([name,value])=>`${visible(name)}: ${visible(JSON.stringify(value))}`);
 const {response}=await dialog.showMessageBox(parent,{type:'warning',buttons:['Cancel','Allow'],defaultId:0,cancelId:0,noLink:true,
  message:TITLES[change.action]||'Confirm this change?',detail:lines.join('\n')+'\n\nOnly allow this if you made this change yourself.'});
 const secret=key();
 if(response!==1||!secret)return null;
 return crypto.createHmac('sha256',secret).update(challenge).digest('hex');
}
// Settings' file and folder pickers: an extension filter only from plain alphanumeric extensions.
function pathDialogOptions(options={}){
 return {title:options.directory?'Choose a folder':'Choose a file',
  ...(Array.isArray(options.extensions)&&options.extensions.length&&options.extensions.every(value=>typeof value==='string'&&/^[a-z0-9]+$/i.test(value))?{filters:[{name:'Supported files',extensions:options.extensions}]}:{}),
  properties:[options.directory?'openDirectory':'openFile'],...(typeof options.defaultPath==='string'&&options.defaultPath?{defaultPath:options.defaultPath}:{})};
}
// A region of the whiteboard's own page, at most 1280 px on its longer side, as PNG bytes.
async function captureWhiteboard(whiteboard,rect){
 const bounds=whiteboard.getContentBounds();
 if(!rect||!['x','y','width','height'].every(key=>Number.isFinite(rect[key]))||rect.x<0||rect.y<0||rect.width<1||rect.height<1||rect.x+rect.width>bounds.width+1||rect.y+rect.height>bounds.height+1)throw new Error('Invalid whiteboard capture bounds');
 let image=await whiteboard.webContents.capturePage(Object.fromEntries(Object.entries(rect).map(([key,value])=>[key,Math.round(value)])),{stayHidden:true,stayAwake:true});
 const size=image.getSize(),scale=Math.min(1,1280/size.width,1280/size.height);
 if(scale<1)image=image.resize({width:Math.max(1,Math.round(size.width*scale)),height:Math.max(1,Math.round(size.height*scale))});
 return image.toPNG();
}
const picked=result=>result.canceled?null:result.filePaths[0];
function desktopRoutes({app,dialog,screen,release,windows,backend,processes,displays,settings,platform,resources,files=fs}){
 const pointer=windows.pointer;
 return [
  // First run: the setup page alone exists, and only it may finish setup (release.saveSetup refuses an application folder).
  handle('setup-hardware',['setup'],'Setup window required',async()=>{const hw=await release.hardware(resources);try{hw.gpus=(await app.getGPUInfo('basic')).gpuDevice?.map(device=>device.deviceString||`GPU vendor ${device.vendorId}, device ${device.deviceId}`)||[];}catch{hw.gpus=[];}return hw;}),
  handle('setup-directory',['setup'],'Setup window required',async({window})=>picked(await dialog.showOpenDialog(window,{properties:['openDirectory','createDirectory']}))),
  handle('setup-model',['setup'],'Setup window required',async({window})=>picked(await dialog.showOpenDialog(window,{properties:['openFile'],filters:[{name:'GGUF models',extensions:['gguf']}]}))),
  handle('setup-sovits',['setup'],'Setup window required',async({window})=>picked(await dialog.showOpenDialog(window,{properties:['openFile']}))),
  handle('setup-finish',['setup'],'Setup window required',async(_sender,values)=>{const directory=await release.saveSetup(values.directory,values.settings,resources,release.installRoot(app.getPath('exe')));files.writeFileSync(path.join(app.getPath('userData'),'data-location.json'),JSON.stringify({directory}));app.relaunch();app.quit();return true;}),
  handle('neural-data-open',['control'],'Settings renderer required',()=>{windows.openNeuralData();return true;}),
  on('sync-processes',['control','overlay','whiteboard','effects'],()=>processes.publish(true)),
  on('settings-dirty',['control'],(_sender,dirty)=>{settings.dirty=dirty===true;}),
  on('show-control',['overlay','control','whiteboard'],()=>windows.showControls()),
  on('show-whiteboard',['control'],()=>windows.showWhiteboard()),
  handle('window-state',CHROME,'Control or whiteboard renderer required',({name})=>windows.chromeState(name)),
  handle('window-action',CHROME,'Control or whiteboard renderer required',({name},action,anchor,instant)=>windows.chromeAction(name,action,anchor,instant===true)),
  handle('chat-mode',CHROME,'Control or whiteboard renderer required',({name},mode,anchor,instant)=>windows.chromeMode(name,mode,anchor,instant===true)),
  handle('window-material',CHROME,'Control or whiteboard renderer required',({name},enabled)=>windows.material(name,enabled)),
  on('window-gesture',CHROME,({name},value)=>windows.windowGesture(name,value)),
  on('compact-interactive',['control'],(_sender,enabled)=>windows.compactInteractive(enabled)),
  handle('compact-scale',['control'],'Dock or compact chat required',(_sender,size)=>windows.compactScale(size)),
  handle('control-visible',['overlay'],'Overlay renderer required',()=>windows.controlVisible()),
  handle('capture-whiteboard',['whiteboard'],'Whiteboard renderer required',({window},rect)=>captureWhiteboard(window,rect)),
  handle('displays',['control'],'Settings renderer required',()=>displays.list()),
  handle('confirm-security-change',['control'],'Settings window required',({window},challenge)=>confirmChange(dialog,window,challenge,()=>backend.secret('confirm_key'))),
  handle('pick-path',['control'],'Path selection is limited to controls',async({window},options={})=>picked(await dialog.showOpenDialog(window,pathDialogOptions(options)))),
  // The overlay takes the pointer only while the renderer reports a hit (popup, avatar, approval card) or a drag it captured.
  handle('avatar-cursor',['overlay'],'Interactive overlay renderer required',({window})=>{if(!pointer.interactive)throw new Error('Interactive overlay renderer required');const p=screen.getCursorScreenPoint(),b=window.getContentBounds();return {x:p.x-b.x,y:p.y-b.y,buttons:platform.nativeButtons?pointer.buttons:null};}),
  ...[['popup-interactive','popup'],['avatar-interactive','avatar'],['approval-interactive','approval']].map(([channel,key])=>on(channel,['overlay'],(_sender,enabled)=>pointer.hover(key,!!enabled))),
  on('overlay-drag',['overlay'],(_sender,value)=>{if(['avatar','popup'].includes(value?.source)&&typeof value.active==='boolean')pointer.drag(value.source,value.active);}),
  on('sync-surfaces',['overlay'],(_sender,state)=>{if(!state||typeof state!=='object')return;if(!state.has_displays)displays.reset();displays.publish();windows.syncSurfaces(state);}),
 ];
}
module.exports={desktopRoutes,confirmChange,pathDialogOptions,captureWhiteboard};
