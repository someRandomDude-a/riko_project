const { app, BrowserWindow, globalShortcut, ipcMain, dialog, screen, shell, Tray, Menu, nativeImage } = require('electron');
// Hardware acceleration is Electron's default; keep WebGL and GPU rasterization enabled.
app.commandLine.appendSwitch('enable-gpu-rasterization');
const path = require('path');
const fs = require('fs');
const YAML = require('yaml');
let root = app.isPackaged?process.resourcesPath:path.resolve(__dirname, '..');
let backendProcess,sovitsProcess,setupWindow,shutdownRequested=false;
if(app.isPackaged&&!app.requestSingleInstanceLock())app.quit();
app.on('second-instance',()=>{if(setupWindow&&!setupWindow.isDestroyed())setupWindow.focus();else if(control&&!control.isDestroyed())showControls();});
const release=require('./release.cjs');
function setupSender(event){if(!setupWindow||setupWindow.isDestroyed()||event.sender.id!==setupWindow.webContents.id)throw new Error('Setup window required');}
ipcMain.handle('setup-hardware',async event=>{setupSender(event);const hw=await release.hardware();try{hw.gpus=(await app.getGPUInfo('basic')).gpuDevice?.map(device=>device.deviceString||`GPU vendor ${device.vendorId}, device ${device.deviceId}`)||[];hw.vulkan=hw.gpus.join('\n')+'\n'+hw.vulkan;}catch{}return hw;});
ipcMain.handle('setup-directory',async event=>{setupSender(event);const result=await dialog.showOpenDialog(setupWindow,{properties:['openDirectory','createDirectory']});return result.canceled?null:result.filePaths[0];});
ipcMain.handle('setup-model',async event=>{setupSender(event);const result=await dialog.showOpenDialog(setupWindow,{properties:['openFile'],filters:[{name:'GGUF models',extensions:['gguf']}]});return result.canceled?null:result.filePaths[0];});
ipcMain.handle('setup-sovits',async event=>{setupSender(event);const result=await dialog.showOpenDialog(setupWindow,{properties:['openFile']});return result.canceled?null:result.filePaths[0];});
ipcMain.handle('setup-finish',(event,values)=>{setupSender(event);const directory=release.saveSetup(values.directory,values.settings,process.resourcesPath);fs.writeFileSync(path.join(app.getPath('userData'),'data-location.json'),JSON.stringify({directory}));app.relaunch();app.quit();return true;});
const brandIcon=path.join(root,'assets','tray.png');
let debug = false;
let overlay, control, whiteboard, effects;
let neuralDataWindow;
ipcMain.handle('neural-data-open',event=>{
 if(!control||control.isDestroyed()||event.sender.id!==control.webContents.id)throw new Error('Settings renderer required');
 if(neuralDataWindow&&!neuralDataWindow.isDestroyed()){neuralDataWindow.show();neuralDataWindow.focus();return true;}
 neuralDataWindow=new BrowserWindow({width:1000,height:800,title:'Expression training data',webPreferences:{preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false}});
 protectNavigation(neuralDataWindow);neuralDataWindow.loadURL(page('neural-data'));return true;
});
const overlayPointer=require('./overlay_input.cjs').overlayInput(enabled=>{if(overlay&&!overlay.isDestroyed())overlay.setIgnoreMouseEvents(!enabled,{forward:true});});
let tray;
let controlMode='full',fullControlBounds,compactControlBounds;
let boardMode='full',boardBounds,controlTween,boardTween,collapsedControlBounds;
const {chatBounds,clampBounds}=require('./window_layout.cjs');
const {gestureBounds}=require('./window_gesture.cjs');
const {applyWindowMaterial}=require('./window_material.cjs');
const {syncWindowContent}=require('./window_content.cjs');
let fullControlBlur=true;
let windowGesture=null;
function endWindowGesture(){if(windowGesture)clearTimeout(windowGesture.timer);windowGesture=null;}
const {windowAction} = require('./window_actions.cjs');
let boardScreen = 0, geometryTimer;
let displaySignature = '', displayPublishing = false;
const API = 'http://127.0.0.1:8765';
const patchBoard = body => fetch(API + '/api/surfaces/whiteboard', {method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}).catch(() => {});
const dev = process.env.RIKO_DEV === '1';
function page(name) { return dev ? `http://localhost:5173/#/${name}` : `file://${path.join(__dirname, 'dist', 'index.html')}#/${name}`; }
function displayList() {
  return screen.getAllDisplays().map((display, index) => ({index, id: display.id,
    label: display.label || `Screen ${index + 1}`, primary: display.id === screen.getPrimaryDisplay().id,
    bounds: display.bounds, scaleFactor: display.scaleFactor}));
}
async function publishDisplays() {
  const displays = displayList(), signature = JSON.stringify(displays);
  if (displayPublishing || signature === displaySignature) return;
  displayPublishing = true;
  try {
    const response = await fetch(API + '/api/displays', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: signature});
    if (response.ok) displaySignature = signature;
  } catch { /* Retry on the next backend snapshot/reconnection. */ }
  finally {displayPublishing = false;}
}
function protectNavigation(window) {
  window.webContents.on('will-navigate', event => event.preventDefault());
  window.webContents.setWindowOpenHandler(({url}) => {
    // Model links belong in the user's browser, never in a privileged renderer.
    if (/^https?:\/\//i.test(url)) shell.openExternal(url).catch(() => {});
    return {action: 'deny'};
  });
}
function prepareAvatarAssets() {
  if(app.isPackaged)return; // Installed application resources are read-only.
  const sourceDir = path.join(__dirname, '..', 'character_files');
  const targetDirs = [path.join(__dirname, 'public', 'models'), path.join(__dirname, 'dist', 'models')];
  targetDirs.forEach(dir => fs.mkdirSync(dir, {recursive: true}));
  for (const name of ['Mita.vrm', 'tiny_gremlin.vrm']) {
    const source = path.join(sourceDir, name);
    for (const targetDir of targetDirs) {
      const target = path.join(targetDir, name);
      if (fs.existsSync(source) && (!fs.existsSync(target) || fs.statSync(source).mtimeMs > fs.statSync(target).mtimeMs)) fs.copyFileSync(source, target);
    }
  }
}
function createWindows(characterName='') {
  overlay = new BrowserWindow({...screen.getPrimaryDisplay().bounds, show:false, transparent: true, backgroundColor:'#00000000', frame: false, resizable: false, alwaysOnTop: true, skipTaskbar: true, focusable: false, webPreferences: {preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false,backgroundThrottling:false}});
  overlay.once('ready-to-show',()=>{overlay.setFocusable(false);overlay.showInactive();});
  overlay.setIgnoreMouseEvents(true, {forward: true}); overlay.loadURL(page('overlay'));
  if(process.platform==='win32'){
   overlay.hookWindowMessage(0x0201,()=>overlayPointer.press());
   overlay.hookWindowMessage(0x0202,()=>{overlayPointer.release();const p=screen.getCursorScreenPoint(),b=overlay.getContentBounds();overlay.webContents.send('overlay-pointer',{x:p.x-b.x,y:p.y-b.y,buttons:0,phase:'up'});});
  }
  control = new BrowserWindow({width: 1050, height: 800, minWidth:600, minHeight:450, show: false, frame:false, thickFrame:false, hasShadow:false, transparent:true, backgroundColor:'#00000000', skipTaskbar:true, alwaysOnTop:true, title: characterName || 'Chat', webPreferences: {preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false,backgroundThrottling:false}});
  for(const name of ['show','hide','minimize','restore'])control.on(name,()=>{if(overlay&&!overlay.isDestroyed())overlay.webContents.send('control-visibility',controlMode!=='collapsed'&&control.isVisible()&&!control.isMinimized());});
  control.loadURL(page('control'));
  const publishChatSurface=()=>{if(overlay&&!overlay.isDestroyed())overlay.webContents.send('chat-surface',{mode:controlMode,bounds:control.getBounds(),screen:overlay.getBounds(),visible:control.isVisible()&&!control.isMinimized()});};
  for(const event of ['move','resize','show','hide'])control.on(event,publishChatSurface);
  overlay.webContents.on('did-finish-load',publishChatSurface);
  if(process.platform==='win32')control.hookWindowMessage(0x0202,()=>{if(windowGesture?.window===control)endWindowGesture();});
  control.webContents.once('did-finish-load',()=>{if(!debug){const a=screen.getPrimaryDisplay().workArea;setControlMode('collapsed',{x:a.x+a.width-80,y:a.y+a.height-70},true);}});
  control.on('close', event => {if (!app.isQuitting) {event.preventDefault();setControlMode(controlMode==='full'?'compact':'collapsed');}});
  control.on('hide',()=>{if(controlMode==='compact')compactControlBounds=control.getBounds();});
  control.on('minimize',()=>{if(app.isQuitting)return;control.setFocusable(false);control.restore();setControlMode('collapsed');});
  whiteboard = new BrowserWindow({width: 900, height: 700, minWidth: 320, minHeight: 240, show: false, frame:false, transparent:true, backgroundColor:'#00000000', skipTaskbar:true, alwaysOnTop:true, title: characterName?characterName+' — Whiteboard':'Whiteboard', webPreferences: {preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false}});
  whiteboard.loadURL(page('whiteboard'));
  whiteboard.on('close', event => {if (!app.isQuitting) {event.preventDefault();setBoardMode('collapsed');}});
  const saveGeometry = () => {
    clearTimeout(geometryTimer);
    geometryTimer = setTimeout(() => {
      if (whiteboard.isDestroyed()) return;
      const bounds = boardMode==='collapsed'?(boardBounds||whiteboard.getBounds()):whiteboard.getBounds(), displays = screen.getAllDisplays();
      const display = screen.getDisplayMatching(bounds); boardScreen = displays.findIndex(item => item.id === display.id);
      patchBoard({geometry: {...bounds, x: bounds.x - display.bounds.x, y: bounds.y - display.bounds.y, screen: boardScreen}});
    }, 200);
  };
  whiteboard.on('move', saveGeometry); whiteboard.on('resize', saveGeometry);
  effects = new BrowserWindow({...screen.getPrimaryDisplay().bounds, show: false, transparent: true, backgroundColor: '#00000000', frame: false, resizable: false, alwaysOnTop: true, skipTaskbar: true, focusable: false, webPreferences: {preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false}});
  effects.setIgnoreMouseEvents(true); effects.loadURL(page('effects'));
  [overlay, control, whiteboard, effects].forEach(window=>{protectNavigation(window);window.setIcon(brandIcon);});
   [control,whiteboard].forEach(window=>{window.setMenuBarVisibility(false);window.on('resize',()=>syncWindowContent(window));window.webContents.on('did-finish-load',()=>syncWindowContent(window));for(const event of ['maximize','unmaximize'])window.on(event,()=>{syncWindowContent(window);window.webContents.send('window-state',{maximized:window.isMaximized()});});});
}
function tweenBounds(window,target,kind,instant=false){
 clearTimeout(kind==='chat'?controlTween:boardTween);
 // Resize once. Repeated native resizing reallocates the transparent swapchain and
 // reflows every message/WebGL surface; only compositor properties animate now.
 window.webContents.send('window-state',{transitioning:!instant});
  window.setBounds(target);
  syncWindowContent(window);
 if(!instant){const timer=setTimeout(()=>{if(!window.isDestroyed())window.webContents.send('window-state',{transitioning:false});},180);if(kind==='chat')controlTween=timer;else boardTween=timer;}
}
function setControlMode(mode,anchor,instant=false){
 endWindowGesture();
 if(!['full','compact','collapsed'].includes(mode))throw new Error('Unsupported chat mode');
 if(mode!==controlMode){
  const current=control.getBounds(),area=screen.getDisplayMatching(current).workArea;
  if(controlMode==='full')fullControlBounds=control.getNormalBounds();if(controlMode==='compact')compactControlBounds=current;if(controlMode==='collapsed')collapsedControlBounds=current;
  if(control.isMaximized())control.unmaximize();
  const safeAnchor=anchor&&Number.isFinite(anchor.x)&&Number.isFinite(anchor.y)?anchor:undefined;
  const target=chatBounds(mode,current,area,safeAnchor,mode==='full'?fullControlBounds:mode==='collapsed'?collapsedControlBounds:compactControlBounds);
  controlMode=mode;control.setMinimumSize(88,88);control.setResizable(mode==='full');control.setAspectRatio(0);control.setFocusable(mode!=='collapsed');control.setAlwaysOnTop(true);control.setIgnoreMouseEvents(mode!=='full',{forward:true});
   applyWindowMaterial(control,mode,fullControlBlur);
  control.webContents.send('window-state',{maximized:false,mode,dockWidth:mode==='collapsed'?target.width:collapsedControlBounds?.width||88});tweenBounds(control,target,'chat',instant);
 }
 control.webContents.send('window-state',{maximized:control.isMaximized(),mode:controlMode});
 if(overlay&&!overlay.isDestroyed())overlay.webContents.send('control-visibility',mode!=='collapsed');
  if(mode==='collapsed')control.showInactive();else{if(control.isMinimized())control.restore();control.show();control.focus();}
  syncWindowContent(control);
}
function setBoardMode(mode,instant=false){
 if(!['full','collapsed'].includes(mode))throw new Error('Unsupported whiteboard mode');
 if(mode!==boardMode){const current=whiteboard.getBounds(),area=screen.getDisplayMatching(current).workArea;if(mode==='collapsed')boardBounds=current;
 const width=mode==='collapsed'?240:boardBounds?.width||900,height=mode==='collapsed'?76:boardBounds?.height||700;
 const target=clampBounds({x:current.x+(current.width-width)/2,y:current.y+current.height-height,width,height},area);
 boardMode=mode;whiteboard.setMinimumSize(200,60);whiteboard.webContents.send('window-state',{maximized:false,mode});tweenBounds(whiteboard,target,'board',instant);}
 whiteboard.setAlwaysOnTop(true);whiteboard.show();whiteboard.focus();
}
function showControls(view,mode=controlMode,anchor){if(view&&view!=='chat')mode='full';else if(mode==='collapsed')mode='compact';setControlMode(mode,anchor);if(control.isMinimized())control.restore();control.show();control.focus();if(view)control.webContents.send('navigate',view);}
function createTray(characterName){
  const icon=nativeImage.createFromPath(brandIcon);
  if(icon.isEmpty())throw new Error('Tray icon is missing or invalid: assets/tray.png');
  tray=new Tray(icon.resize({width:24,height:24}));tray.setToolTip(characterName||'Companion');
  tray.setContextMenu(Menu.buildFromTemplate([{label:'Open chat',click:()=>showControls('chat')},{label:'Start Discord client',click:async()=>{try{const response=await fetch(API+'/api/discord/start',{method:'POST'});if(!response.ok){let message='Discord could not be started';try{const body=await response.json();if(typeof body.detail==='string')message=body.detail;}catch{}dialog.showErrorBox('Discord',message);}}catch{dialog.showErrorBox('Discord','The Python backend is not available. Start it before launching Discord.');}}},{label:'Settings',click:()=>showControls('settings')},{label:'Appearance',click:()=>showControls('appearance')},{label:'Whiteboard',click:()=>{patchBoard({visible:true});setBoardMode('full');}},{type:'separator'},{label:'Quit',click:()=>app.quit()}]));
  tray.on('double-click',()=>showControls('chat'));
}
app.whenReady().then(async () => {
  try {
    if(app.isPackaged){
      const locator=path.join(app.getPath('userData'),'data-location.json');
      if(!fs.existsSync(locator)){
        setupWindow=new BrowserWindow({width:1000,height:850,webPreferences:{preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false}});
        protectNavigation(setupWindow);await setupWindow.loadURL(page('setup'));return;
      }
      root=JSON.parse(fs.readFileSync(locator,'utf8')).directory;
      process.env.RIKO_CONFIG=path.join(root,'character_config.yaml');
      backendProcess=release.startBackend(root,process.resourcesPath);
      backendProcess.on('error',error=>dialog.showErrorBox('Backend failed to launch',error.message));
      backendProcess.on('exit',code=>{if(!app.isQuitting&&code)dialog.showErrorBox('Backend stopped','Review logs/backend-launch.log in your data folder. Open Settings to correct the model or backend configuration.');});
    }
    const configPath = path.resolve(root, process.env.RIKO_CONFIG || 'character_config.yaml');
    const config = YAML.parse(fs.readFileSync(configPath, 'utf8')) || {};
    if(app.isPackaged){
      try{sovitsProcess=release.startSovits(config.sovits_ping_config);sovitsProcess?.on('error',error=>dialog.showErrorBox('GPT-SoVITS could not start',error.message));}
      catch(error){dialog.showErrorBox('GPT-SoVITS could not start',error.message);}
    }
    debug = config.desktop?.debug === true;
    prepareAvatarAssets(); createWindows(config.presets?.default?.name || '');
    createTray(config.presets?.default?.name || '');
    for (const name of ['display-added', 'display-removed', 'display-metrics-changed']) {
      screen.on(name, () => {displaySignature = ''; publishDisplays();});
    }
    if (debug) control.show();
    const shortcuts=config.desktop?.shortcuts||{};
    const actions={popup:()=>showControls(),quit:()=>app.quit(),whiteboard:()=>patchBoard({visible:!whiteboard.isVisible()}),
      settings:()=>showControls('settings'),
      mic:()=>fetch(API+'/api/mic/toggle',{method:'POST'}).catch(()=>{}),audio:()=>fetch(API+'/api/audio/toggle',{method:'POST'}).catch(()=>{}),sleep:()=>fetch(API+'/api/sleep/toggle',{method:'POST'}).catch(()=>{})};
    const defaults={popup:'CommandOrControl+Shift+Space',quit:'CommandOrControl+Shift+Q',whiteboard:'CommandOrControl+Shift+W',settings:'CommandOrControl+Shift+,'};
    for(const [name,action] of Object.entries(actions)){
      const accelerator=shortcuts[name]??defaults[name];
      if(accelerator){try{if(!globalShortcut.register(accelerator,action))console.warn('Shortcut unavailable:',name,accelerator);}catch(error){console.warn('Invalid shortcut:',name,error.message);}}
    }
  } catch (error) {
    dialog.showErrorBox('Desktop startup failed', error.stack || error.message);
    app.quit();
  }
});
let processSignature='',processUpdates=Promise.resolve();
function publishProcesses(force=false,empty=false){
  if(!app.isReady())return;
  const body=JSON.stringify({processes:empty?[]:app.getAppMetrics().map(item=>({pid:item.pid,kind:['Browser','GPU','Renderer','Utility','Zygote','Sandbox helper'].includes(item.type)?item.type:'Utility'}))});
  if(!force&&body===processSignature)return;processSignature=body;
  processUpdates=processUpdates.then(()=>fetch(API+'/api/resources/electron',{method:'POST',headers:{'Content-Type':'application/json'},body})).catch(()=>{processSignature='';});
}
app.whenReady().then(()=>publishProcesses());
app.on('web-contents-created',(_event,contents)=>{contents.on('did-finish-load',()=>publishProcesses());contents.on('destroyed',()=>publishProcesses());contents.on('render-process-gone',()=>publishProcesses());});
app.on('gpu-info-update',()=>publishProcesses());
app.on('child-process-gone',()=>publishProcesses());
ipcMain.on('sync-processes',event=>{if([control,overlay,whiteboard,effects].some(window=>window&&!window.isDestroyed()&&window.webContents.id===event.sender.id))publishProcesses(true);});
app.on('will-quit', () => {globalShortcut.unregisterAll();publishProcesses(true,true);});
app.on('before-quit', event => {app.isQuitting = true;if(backendProcess&&!shutdownRequested){event.preventDefault();shutdownRequested=true;release.stopBackend(backendProcess).finally(()=>app.quit());}if(sovitsProcess)sovitsProcess.kill();endWindowGesture(); clearTimeout(geometryTimer);clearTimeout(controlTween);clearTimeout(boardTween);});
ipcMain.on('show-control', event => {if([overlay,control,whiteboard].some(w=>w&&!w.isDestroyed()&&w.webContents.id===event.sender.id))showControls();});
function chromeWindow(event){const window=[control,whiteboard].find(w=>w&&!w.isDestroyed()&&w.webContents.id===event.sender.id);if(!window)throw new Error('Control or whiteboard renderer required');return window;}
ipcMain.handle('window-state',event=>{const w=chromeWindow(event);return {maximized:w.isMaximized(),mode:w===control?controlMode:boardMode,dockWidth:collapsedControlBounds?.width||88};});
ipcMain.handle('window-action',(event,action,anchor,instant)=>{const w=chromeWindow(event);if(['close','minimize'].includes(action)){if(w===control)setControlMode(controlMode==='full'?'compact':'collapsed',anchor,instant===true);else setBoardMode('collapsed',instant===true);return {mode:w===control?controlMode:boardMode};}if(w===control&&action==='maximize'&&controlMode!=='full')setControlMode('full',anchor,instant===true);return windowAction(w,action);});
ipcMain.handle('chat-mode',(event,mode,anchor,instant)=>{const w=chromeWindow(event);if(w===control)setControlMode(mode,anchor,instant===true);else setBoardMode(mode,instant===true);return {mode:w===control?controlMode:boardMode};});
ipcMain.on('compact-interactive',(event,enabled)=>{if(controlMode!=='full'&&event.sender.id===control.webContents.id&&!windowGesture)control.setIgnoreMouseEvents(!enabled,{forward:true});});
ipcMain.on('window-gesture',(event,value)=>{
 const w=[control,whiteboard].find(w=>w&&!w.isDestroyed()&&w.webContents.id===event.sender.id);
 if(!w||!value||typeof value.id!=='string'||value.id.length>64)return;
 if(value.phase==='end'){if(windowGesture?.id===value.id&&windowGesture.window===w)endWindowGesture();return;}
 if(value.phase!=='begin'||!['move','resize'].includes(value.kind)||(value.kind==='resize'&&(w!==control||controlMode==='full')))return;
 endWindowGesture();clearTimeout(w===control?controlTween:boardTween);w.webContents.send('window-state',{transitioning:false});
 const g={id:value.id,window:w,kind:value.kind,mode:controlMode,bounds:w.getBounds(),cursor:screen.getCursorScreenPoint(),began:Date.now()};
 windowGesture=g;w.setIgnoreMouseEvents(false);
 function frame(){
  if(windowGesture!==g)return;
  if(w.isDestroyed()||Date.now()-g.began>60000){endWindowGesture();return;}
  const cursor=screen.getCursorScreenPoint(),area=screen.getDisplayMatching(g.kind==='move'?{...g.bounds,x:g.bounds.x+cursor.x-g.cursor.x,y:g.bounds.y+cursor.y-g.cursor.y}:g.bounds).workArea;
  const next=gestureBounds(g,cursor,area),current=w.getBounds();
  if(!Object.keys(next).every(k=>next[k]===current[k])){
  w.setBounds(next);
  if(w===control){if(controlMode==='compact')compactControlBounds=next;else if(controlMode==='collapsed'){collapsedControlBounds=next;if(g.kind==='resize')w.webContents.send('window-state',{mode:controlMode,dockWidth:next.width});}}
  }
  g.timer=setTimeout(frame,16);
 }
 frame();
});
ipcMain.handle('compact-scale',(event,size)=>{if(chromeWindow(event)!==control||controlMode==='full'||!size||!Number.isFinite(size.width)||!Number.isFinite(size.height))throw new Error('Dock or compact chat required');clearTimeout(controlTween);const b=control.getBounds(),a=screen.getDisplayMatching(b).workArea;const w=Math.round(Math.max(controlMode==='collapsed'?88:a.width/6,Math.min(controlMode==='collapsed'?260:Math.min(a.width,a.height),size.width))),h=controlMode==='collapsed'?w:Math.round(Math.max(w,Math.min(a.height,size.height)));control.setBounds(clampBounds({x:b.x,y:b.y+b.height-h,width:w,height:h},a));if(controlMode==='compact')compactControlBounds=control.getBounds();else{collapsedControlBounds=control.getBounds();control.webContents.send('window-state',{mode:controlMode,dockWidth:w});}});
ipcMain.handle('window-material',(event,enabled)=>{const w=chromeWindow(event);if(typeof enabled!=='boolean')throw new Error('Invalid material preference');if(w===control&&controlMode==='full')fullControlBlur=enabled;const result=applyWindowMaterial(w,w===control?controlMode:'transparent',w===control&&fullControlBlur);syncWindowContent(w);return result;});
ipcMain.handle('control-visible',event=>{
  if(!overlay||event.sender.id!==overlay.webContents.id)throw new Error('Overlay renderer required');
  return !!control&&!control.isDestroyed()&&controlMode!=='collapsed'&&control.isVisible()&&!control.isMinimized();
});
ipcMain.handle('capture-whiteboard', async (event, rect) => {
  if(!whiteboard||whiteboard.isDestroyed()||event.sender.id!==whiteboard.webContents.id)throw new Error('Whiteboard renderer required');
  const bounds=whiteboard.getContentBounds();
  if(!rect||!['x','y','width','height'].every(key=>Number.isFinite(rect[key]))||rect.x<0||rect.y<0||rect.width<1||rect.height<1||rect.x+rect.width>bounds.width+1||rect.y+rect.height>bounds.height+1)throw new Error('Invalid whiteboard capture bounds');
  let image=await whiteboard.webContents.capturePage(Object.fromEntries(Object.entries(rect).map(([key,value])=>[key,Math.round(value)])),{stayHidden:true,stayAwake:true});
  const size=image.getSize(),scale=Math.min(1,1280/size.width,1280/size.height);
  if(scale<1)image=image.resize({width:Math.max(1,Math.round(size.width*scale)),height:Math.max(1,Math.round(size.height*scale))});
  return image.toPNG();
});
ipcMain.handle('displays', () => displayList());
ipcMain.handle('pick-path', async (event, options={}) => {
  if(!control||event.sender.id!==control.webContents.id)throw new Error('Path selection is limited to controls');
  const result=await dialog.showOpenDialog(control,{title:options.directory?'Choose a folder':'Choose a file',
    ...(Array.isArray(options.extensions)&&options.extensions.length&&options.extensions.every(value=>typeof value==='string'&&/^[a-z0-9]+$/i.test(value))?{filters:[{name:'Supported files',extensions:options.extensions}]}:{}),
    properties:[options.directory?'openDirectory':'openFile'],...(typeof options.defaultPath==='string'&&options.defaultPath?{defaultPath:options.defaultPath}:{})});
  return result.canceled?null:result.filePaths[0];
});
ipcMain.handle('avatar-cursor',event=>{if(!overlay||overlay.isDestroyed()||event.sender.id!==overlay.webContents.id||!overlayPointer.interactive)throw new Error('Interactive overlay renderer required');const p=screen.getCursorScreenPoint(),b=overlay.getContentBounds();return {x:p.x-b.x,y:p.y-b.y,buttons:process.platform==='win32'?overlayPointer.buttons:null};});
for(const [channel,key] of [['popup-interactive','popup'],['avatar-interactive','avatar'],['approval-interactive','approval']])ipcMain.on(channel,(event,enabled)=>{if(overlay&&!overlay.isDestroyed()&&event.sender.id===overlay.webContents.id)overlayPointer.hover(key,!!enabled);});
ipcMain.on('overlay-drag',(event,value)=>{if(overlay&&!overlay.isDestroyed()&&event.sender.id===overlay.webContents.id&&['avatar','popup'].includes(value?.source)&&typeof value.active==='boolean')overlayPointer.drag(value.source,value.active);});
ipcMain.on('show-whiteboard', event => {if (event.sender.id === control.webContents.id){patchBoard({visible: true});setBoardMode('full');}});
ipcMain.on('sync-surfaces', (event, state) => {
  if (event.sender.id !== overlay.webContents.id || !state || typeof state !== 'object') return;
  if (!state.has_displays) displaySignature = '';
  publishDisplays();
  const geometry = state.whiteboard_geometry;
  if(Number.isInteger(state.avatar_screen)){
    const displays=screen.getAllDisplays(),display=displays[state.avatar_screen] || screen.getPrimaryDisplay();
    const bounds=overlay.getBounds();if(Object.keys(display.bounds).some(key=>bounds[key]!==display.bounds[key])){overlay.setBounds(display.bounds);effects.setBounds(display.bounds);}
  }
  if (geometry && ['x', 'y', 'width', 'height', 'screen'].every(key => Number.isFinite(geometry[key]))) {
    const displays = screen.getAllDisplays(); boardScreen = Math.max(0, Math.min(displays.length - 1, Math.trunc(geometry.screen)));
    const display = displays[boardScreen];
    const bounds = {x: Math.round(display.bounds.x + geometry.x), y: Math.round(display.bounds.y + geometry.y), width: Math.max(200, Math.min(4096, Math.round(geometry.width))), height: Math.max(200, Math.min(4096, Math.round(geometry.height)))};
    const current = whiteboard.getBounds();
    if (boardMode==='full'&&Object.keys(bounds).some(key => bounds[key] !== current[key])) whiteboard.setBounds(bounds);
  }
  if(state.whiteboard_visible){if(!whiteboard.isVisible()){whiteboard.setAlwaysOnTop(true);whiteboard.show();whiteboard.focus();}}else whiteboard.hide();
  state.effect ? effects.showInactive() : effects.hide();
});
