const {chatBounds,clampBounds,compactSize}=require('./window_layout.cjs');
const {gestureLoop}=require('./window_gesture.cjs');
const {syncWindowContent}=require('./window_content.cjs');
const {windowAction}=require('./window_actions.cjs');
const {overlayInput,pointerForwarder,forwardTargets}=require('./overlay_input.cjs');
const TRANSPARENT='#00000000';
// Standalone pages paint no canvas (morph.css keeps html/body transparent), so these windows supply style.css's --bg.
const PAGE_BACKGROUND='#101116';
// BrowserWindow options for each window. preferences(extra) is main's rendererPreferences: the preload, isolated from Node,
// with the backend origin. The overlay and effects layers cover a display and never take focus; the chat and whiteboard are
// frameless and transparent, drawing their own chrome; none shows in the taskbar.
function windowOptions(kind,{bounds,title='',preferences}){
 const layer={...bounds,show:false,transparent:true,backgroundColor:TRANSPARENT,frame:false,resizable:false,alwaysOnTop:true,skipTaskbar:true,focusable:false};
 switch(kind){
  case 'overlay':return {...layer,webPreferences:preferences({backgroundThrottling:false})};
  case 'effects':return {...layer,webPreferences:preferences()};
  case 'control':return {width:1050,height:800,minWidth:600,minHeight:450,show:false,frame:false,thickFrame:false,hasShadow:false,transparent:true,backgroundColor:TRANSPARENT,skipTaskbar:true,alwaysOnTop:true,title:title||'Chat',webPreferences:preferences({backgroundThrottling:false})};
  case 'whiteboard':return {width:900,height:700,minWidth:320,minHeight:240,show:false,frame:false,transparent:true,backgroundColor:TRANSPARENT,skipTaskbar:true,alwaysOnTop:true,title:title?title+' — Whiteboard':'Whiteboard',webPreferences:preferences()};
  case 'setup':return {width:1000,height:850,backgroundColor:PAGE_BACKGROUND,webPreferences:preferences()};
  case 'neural-data':return {width:1000,height:800,title:'Expression training data',backgroundColor:PAGE_BACKGROUND,webPreferences:preferences()};
 }
 throw new Error('Unknown window: '+kind);
}
// Pages never navigate away; model links open in the user's browser, never in a privileged renderer.
function protectNavigation(window,openExternal){
 window.webContents.on('will-navigate',event=>event.preventDefault());
 window.webContents.setWindowOpenHandler(({url})=>{if(/^https?:\/\//i.test(url))openExternal(url).catch(()=>{});return {action:'deny'};});
}
const alive=window=>!!window&&!window.isDestroyed();
// Every window and its state: the overlay (avatar, popups, approvals; click-through except where overlayInput says), the chat
// (`control`: full, compact or collapsed to the dock), the whiteboard (full or collapsed), the effects layer, and the setup and
// training-data pages. Each mode machine owns its bounds; the IPC routes, tray and shortcuts call the methods below.
function windowManager({BrowserWindow,screen,dialog,app,page,preferences,icon,openExternal,platform,patchBoard,settings,timers={set:setTimeout,clear:clearTimeout,now:Date.now}}){
 const windows={overlay:null,control:null,whiteboard:null,effects:null,setup:null,'neural-data':null};
 const chat={mode:'full',full:null,compact:null,collapsed:null,tween:null,blur:true,ignoring:false};
 const board={mode:'full',bounds:null,tween:null,geometry:null,screen:0};
 const gesture=gestureLoop({screen,setTimer:timers.set,clearTimer:timers.clear,now:timers.now});
 const pointer=overlayInput(enabled=>{if(alive(windows.overlay))windows.overlay.setIgnoreMouseEvents(!enabled,{forward:true});});
 const forwarder=platform.forwardsPointer?pointerForwarder({cursor:()=>screen.getCursorScreenPoint(),bounds:w=>w.getContentBounds(),
  targets:()=>forwardTargets({point:screen.getCursorScreenPoint(),overlay:windows.overlay,overlayInteractive:pointer.interactive,control:windows.control,controlIgnoring:chat.ignoring,gesture:gesture.active,covering:[windows.whiteboard,windows.setup,windows['neural-data']]}),
  send:(w,point)=>w.webContents.send('forwarded-pointer',point),schedule:timers.set,cancel:timers.clear}):null;
 let debug=false;
 const protect=window=>protectNavigation(window,openExternal);
 function tweenBounds(window,target,state,instant=false){
  timers.clear(state.tween);
  // Resize once. Repeated native resizing reallocates the transparent swapchain and reflows every message/WebGL surface;
  // only compositor properties animate (window-state transitioning).
  window.webContents.send('window-state',{transitioning:!instant});
  window.setBounds(target);syncWindowContent(window);
  if(!instant)state.tween=timers.set(()=>{if(!window.isDestroyed())window.webContents.send('window-state',{transitioning:false});},180);
 }
 function setControlMode(mode,anchor,instant=false){
  gesture.end();
  if(!['full','compact','collapsed'].includes(mode))throw new Error('Unsupported chat mode');
  const control=windows.control;
  if(mode!==chat.mode){
   const current=control.getBounds(),area=screen.getDisplayMatching(current).workArea;
   if(chat.mode==='full')chat.full=control.getNormalBounds();else chat[chat.mode]=current;
   if(control.isMaximized())control.unmaximize();
   const target=chatBounds(mode,current,area,anchor&&Number.isFinite(anchor.x)&&Number.isFinite(anchor.y)?anchor:undefined,chat[mode]);
   chat.mode=mode;control.setMinimumSize(88,88);control.setResizable(mode==='full');control.setAspectRatio(0);control.setFocusable(mode!=='collapsed');control.setAlwaysOnTop(true);control.setIgnoreMouseEvents(chat.ignoring=mode!=='full',{forward:true});
   platform.material(control,mode,chat.blur);
   control.webContents.send('window-state',{maximized:false,mode,dockWidth:mode==='collapsed'?target.width:chat.collapsed?.width||88});tweenBounds(control,target,chat,instant);
  }
  control.webContents.send('window-state',{maximized:control.isMaximized(),mode:chat.mode});
  if(alive(windows.overlay))windows.overlay.webContents.send('control-visibility',mode!=='collapsed');
  if(mode==='collapsed')control.showInactive();else{if(control.isMinimized())control.restore();control.show();control.focus();}
  syncWindowContent(control);
 }
 function setBoardMode(mode,instant=false){
  if(!['full','collapsed'].includes(mode))throw new Error('Unsupported whiteboard mode');
  const whiteboard=windows.whiteboard;
  if(mode!==board.mode){
   const current=whiteboard.getBounds(),area=screen.getDisplayMatching(current).workArea;if(mode==='collapsed')board.bounds=current;
   const width=mode==='collapsed'?240:board.bounds?.width||900,height=mode==='collapsed'?76:board.bounds?.height||700;
   const target=clampBounds({x:current.x+(current.width-width)/2,y:current.y+current.height-height,width,height},area);
   board.mode=mode;whiteboard.setMinimumSize(200,60);whiteboard.webContents.send('window-state',{maximized:false,mode});tweenBounds(whiteboard,target,board,instant);
  }
  whiteboard.setAlwaysOnTop(true);whiteboard.show();whiteboard.focus();
 }
 function showControls(view,mode=chat.mode,anchor){
  if(view&&view!=='chat')mode='full';else if(mode==='collapsed')mode='compact';
  setControlMode(mode,anchor);const control=windows.control;
  if(control.isMinimized())control.restore();control.show();control.focus();if(view)control.webContents.send('navigate',view);
 }
 function createWindows(name='',options={}){
  debug=options.debug===true;
  const bounds=screen.getPrimaryDisplay().bounds;
  const overlay=windows.overlay=new BrowserWindow(windowOptions('overlay',{bounds,preferences}));
  overlay.once('ready-to-show',()=>{overlay.setFocusable(false);overlay.showInactive();});
  overlay.setIgnoreMouseEvents(true,{forward:true});overlay.loadURL(page('overlay'));
  platform.hookOverlay(overlay,{press:()=>pointer.press(),release:()=>{pointer.release();const p=screen.getCursorScreenPoint(),b=overlay.getContentBounds();overlay.webContents.send('overlay-pointer',{x:p.x-b.x,y:p.y-b.y,buttons:0,phase:'up'});}});
  const control=windows.control=new BrowserWindow(windowOptions('control',{title:name,preferences}));
  for(const event of ['show','hide','minimize','restore'])control.on(event,()=>{if(alive(overlay))overlay.webContents.send('control-visibility',chat.mode!=='collapsed'&&control.isVisible()&&!control.isMinimized());});
  control.loadURL(page('control'));
  const publishChatSurface=()=>{if(alive(overlay))overlay.webContents.send('chat-surface',{mode:chat.mode,bounds:control.getBounds(),screen:overlay.getBounds(),visible:control.isVisible()&&!control.isMinimized()});};
  for(const event of ['move','resize','show','hide'])control.on(event,publishChatSurface);
  overlay.webContents.on('did-finish-load',publishChatSurface);
  platform.hookChat(control,()=>{if(gesture.active?.window===control)gesture.end();});
  control.webContents.once('did-finish-load',()=>{if(!debug){const a=screen.getPrimaryDisplay().workArea;setControlMode('collapsed',{x:a.x+a.width-80,y:a.y+a.height-70},true);}});
  control.on('close',event=>{if(!app.isQuitting){event.preventDefault();setControlMode(chat.mode==='full'?'compact':'collapsed');}});
  // Electron cancels a vetoed unload silently. before-quit already asked about unsaved Settings; a reload asks here.
  control.webContents.on('will-prevent-unload',event=>{if(app.isQuitting||dialog.showMessageBoxSync(control,{type:'question',buttons:['Discard changes','Keep editing'],defaultId:1,cancelId:1,message:'Discard unsaved settings changes?'})===0){settings.dirty=false;event.preventDefault();}});
  control.webContents.on('render-process-gone',()=>{settings.dirty=false;});
  control.on('hide',()=>{if(chat.mode==='compact')chat.compact=control.getBounds();});
  control.on('minimize',()=>{if(app.isQuitting)return;control.setFocusable(false);control.restore();setControlMode('collapsed');});
  const whiteboard=windows.whiteboard=new BrowserWindow(windowOptions('whiteboard',{title:name,preferences}));
  whiteboard.loadURL(page('whiteboard'));
  whiteboard.on('close',event=>{if(!app.isQuitting){event.preventDefault();setBoardMode('collapsed');}});
  // The backend keeps the whiteboard's geometry relative to its display (sync-surfaces brings it back). While a move or resize is
  // still being saved (board.geometry pending), syncSurfaces leaves the bounds alone: an older saved geometry must not snap it back.
  const saveGeometry=()=>{
   timers.clear(board.geometry);
   board.geometry=timers.set(()=>{
    board.geometry=null;
    if(whiteboard.isDestroyed())return;
    const bounds=board.mode==='collapsed'?(board.bounds||whiteboard.getBounds()):whiteboard.getBounds(),displays=screen.getAllDisplays();
    const display=screen.getDisplayMatching(bounds);board.screen=displays.findIndex(item=>item.id===display.id);
    patchBoard({geometry:{...bounds,x:bounds.x-display.bounds.x,y:bounds.y-display.bounds.y,screen:board.screen}});
   },200);
  };
  whiteboard.on('move',saveGeometry);whiteboard.on('resize',saveGeometry);
  const effects=windows.effects=new BrowserWindow(windowOptions('effects',{bounds,preferences}));
  effects.setIgnoreMouseEvents(true);effects.loadURL(page('effects'));
  for(const window of [overlay,control,whiteboard,effects]){protect(window);window.setIcon(icon);}
  for(const window of [control,whiteboard]){
   window.setMenuBarVisibility(false);window.on('resize',()=>syncWindowContent(window));window.webContents.on('did-finish-load',()=>syncWindowContent(window));
   for(const event of ['maximize','unmaximize'])window.on(event,()=>{syncWindowContent(window);window.webContents.send('window-state',{maximized:window.isMaximized()});});
  }
  forwarder?.start();
 }
 return {
  get:name=>windows[name],pointer,createWindows,setControlMode,setBoardMode,showControls,
  get chatMode(){return chat.mode;},get boardMode(){return board.mode;},
  // First run (packaged, no data folder yet): only the setup page, before any backend exists.
  openSetup(){const setup=windows.setup=new BrowserWindow(windowOptions('setup',{preferences}));protect(setup);return setup.loadURL(page('setup'));},
  openNeuralData(){
   const current=windows['neural-data'];
   if(alive(current)){current.show();current.focus();return;}
   const window=windows['neural-data']=new BrowserWindow(windowOptions('neural-data',{preferences}));protect(window);window.loadURL(page('neural-data'));
  },
  // A second launch wakes this one.
  focus(){if(alive(windows.setup))windows.setup.focus();else if(alive(windows.control))showControls();},
  showWhiteboard(){patchBoard({visible:true});setBoardMode('full');},
  toggleWhiteboard(){patchBoard({visible:!windows.whiteboard.isVisible()});},
  // The frameless chrome of the chat and whiteboard (window_chrome.jsx): state, the title-bar buttons and the mode switch.
  chromeState(name){return {maximized:windows[name].isMaximized(),mode:name==='control'?chat.mode:board.mode,dockWidth:chat.collapsed?.width||88};},
  chromeAction(name,action,anchor,instant){
   if(['close','minimize'].includes(action)){if(name==='control')setControlMode(chat.mode==='full'?'compact':'collapsed',anchor,instant);else setBoardMode('collapsed',instant);return {mode:name==='control'?chat.mode:board.mode};}
   if(name==='control'&&action==='maximize'&&chat.mode!=='full')setControlMode('full',anchor,instant);
   return windowAction(windows[name],action);
  },
  chromeMode(name,mode,anchor,instant){if(name==='control')setControlMode(mode,anchor,instant);else setBoardMode(mode,instant);return {mode:name==='control'?chat.mode:board.mode};},
  material(name,enabled){
   if(typeof enabled!=='boolean')throw new Error('Invalid material preference');
   const window=windows[name];if(name==='control'&&chat.mode==='full')chat.blur=enabled;
   const result=platform.material(window,name==='control'?chat.mode:'transparent',name==='control'&&chat.blur);syncWindowContent(window);return result;
  },
  // The dock and mini chat are click-through except where the renderer's hit test says otherwise, and never during a gesture. A report
  // that changes nothing makes no native call.
  compactInteractive(enabled){if(chat.mode!=='full'&&!gesture.active&&chat.ignoring!==!enabled)windows.control.setIgnoreMouseEvents(chat.ignoring=!enabled,{forward:true});},
  windowGesture(name,value){
   if(!value||typeof value.id!=='string'||value.id.length>64)return;
   const window=windows[name];
   if(value.phase==='end'){gesture.endIf(value.id,window);return;}
   if(value.phase!=='begin'||!['move','resize'].includes(value.kind)||(value.kind==='resize'&&(name!=='control'||chat.mode==='full')))return;
   gesture.end();timers.clear(name==='control'?chat.tween:board.tween);window.webContents.send('window-state',{transitioning:false});
   window.setIgnoreMouseEvents(false);if(name==='control')chat.ignoring=false;
   gesture.begin({id:value.id,window,kind:value.kind,mode:chat.mode,onBounds(next,g){
    if(name!=='control')return;
    if(chat.mode==='compact')chat.compact=next;
    else if(chat.mode==='collapsed'){chat.collapsed=next;if(g.kind==='resize')window.webContents.send('window-state',{mode:chat.mode,dockWidth:next.width});}
   }});
  },
  compactScale(size){
   if(chat.mode==='full'||!size||!Number.isFinite(size.width)||!Number.isFinite(size.height))throw new Error('Dock or compact chat required');
   const control=windows.control;timers.clear(chat.tween);
   const b=control.getBounds(),a=screen.getDisplayMatching(b).workArea,{width,height}=compactSize(chat.mode,a,size.width,size.height);
   control.setBounds(clampBounds({x:b.x,y:b.y+b.height-height,width,height},a));
   if(chat.mode==='compact')chat.compact=control.getBounds();else{chat.collapsed=control.getBounds();control.webContents.send('window-state',{mode:chat.mode,dockWidth:width});}
  },
  controlVisible(){const control=windows.control;return alive(control)&&chat.mode!=='collapsed'&&control.isVisible()&&!control.isMinimized();},
  // The overlay reports the backend's surface state: the avatar's display, the whiteboard's geometry and visibility, effects.
  syncSurfaces(state){
   const {overlay,whiteboard,effects}=windows,geometry=state.whiteboard_geometry;
   if(Number.isInteger(state.avatar_screen)){
    const display=screen.getAllDisplays()[state.avatar_screen]||screen.getPrimaryDisplay();
    const bounds=overlay.getBounds();if(Object.keys(display.bounds).some(key=>bounds[key]!==display.bounds[key])){overlay.setBounds(display.bounds);effects.setBounds(display.bounds);}
   }
   if(geometry&&['x','y','width','height','screen'].every(key=>Number.isFinite(geometry[key]))){
    const displays=screen.getAllDisplays();board.screen=Math.max(0,Math.min(displays.length-1,Math.trunc(geometry.screen)));
    const display=displays[board.screen];
    const bounds={x:Math.round(display.bounds.x+geometry.x),y:Math.round(display.bounds.y+geometry.y),width:Math.max(200,Math.min(4096,Math.round(geometry.width))),height:Math.max(200,Math.min(4096,Math.round(geometry.height)))};
    const current=whiteboard.getBounds();
    if(board.mode==='full'&&!board.geometry&&Object.keys(bounds).some(key=>bounds[key]!==current[key]))whiteboard.setBounds(bounds);
   }
   if(state.whiteboard_visible){if(!whiteboard.isVisible()){whiteboard.setAlwaysOnTop(true);whiteboard.show();whiteboard.focus();}}else whiteboard.hide();
   state.effect?effects.showInactive():effects.hide();
  },
  // before-quit: no timer or poll outlives the windows.
  dispose(){forwarder?.stop();gesture.end();timers.clear(board.geometry);timers.clear(chat.tween);timers.clear(board.tween);}};
}
module.exports={windowOptions,protectNavigation,windowManager,PAGE_BACKGROUND};
