const {contextBridge, ipcRenderer} = require('electron');
contextBridge.exposeInMainWorld('gestureBridge',{send:value=>ipcRenderer.send('window-gesture',value)});
contextBridge.exposeInMainWorld('chatSurfaceBridge',{subscribe:callback=>{const listener=(_event,value)=>callback(value);ipcRenderer.on('chat-surface',listener);return()=>ipcRenderer.removeListener('chat-surface',listener);}});
contextBridge.exposeInMainWorld('popupBridge',{interactive:value=>ipcRenderer.send('popup-interactive',value)});
function micAnchor(){const r=document.querySelector('.dock-mic')?.getBoundingClientRect();return r?.width?{x:window.screenX+r.x+r.width/2,y:window.screenY+r.y+r.height/2}:undefined;}
function reducedMotion(){return !!document.querySelector('.reduce-motion')||window.matchMedia('(prefers-reduced-motion: reduce)').matches;}
contextBridge.exposeInMainWorld('windowBridge',{state:()=>ipcRenderer.invoke('window-state'),action:action=>ipcRenderer.invoke('window-action',action,micAnchor(),reducedMotion()),move:delta=>ipcRenderer.invoke('move-window',delta),mode:mode=>ipcRenderer.invoke('chat-mode',mode,micAnchor(),reducedMotion()),subscribe:callback=>{const listener=(_event,state)=>callback(state);ipcRenderer.on('window-state',listener);return()=>ipcRenderer.removeListener('window-state',listener);}});
contextBridge.exposeInMainWorld('dockBridge',{interactive:enabled=>ipcRenderer.send('dock-interactive',enabled),open:anchor=>ipcRenderer.send('open-compact-chat',anchor)});
contextBridge.exposeInMainWorld('materialBridge',{set:enabled=>ipcRenderer.invoke('window-material',enabled)});
contextBridge.exposeInMainWorld('compactBridge',{interactive:enabled=>ipcRenderer.send('compact-interactive',!!enabled),scale:size=>ipcRenderer.invoke('compact-scale',size)});
contextBridge.exposeInMainWorld('controlVisibilityBridge', {get:()=>ipcRenderer.invoke('control-visible'),subscribe:callback=>{const listener=(_event,value)=>callback(value);ipcRenderer.on('control-visibility',listener);return()=>ipcRenderer.removeListener('control-visibility',listener);}});
contextBridge.exposeInMainWorld('whiteboardBridge', {capture: rect => ipcRenderer.invoke('capture-whiteboard', rect)});
contextBridge.exposeInMainWorld('processBridge', {sync:()=>ipcRenderer.send('sync-processes')});
contextBridge.exposeInMainWorld('approvalBridge', {interactive: enabled => ipcRenderer.send('approval-interactive', enabled)});
contextBridge.exposeInMainWorld('riko', {showControl: () => ipcRenderer.send('show-control'), showWhiteboard: () => ipcRenderer.send('show-whiteboard'), syncSurfaces: state => ipcRenderer.send('sync-surfaces', state), displays: () => ipcRenderer.invoke('displays'), avatarInteractive: enabled => ipcRenderer.send('avatar-interactive', enabled), pickPath: options => ipcRenderer.invoke('pick-path', options), onNavigate: callback => {const listener=(_event,view)=>callback(view);ipcRenderer.on('navigate',listener);return()=>ipcRenderer.removeListener('navigate',listener);}});
