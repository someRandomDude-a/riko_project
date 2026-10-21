const {shortcutBindings}=require('./shortcuts.cjs');
// The tray menu and the global shortcuts, both acting through the same few desktop actions (main.cjs wires them):
// showControls(view), whiteboard(), toggleWhiteboard(), quit(), post(route) for the backend toggles, and request and dialog
// for the Discord start.
async function startDiscord({request,dialog}){
 try{
  const response=await request('/api/discord/start',{method:'POST'});
  if(!response.ok){let message='Discord could not be started';try{const body=await response.json();if(typeof body.detail==='string')message=body.detail;}catch{}dialog.showErrorBox('Discord',message);}
 }catch{dialog.showErrorBox('Discord','The Python backend is not available. Start it before launching Discord.');}
}
const trayTemplate=actions=>[{label:'Open chat',click:()=>actions.showControls('chat')},{label:'Start Discord client',click:()=>startDiscord(actions)},
 {label:'Settings',click:()=>actions.showControls('settings')},{label:'Appearance',click:()=>actions.showControls('appearance')},
 {label:'Whiteboard',click:()=>actions.whiteboard()},{type:'separator'},{label:'Quit',click:()=>actions.quit()}];
function createTray({Tray,Menu,nativeImage,icon,name},actions){
 const image=nativeImage.createFromPath(icon);
 if(image.isEmpty())throw new Error('Tray icon is missing or invalid: assets/tray.png');
 const tray=new Tray(image.resize({width:24,height:24}));tray.setToolTip(name||'Companion');
 tray.setContextMenu(Menu.buildFromTemplate(trayTemplate(actions)));
 tray.on('double-click',()=>actions.showControls('chat'));
 return tray;
}
// Key order is registration order. desktop.shortcuts binds these names; shortcuts.cjs keeps only the summon shortcut by default.
const shortcutActions=actions=>({popup:()=>actions.showControls(),quit:()=>actions.quit(),whiteboard:()=>actions.toggleWhiteboard(),
 settings:()=>actions.showControls('settings'),
 mic:()=>actions.post('/api/mic/toggle'),audio:()=>actions.post('/api/audio/toggle'),sleep:()=>actions.post('/api/sleep/toggle')});
function registerShortcuts(globalShortcut,configured,actions,warn=console.warn){
 const handlers=shortcutActions(actions);
 for(const [name,accelerator] of shortcutBindings(configured,Object.keys(handlers))){
  try{if(!globalShortcut.register(accelerator,handlers[name]))warn('Shortcut unavailable:',name,accelerator);}catch(error){warn('Invalid shortcut:',name,error.message);}
 }
}
module.exports={startDiscord,trayTemplate,createTray,shortcutActions,registerShortcuts};
