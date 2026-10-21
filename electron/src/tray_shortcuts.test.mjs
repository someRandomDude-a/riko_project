import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fakeElectron} from './electron_fakes.mjs';
const {trayTemplate,createTray,startDiscord,registerShortcuts}=createRequire(import.meta.url)('../tray_shortcuts.cjs');
function desktop(){
 const calls=[],errors=[];
 return {calls,errors,actions:{showControls:view=>calls.push(['show',view]),whiteboard:()=>calls.push(['whiteboard']),toggleWhiteboard:()=>calls.push(['toggle']),quit:()=>calls.push(['quit']),
  post:route=>calls.push(['post',route]),request:async route=>{calls.push(['request',route]);return {ok:true};},dialog:{showErrorBox:(...args)=>errors.push(args)}}};
}
test('the tray opens the chat, Settings, Appearance and the whiteboard, starts Discord and quits',async()=>{
 const {calls,actions}=desktop(),electron=fakeElectron();
 const tray=createTray({...electron.module,icon:'/assets/tray.png',name:''},actions);
 assert.deepEqual(tray.image,{file:'/assets/tray.png',size:{width:24,height:24}});assert.deepEqual(electron.calls(tray,'setToolTip')[0].args,['Companion']);
 const menu=tray.menu.template;
 assert.deepEqual(menu.map(item=>item.label||item.type),['Open chat','Start Discord client','Settings','Appearance','Whiteboard','separator','Quit']);
 for(const item of menu)await item.click?.();
 tray['double-click']();
 assert.deepEqual(calls,[['show','chat'],['request','/api/discord/start'],['show','settings'],['show','appearance'],['whiteboard'],['quit'],['show','chat']]);
 assert.deepEqual(trayTemplate(actions).map(item=>item.label||item.type),menu.map(item=>item.label||item.type));
 assert.throws(()=>createTray({...electron.module,nativeImage:{createFromPath:()=>({isEmpty:()=>true})},icon:'missing.png'},actions),/Tray icon is missing or invalid/);
});
test('starting Discord from the tray explains a refusal or a missing backend',async()=>{
 const errors=[],dialog={showErrorBox:(...args)=>errors.push(args)};
 await startDiscord({dialog,request:async()=>({ok:false,json:async()=>({detail:'Discord access is not configured'})})});
 await startDiscord({dialog,request:async()=>({ok:false,json:async()=>{throw new Error('not json');}})});
 await startDiscord({dialog,request:async()=>{throw new Error('ECONNREFUSED');}});
 await startDiscord({dialog,request:async()=>({ok:true})});
 assert.deepEqual(errors,[['Discord','Discord access is not configured'],['Discord','Discord could not be started'],['Discord','The Python backend is not available. Start it before launching Discord.']]);
});
test('only the summon shortcut is global by default; configured ones bind their actions, and a failed one only warns',()=>{
 const {calls,actions}=desktop(),registered=[],warnings=[];
 const globalShortcut={register:(accelerator,fn)=>{registered.push([accelerator,fn]);return accelerator!=='Alt+Taken';}};
 registerShortcuts(globalShortcut,undefined,actions,(...args)=>warnings.push(args));
 assert.deepEqual(registered.map(([accelerator])=>accelerator),['CommandOrControl+Shift+Space']);registered[0][1]();assert.deepEqual(calls,[['show',undefined]]);
 registered.length=0;calls.length=0;
 registerShortcuts(globalShortcut,{popup:null,quit:'Alt+Q',whiteboard:'Alt+W',settings:'Alt+S',mic:'Alt+M',audio:'Alt+A',sleep:'Alt+Taken'},actions,(...args)=>warnings.push(args));
 for(const [,fn] of registered)fn();
 assert.deepEqual(calls,[['quit'],['toggle'],['show','settings'],['post','/api/mic/toggle'],['post','/api/audio/toggle'],['post','/api/sleep/toggle']]);
 registerShortcuts({register:()=>{throw new Error('bad accelerator');}},{popup:'Nonsense+'},actions,(...args)=>warnings.push(args));
 assert.deepEqual(warnings,[['Shortcut unavailable:','sleep','Alt+Taken'],['Invalid shortcut:','popup','bad accelerator']]);
});
