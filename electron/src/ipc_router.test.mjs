import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {senderWindow,registerRoutes,handle,on}=createRequire(import.meta.url)('../ipc_router.cjs');
function fixture(){
 const window=(id,destroyed=false)=>({webContents:{id},isDestroyed:()=>destroyed});
 const all={overlay:window(1),control:window(2),whiteboard:window(3,true)};
 const windows={get:name=>all[name]??null},ipc={handlers:new Map(),listeners:new Map()};
 const ipcMain={handle:(channel,fn)=>ipc.handlers.set(channel,fn),on:(channel,fn)=>ipc.listeners.set(channel,fn)};
 // Electron turns a handler's synchronous throw into a rejected invoke.
 const invoke=async(channel,id,...args)=>ipc.handlers.get(channel)({sender:{id}},...args);
 return {all,windows,ipcMain,ipc,invoke,send:(channel,id,...args)=>ipc.listeners.get(channel)({sender:{id}},...args)};
}
test('a request is answered only for a live window the route names',()=>{
 const {all,windows}=fixture();
 assert.deepEqual(senderWindow(windows,['overlay','control'],{sender:{id:2}}),{name:'control',window:all.control});
 assert.equal(senderWindow(windows,['overlay'],{sender:{id:2}}),null,'another window');
 assert.equal(senderWindow(windows,['whiteboard'],{sender:{id:3}}),null,'a destroyed window');
 assert.equal(senderWindow(windows,['setup'],{sender:{id:4}}),null,'a window that does not exist yet');
 assert.equal(senderWindow(windows,['overlay'],{}),null);
});
test('invokes from anyone else reject with the route refusal; sends from anyone else are dropped',async()=>{
 const h=fixture(),seen=[];
 const channels=registerRoutes(h.ipcMain,h.windows,[
  handle('ask',['control'],'Control renderer required',(sender,value)=>{seen.push(['ask',sender.name,value]);return value*2;}),
  on('tell',['overlay','control'],(sender,value)=>seen.push(['tell',sender.name,value]))]);
 assert.deepEqual([...channels],['ask','tell']);
 assert.equal(await h.invoke('ask',2,21),42);
 for(const id of [1,3,99])await assert.rejects(h.invoke('ask',id,1),/^Error: Control renderer required$/);
 h.send('tell',1,'a');h.send('tell',2,'b');h.send('tell',3,'c');h.send('tell',99,'d');
 assert.deepEqual(seen,[['ask','control',21],['tell','overlay','a'],['tell','control','b']]);
});
test('a channel has one route, and every route names its windows, a handler and, to invoke, a refusal',()=>{
 const h=fixture(),noop=()=>{};
 assert.throws(()=>registerRoutes(h.ipcMain,h.windows,[on('x',['overlay'],noop),on('x',['control'],noop)]),/registered twice: x/);
 for(const route of [on('y',[],noop),on('y','overlay',noop),{channel:'y',from:['overlay'],invoke:false},handle('y',['overlay'],'',noop)])
  assert.throws(()=>registerRoutes(fixture().ipcMain,h.windows,[route]),/needs its windows/,JSON.stringify(route));
});
