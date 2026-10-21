import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import path from 'node:path';
import {createRequire} from 'node:module';
const {desktopRoutes,confirmChange,pathDialogOptions,captureWhiteboard}=createRequire(import.meta.url)('../ipc_routes.cjs');
// Who may use each channel, written out rather than derived, so widening a route is a deliberate change here too.
// main_wiring.test.mjs checks that main serves exactly these, that the preload uses exactly these, and that every other
// window is refused on each.
const CHROME=['control','whiteboard'],OWNERS={
 'setup-hardware':['setup'],'setup-directory':['setup'],'setup-model':['setup'],'setup-sovits':['setup'],'setup-finish':['setup'],
 'neural-data-open':['control'],'sync-processes':['control','overlay','whiteboard','effects'],'settings-dirty':['control'],
 'show-control':['overlay','control','whiteboard'],'show-whiteboard':['control'],
 'window-state':CHROME,'window-action':CHROME,'chat-mode':CHROME,'window-material':CHROME,'window-gesture':CHROME,
 'compact-interactive':['control'],'compact-scale':['control'],'control-visible':['overlay'],'capture-whiteboard':['whiteboard'],
 'displays':['control'],'confirm-security-change':['control'],'pick-path':['control'],'avatar-cursor':['overlay'],
 'popup-interactive':['overlay'],'avatar-interactive':['overlay'],'approval-interactive':['overlay'],'overlay-drag':['overlay'],'sync-surfaces':['overlay']};
test('every channel names the windows that own it; neither the setup nor the training-data page reaches the app',()=>{
 const routes=desktopRoutes({windows:{pointer:{}}});
 assert.deepEqual(Object.fromEntries(routes.map(route=>[route.channel,route.from])),OWNERS);
 assert.ok(routes.every(route=>!route.from.includes('neural-data')));
 assert.deepEqual(routes.filter(route=>route.from.includes('setup')).map(route=>route.channel),['setup-hardware','setup-directory','setup-model','setup-sovits','setup-finish']);
});
test('the overlay routes move the pointer state, and its cursor reports buttons only where the OS tracks them',()=>{
 const calls=[],pointer={interactive:false,buttons:1,hover:(...args)=>calls.push(['hover',...args]),drag:(...args)=>calls.push(['drag',...args])};
 const screen={getCursorScreenPoint:()=>({x:110,y:220})},overlay={getContentBounds:()=>({x:10,y:20,width:100,height:100})};
 const route=(platform,channel)=>desktopRoutes({windows:{pointer},screen,platform:{nativeButtons:platform==='win32'}}).find(r=>r.channel===channel).handler;
 assert.throws(()=>route('win32','avatar-cursor')({window:overlay}),/Interactive overlay renderer required/);
 pointer.interactive=true;
 assert.deepEqual(route('win32','avatar-cursor')({window:overlay}),{x:100,y:200,buttons:1});assert.deepEqual(route('darwin','avatar-cursor')({window:overlay}),{x:100,y:200,buttons:null});
 route('darwin','popup-interactive')({},1);route('darwin','approval-interactive')({},0);
 for(const value of [{source:'avatar',active:true},{source:'window',active:true},{source:'popup',active:'yes'},null])route('darwin','overlay-drag')({},value);
 assert.deepEqual(calls,[['hover','popup',true],['hover','approval',false],['drag','avatar',true]]);
});
test('surface sync republishes the displays the backend lost before applying the surfaces',()=>{
 const calls=[],displays={reset:()=>calls.push('reset'),publish:()=>calls.push('publish')},windows={pointer:{},syncSurfaces:state=>calls.push(['sync',state])};
 const sync=desktopRoutes({windows,displays}).find(r=>r.channel==='sync-surfaces').handler;
 sync({},null);sync({},'state');sync({},{has_displays:true});sync({},{has_displays:false});
 assert.deepEqual(calls,['publish',['sync',{has_displays:true}],'reset','publish',['sync',{has_displays:false}]]);
});
test('a security confirmation shows exactly what is signed, and signs only an allowed change with the key',async()=>{
 const shown=[],dialog={showMessageBox:async(parent,options)=>{shown.push([parent,options]);return {response:dialog.answer};}};
 const challenge=JSON.stringify({action:'settings',changes:{'runtime.native_library':'/lib‮exe.so','tools.require_approval':false}});
 dialog.answer=0;assert.equal(await confirmChange(dialog,'control',challenge,()=>'key'),null);
 const [parent,options]=shown[0];
 assert.equal(parent,'control');assert.equal(options.message,'Change security-sensitive settings?');assert.deepEqual([options.buttons,options.defaultId,options.cancelId],[['Cancel','Allow'],0,0]);
 assert.equal(options.detail,'runtime.native_library: "/lib\\u{202e}exe.so"\ntools.require_approval: false\n\nOnly allow this if you made this change yourself.');
 dialog.answer=1;
 assert.equal(await confirmChange(dialog,'control',challenge,()=>'key'),crypto.createHmac('sha256','key').update(challenge).digest('hex'));
 assert.equal(await confirmChange(dialog,'control',challenge,()=>''),null,'no key, no signature');
 for(const [action,title] of [['tool_approvals','Let these tools run without asking first?'],['discord_access','Change who can use Riko from Discord?'],['other','Confirm this change?']]){
  await confirmChange(dialog,'control',JSON.stringify({action}),()=>'key');assert.equal(shown.at(-1)[1].message,title);
 }
 await assert.rejects(confirmChange(dialog,'control','{',()=>'key'),/Invalid confirmation request/);
});
test('path pickers filter only by plain extensions',()=>{
 assert.deepEqual(pathDialogOptions({directory:true,defaultPath:'/home/me'}),{title:'Choose a folder',properties:['openDirectory'],defaultPath:'/home/me'});
 assert.deepEqual(pathDialogOptions({extensions:['gguf','bin']}),{title:'Choose a file',filters:[{name:'Supported files',extensions:['gguf','bin']}],properties:['openFile']});
 for(const extensions of [['gguf','../x'],['*'],[],'gguf',[1]])assert.equal(pathDialogOptions({extensions}).filters,undefined,JSON.stringify(extensions));
 assert.deepEqual(pathDialogOptions(),{title:'Choose a file',properties:['openFile']});
});
test('a whiteboard capture stays inside the page and comes back at most 1280 px on its longer side',async()=>{
 const captured=[],board=size=>({getContentBounds:()=>({x:0,y:0,width:3000,height:900}),webContents:{capturePage:async(rect,options)=>{captured.push([rect,options]);return {getSize:()=>size,resize:next=>({toPNG:()=>next}),toPNG:()=>size};}}});
 assert.deepEqual(await captureWhiteboard(board({width:2560,height:640}),{x:0.4,y:1.6,width:2560,height:640}),{width:1280,height:320});
 assert.deepEqual(captured[0],[{x:0,y:2,width:2560,height:640},{stayHidden:true,stayAwake:true}]);
 assert.deepEqual(await captureWhiteboard(board({width:600,height:300}),{x:0,y:0,width:600,height:300}),{width:600,height:300});
 for(const rect of [null,{x:-1,y:0,width:10,height:10},{x:0,y:0,width:0.5,height:10},{x:2995,y:0,width:10,height:10},{x:0,y:0,width:'10',height:10}])
  await assert.rejects(captureWhiteboard(board({width:1,height:1}),rect),/Invalid whiteboard capture bounds/,JSON.stringify(rect));
});
test('setup finishes by recording the data folder outside the application and relaunching',async()=>{
 const calls=[],written=[];
 const app={getPath:name=>({exe:'/Applications/Riko.app/Contents/MacOS/Riko',userData:'/Users/me/Library/Riko'}[name]),relaunch:()=>calls.push('relaunch'),quit:()=>calls.push('quit')};
 const release={saveSetup:async(...args)=>{calls.push(['saveSetup',...args]);if(args[1].name==='Broken')throw new Error('Invalid context/output budget');return '/data/Riko';},installRoot:exe=>exe.split('/Contents')[0]};
 const finish=desktopRoutes({windows:{pointer:{}},app,release,resources:'/Applications/Riko.app/Contents/Resources',files:{writeFileSync:(...args)=>written.push(args)}}).find(r=>r.channel==='setup-finish').handler;
 assert.equal(await finish({},{directory:'/data/Riko',settings:{name:'Mika'}}),true);
 assert.deepEqual(calls,[['saveSetup','/data/Riko',{name:'Mika'},'/Applications/Riko.app/Contents/Resources','/Applications/Riko.app'],'relaunch','quit']);
 assert.deepEqual(written,[[path.join('/Users/me/Library/Riko','data-location.json'),'{"directory":"/data/Riko"}']]);
 // The backend's answer arrives later: a refusal reaches the page, and nothing is recorded or relaunched.
 calls.length=0;written.length=0;
 await assert.rejects(finish({},{directory:'/data/Riko',settings:{name:'Broken'}}),/Invalid context\/output budget/);
 assert.equal(calls.length,1);assert.deepEqual(written,[]);
});
