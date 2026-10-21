import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {fakeElectron,fakeTimers,PRIMARY} from './electron_fakes.mjs';
const require=createRequire(import.meta.url);
const {windowManager,windowOptions,protectNavigation}=require('../window_manager.cjs');
const {windowPlatform,WM_LBUTTONDOWN,WM_LBUTTONUP}=require('../window_platform.cjs');
const {compactSize}=require('../window_layout.cjs');
const SECOND={id:2,label:'Side',bounds:{x:1920,y:0,width:2560,height:1440},workArea:{x:1920,y:0,width:2560,height:1400},scaleFactor:1};
function setup({platform='darwin',debug=false,displays=[PRIMARY],create=true}={}){
 const electron=fakeElectron({displays}),timers=fakeTimers(),boards=[],opened=[],settings={dirty:false};
 const windows=windowManager({BrowserWindow:electron.module.BrowserWindow,screen:electron.module.screen,dialog:electron.module.dialog,app:electron.module.app,
  page:name=>'app://index.html#/'+name,preferences:(extra={})=>({preload:'preload.cjs',...extra}),icon:'tray.png',openExternal:url=>{opened.push(url);return Promise.resolve();},
  platform:windowPlatform(platform),patchBoard:body=>boards.push(body),settings,timers});
 if(create)windows.createWindows('Mika',{debug});
 const get=name=>electron.window(name);
 return {...electron,windows,timers,boards,opened,settings,overlay:get('overlay'),control:get('control'),whiteboard:get('whiteboard'),effects:get('effects')};
}
const sent=(window,channel)=>window.webContents.sentOn(channel);
const docked=h=>{h.control.webContents.emit('did-finish-load');return h;};

test('each window has the options its role needs, the same on every platform',()=>{
 const preferences=(extra={})=>({preload:'preload.cjs',...extra}),bounds=PRIMARY.bounds;
 for(const kind of ['overlay','effects']){
  const options=windowOptions(kind,{bounds,preferences});
  assert.deepEqual({...options,webPreferences:undefined},{...bounds,show:false,transparent:true,backgroundColor:'#00000000',frame:false,resizable:false,alwaysOnTop:true,skipTaskbar:true,focusable:false,webPreferences:undefined},kind);
 }
 assert.equal(windowOptions('overlay',{bounds,preferences}).webPreferences.backgroundThrottling,false);
 assert.equal(windowOptions('effects',{bounds,preferences}).webPreferences.backgroundThrottling,undefined);
 const control=windowOptions('control',{title:'Mika',preferences});
 assert.deepEqual([control.frame,control.thickFrame,control.hasShadow,control.transparent,control.backgroundColor,control.skipTaskbar,control.alwaysOnTop,control.minWidth,control.minHeight,control.title,control.webPreferences.backgroundThrottling],
  [false,false,false,true,'#00000000',true,true,600,450,'Mika',false]);
 assert.equal(windowOptions('control',{preferences}).title,'Chat');
 assert.equal(windowOptions('whiteboard',{title:'Mika',preferences}).title,'Mika — Whiteboard');assert.equal(windowOptions('whiteboard',{preferences}).title,'Whiteboard');
 // Standalone setup and training-data pages paint the renderer's own page background.
 const bg=fs.readFileSync(new URL('./style.css',import.meta.url),'utf8').match(/--bg:(#[0-9a-f]{6})/)[1];
 for(const kind of ['setup','neural-data'])assert.equal(windowOptions(kind,{preferences}).backgroundColor,bg,kind);
 for(const kind of ['overlay','effects','control','whiteboard','setup','neural-data'])assert.equal(windowOptions(kind,{bounds,preferences}).webPreferences.preload,'preload.cjs',kind);
 assert.throws(()=>windowOptions('settings',{preferences}),/Unknown window/);
});

test('the overlay stays click-through and is shown without ever taking focus',()=>{
 const h=setup();
 assert.equal(h.overlay.ignoring,true);assert.equal(h.overlay.forward,true);assert.equal(h.effects.ignoring,true);
 h.overlay.emit('ready-to-show');
 assert.equal(h.overlay.focusable,false);assert.equal(h.overlay.visible,true);
 assert.deepEqual(h.calls(h.overlay,'show').concat(h.calls(h.overlay,'focus')),[]);
 h.windows.pointer.hover('avatar',true);assert.equal(h.overlay.ignoring,false);
 h.windows.pointer.hover('avatar',false);assert.equal(h.overlay.ignoring,true);
 for(const window of [h.overlay,h.control,h.whiteboard,h.effects])assert.deepEqual(h.calls(window,'setIcon').map(e=>e.args),[['tray.png']]);
});

test('once loaded the chat docks at the lower right of the work area, click-through and unfocusable, unless debugging',()=>{
 const h=docked(setup());
 assert.deepEqual(h.control.bounds,{x:1796,y:935,width:88,height:88});
 assert.equal(h.windows.chatMode,'collapsed');assert.equal(h.control.visible,true);assert.equal(h.control.focusable,false);assert.equal(h.control.ignoring,true);
 assert.deepEqual(h.calls(h.control,'focus'),[]);
 assert.deepEqual(sent(h.overlay,'control-visibility').at(-1),false);
 const debug=docked(setup({debug:true}));
 assert.equal(debug.windows.chatMode,'full');assert.deepEqual(debug.control.bounds,{x:0,y:0,width:1050,height:800});
});

test('a mode change resizes the native window once and animates only compositor state',()=>{
 const h=setup();h.control.bounds={x:100,y:100,width:1000,height:700};
 const before=h.calls(h.control,'setBounds').length;
 h.windows.setControlMode('compact',{x:960,y:900});
 assert.equal(h.calls(h.control,'setBounds').length-before,1,'one native resize');
 assert.deepEqual(sent(h.control,'window-state').filter(s=>'transitioning' in s),[{transitioning:true}]);
 h.timers.run(179);assert.equal(sent(h.control,'window-state').filter(s=>'transitioning' in s).length,1);
 h.timers.run(1);assert.deepEqual(sent(h.control,'window-state').filter(s=>'transitioning' in s).at(-1),{transitioning:false});
 // Small modes: resizable only in full, no aspect lock, focusable except as the dock, click-through with forwarding.
 assert.deepEqual(h.calls(h.control,'setMinimumSize').at(-1).args,[88,88]);
 assert.deepEqual(h.calls(h.control,'setResizable').at(-1).args,[false]);assert.deepEqual(h.calls(h.control,'setAspectRatio').at(-1).args,[0]);
 assert.equal(h.control.focusable,true);assert.equal(h.control.ignoring,true);assert.equal(h.control.forward,true);
 assert.deepEqual(h.control.bounds,{x:720,y:310,width:480,height:648});
 h.windows.setControlMode('collapsed',undefined,true);
 assert.equal(h.control.focusable,false);assert.deepEqual(sent(h.control,'window-state').filter(s=>'transitioning' in s).at(-1),{transitioning:false});
 assert.equal(h.timers.pending,0,'an instant change leaves no timer');
 // Full again: the size it had before, focused, taking the mouse.
 h.windows.setControlMode('full',undefined,true);
 assert.deepEqual({width:h.control.bounds.width,height:h.control.bounds.height},{width:1000,height:700});
 assert.equal(h.control.ignoring,false);assert.deepEqual(h.calls(h.control,'setResizable').at(-1).args,[true]);assert.ok(h.calls(h.control,'focus').length>0);
 assert.throws(()=>h.windows.setControlMode('huge'),/Unsupported chat mode/);
});

test('closing or minimizing the chat or whiteboard shrinks it instead of quitting',()=>{
 const h=setup();
 assert.equal(h.control.close().defaultPrevented,true);assert.equal(h.windows.chatMode,'compact');
 assert.equal(h.control.close().defaultPrevented,true);assert.equal(h.windows.chatMode,'collapsed');
 h.windows.setControlMode('full',undefined,true);h.control.emit('minimize');
 assert.equal(h.windows.chatMode,'collapsed');assert.ok(h.calls(h.control,'restore').length>0);
 assert.equal(h.whiteboard.close().defaultPrevented,true);assert.equal(h.windows.boardMode,'collapsed');
 h.module.app.isQuitting=true;
 assert.equal(h.control.close().defaultPrevented,false);assert.equal(h.whiteboard.close().defaultPrevented,false);
});

test('the whiteboard collapses to a bar and returns to its size, saving its geometry relative to its display',()=>{
 const h=setup({displays:[PRIMARY,SECOND]});h.whiteboard.bounds={x:2000,y:100,width:1000,height:800};
 h.windows.setBoardMode('collapsed',true);
 assert.deepEqual(h.whiteboard.bounds,{x:2380,y:824,width:240,height:76});
 h.windows.setBoardMode('full',true);
 assert.deepEqual(h.whiteboard.bounds,{x:2000,y:100,width:1000,height:800});assert.equal(h.whiteboard.visible,true);
 for(let i=0;i<5;i++)h.whiteboard.emit('move');h.whiteboard.emit('resize');
 h.timers.run(199);assert.deepEqual(h.boards,[]);
 h.timers.run(1);assert.deepEqual(h.boards,[{geometry:{x:80,y:100,width:1000,height:800,screen:1}}]);
 h.windows.showWhiteboard();assert.deepEqual(h.boards.at(-1),{visible:true});
 h.windows.toggleWhiteboard();assert.deepEqual(h.boards.at(-1),{visible:false});
 assert.throws(()=>h.windows.setBoardMode('compact'),/Unsupported whiteboard mode/);
});

test('surface sync moves the overlay to the avatar display and applies the backend whiteboard state',()=>{
 const h=setup({displays:[PRIMARY,SECOND]});
 h.windows.syncSurfaces({avatar_screen:1,whiteboard_geometry:{x:10,y:20.4,width:9000,height:50,screen:7},whiteboard_visible:true,effect:true});
 assert.deepEqual(h.overlay.bounds,SECOND.bounds);assert.deepEqual(h.effects.bounds,SECOND.bounds);
 assert.deepEqual(h.whiteboard.bounds,{x:1930,y:20,width:4096,height:200});assert.equal(h.whiteboard.visible,true);assert.equal(h.effects.visible,true);
 assert.deepEqual(h.calls(h.effects,'show'),[],'effects never take focus');
 h.windows.setBoardMode('collapsed',true);const collapsed=h.whiteboard.getBounds();
 h.windows.syncSurfaces({avatar_screen:9,whiteboard_geometry:{x:0,y:0,width:500,height:500,screen:0},whiteboard_visible:false,effect:false});
 assert.deepEqual(h.overlay.bounds,PRIMARY.bounds,'an unknown display falls back to the primary');
 assert.deepEqual(h.whiteboard.bounds,collapsed,'a collapsed whiteboard keeps its bar');assert.equal(h.whiteboard.visible,false);assert.equal(h.effects.visible,false);
});

test('a whiteboard being moved is not snapped back by an older saved geometry; once saved, the backend\'s geometry applies again',()=>{
 const h=setup();h.windows.syncSurfaces({whiteboard_geometry:{x:100,y:100,width:900,height:700,screen:0},whiteboard_visible:true});
 h.whiteboard.bounds={x:300,y:200,width:900,height:700};h.whiteboard.emit('move');  // the user drags it
 h.windows.syncSurfaces({whiteboard_geometry:{x:100,y:100,width:900,height:700,screen:0},whiteboard_visible:true});
 assert.deepEqual(h.whiteboard.bounds,{x:300,y:200,width:900,height:700},'mid-drag, the stale geometry is ignored');
 h.timers.run(200);assert.deepEqual(h.boards.at(-1),{geometry:{x:300,y:200,width:900,height:700,screen:0}});
 h.windows.syncSurfaces({whiteboard_geometry:{x:50,y:60,width:800,height:600,screen:0},whiteboard_visible:true});
 assert.deepEqual(h.whiteboard.bounds,{x:50,y:60,width:800,height:600},'saved: the backend places it again');
});

test('full-view blur is a preference of the full chat; Windows alone draws acrylic',()=>{
 for(const platform of ['win32','darwin','linux']){
  const h=setup({platform});h.whiteboard.setBackgroundMaterial=h.control.setBackgroundMaterial;
  const materials=()=>h.calls(h.control,'setBackgroundMaterial').map(e=>e.args[0]);
  assert.deepEqual(h.windows.material('control',true),{supported:platform==='win32',blurred:platform==='win32'},platform);
  if(platform==='win32')assert.equal(materials().at(-1),'acrylic');else assert.deepEqual(materials(),[]);
  h.windows.setControlMode('compact',undefined,true);
  assert.deepEqual(h.windows.material('control',false),{supported:platform==='win32',blurred:false});
  h.windows.setControlMode('full',undefined,true);
  assert.equal(h.windows.material('control',true).blurred,platform==='win32','turning blur off in a small mode keeps the full view preference');
  assert.equal(h.windows.material('whiteboard',true).blurred,false,'the whiteboard stays transparent');
  assert.throws(()=>h.windows.material('control','yes'),/Invalid material preference/);
 }
});

test('a window gesture follows the cursor every 16 ms from where it began and owns the dock until it ends',()=>{
 const h=docked(setup());h.state.cursor={x:1840,y:1000};
 h.windows.windowGesture('control',{id:'g1',phase:'begin',kind:'resize'});
 assert.equal(h.control.ignoring,false,'the dock takes the mouse while dragged');
 h.windows.compactInteractive(false);assert.equal(h.control.ignoring,false,'hit tests cannot make a dragged dock click-through');
 h.state.cursor={x:1928,y:1000};h.timers.run(16);
 assert.deepEqual(h.control.bounds,{x:1744,y:847,width:176,height:176},'scaled about its lower left corner, kept on screen');
 assert.deepEqual(sent(h.control,'window-state').at(-1),{mode:'collapsed',dockWidth:176});
 h.state.cursor={x:4000,y:1000};h.timers.run(16);assert.equal(h.control.bounds.width,260,'the dock is at most 260 px');
 h.windows.windowGesture('control',{id:'other',phase:'end'});h.state.cursor={x:1900,y:1000};h.timers.run(16);
 assert.equal(h.control.bounds.width,148,'another gesture id cannot end it');
 h.windows.windowGesture('control',{id:'g1',phase:'end'});assert.equal(h.timers.pending,0);
 h.windows.compactInteractive(true);assert.equal(h.control.ignoring,false);h.windows.compactInteractive(false);assert.equal(h.control.ignoring,true);
 const native=()=>h.calls(h.control,'setIgnoreMouseEvents').length,before=native();
 h.windows.compactInteractive(false);h.windows.compactInteractive(false);assert.equal(native(),before,'an unchanged report makes no native call');
 h.windows.compactInteractive(true);assert.equal(native(),before+1);assert.equal(h.control.ignoring,false);h.windows.compactInteractive(false);
 // The whiteboard moves but never resizes; the full chat neither; oversized ids are ignored.
 for(const [name,value] of [['whiteboard',{id:'g',phase:'begin',kind:'resize'}],['control',{id:'x'.repeat(65),phase:'begin',kind:'move'}],['control',{id:'g',phase:'begin',kind:'spin'}]]){h.windows.windowGesture(name,value);assert.equal(h.timers.pending,0,JSON.stringify(value));}
 h.state.cursor={x:0,y:0};h.windows.windowGesture('whiteboard',{id:'m',phase:'begin',kind:'move'});h.state.cursor={x:30,y:40};h.timers.run(16);
 assert.deepEqual(h.whiteboard.bounds,{x:30,y:40,width:900,height:700});
 // A minute, or a destroyed window, ends it.
 h.timers.run(60000);assert.equal(h.timers.pending,0);
 h.windows.windowGesture('whiteboard',{id:'n',phase:'begin',kind:'move'});h.whiteboard.destroy();h.timers.run(16);assert.equal(h.timers.pending,0);
});

test('the mini chat and dock scale through the one size rule, which never outgrows the work area',()=>{
 const ultrawide={...PRIMARY,bounds:{x:0,y:0,width:5120,height:720},workArea:{x:0,y:0,width:5120,height:700}};
 const h=setup({displays:[ultrawide]});
 assert.throws(()=>h.windows.compactScale({width:300,height:300}),/Dock or compact chat required/);
 h.windows.setControlMode('compact',undefined,true);
 h.windows.compactScale({width:100,height:100});
 assert.deepEqual(compactSize('compact',ultrawide.workArea,100,100),{width:700,height:700});
 assert.equal(h.control.bounds.width,700);assert.equal(h.control.bounds.height,700,'a sixth of a 5120 px work area would be taller than it');
 h.windows.setControlMode('collapsed',undefined,true);h.windows.compactScale({width:500,height:10});
 assert.deepEqual([h.control.bounds.width,h.control.bounds.height],[260,260]);assert.deepEqual(sent(h.control,'window-state').at(-1),{mode:'collapsed',dockWidth:260});
 assert.equal(h.windows.chromeState('control').dockWidth,260);
 for(const size of [null,{width:'1',height:1},{width:1}])assert.throws(()=>h.windows.compactScale(size),/Dock or compact chat required/);
});

test('Windows hears the overlay button natively; macOS and Linux add no message hooks',()=>{
 const h=docked(setup({platform:'win32'}));h.state.cursor={x:50,y:60};
 assert.deepEqual(Object.keys(h.overlay.hooks).map(Number).sort(),[WM_LBUTTONDOWN,WM_LBUTTONUP]);assert.deepEqual(Object.keys(h.control.hooks).map(Number),[WM_LBUTTONUP]);
 h.overlay.hooks[WM_LBUTTONDOWN]();assert.equal(h.windows.pointer.buttons,1);
 h.overlay.bounds={...h.overlay.bounds,x:10,y:20};h.overlay.hooks[WM_LBUTTONUP]();
 assert.equal(h.windows.pointer.buttons,0);assert.deepEqual(sent(h.overlay,'overlay-pointer'),[{x:40,y:40,buttons:0,phase:'up'}]);
 h.windows.windowGesture('control',{id:'g',phase:'begin',kind:'move'});assert.equal(h.timers.pending,1);
 h.control.hooks[WM_LBUTTONUP]();assert.equal(h.timers.pending,0,'a released button ends the chat drag');
 for(const platform of ['darwin','linux']){const other=setup({platform});assert.deepEqual([other.overlay.hooks,other.control.hooks],[{},{}],platform);assert.deepEqual(other.calls(other.overlay,'hookWindowMessage'),[]);}
});

test('only Linux polls the cursor, forwarding it to click-through windows that nothing covers',()=>{
 const h=setup({platform:'linux'});h.overlay.emit('ready-to-show');h.state.cursor={x:100,y:100};
 h.timers.run(33);assert.deepEqual(sent(h.overlay,'forwarded-pointer'),[{x:100,y:100}]);
 h.whiteboard.visible=true;h.whiteboard.bounds={x:50,y:50,width:200,height:200};h.state.cursor={x:101,y:100};h.timers.run(33);
 assert.equal(sent(h.overlay,'forwarded-pointer').length,1,'the whiteboard covers the overlay there');
 docked(h);h.whiteboard.visible=false;h.state.cursor={x:1800,y:950};h.timers.run(33);
 assert.deepEqual(sent(h.control,'forwarded-pointer'),[{x:4,y:15}],'the click-through dock gets it');
 assert.deepEqual(sent(h.overlay,'forwarded-pointer').at(-1),{x:1800,y:950},'and, as it passes the mouse through, so does the overlay');
 h.windows.pointer.hover('avatar',true);h.windows.windowGesture('control',{id:'g',phase:'begin',kind:'move'});h.state.cursor={x:1801,y:950};h.timers.run(33);
 assert.equal(sent(h.control,'forwarded-pointer').length,1,'not while the dock is dragged');assert.equal(sent(h.overlay,'forwarded-pointer').length,2,'not while the overlay takes the mouse');
 h.windows.dispose();assert.equal(h.timers.pending,0,'quitting stops the poll');
 for(const platform of ['darwin','win32']){const other=setup({platform});other.overlay.emit('ready-to-show');assert.equal(other.timers.pending,0,platform);other.timers.run(1000);assert.deepEqual(sent(other.overlay,'forwarded-pointer'),[],platform);}
});

test('the overlay hears where the chat is and whether it shows',()=>{
 const h=docked(setup());h.control.bounds={...h.control.bounds,x:500};h.control.emit('move');
 assert.deepEqual(sent(h.overlay,'chat-surface').at(-1),{mode:'collapsed',bounds:{x:500,y:935,width:88,height:88},screen:PRIMARY.bounds,visible:true});
 h.control.visible=false;h.control.emit('hide');assert.equal(sent(h.overlay,'control-visibility').at(-1),false);
 h.windows.setControlMode('compact',undefined,true);h.control.emit('show');assert.equal(sent(h.overlay,'control-visibility').at(-1),true);
 assert.equal(h.windows.controlVisible(),true);h.control.minimized=true;assert.equal(h.windows.controlVisible(),false);
});

test('showing the controls opens a view in the full chat and wakes the dock as the mini chat',()=>{
 const h=docked(setup());
 h.windows.showControls('settings');assert.equal(h.windows.chatMode,'full');assert.deepEqual(sent(h.control,'navigate'),['settings']);
 h.windows.setControlMode('collapsed',undefined,true);h.windows.showControls('chat');assert.equal(h.windows.chatMode,'compact');
 h.windows.setControlMode('collapsed',undefined,true);h.windows.focus();assert.equal(h.windows.chatMode,'compact','a second launch wakes the dock');
 assert.deepEqual(h.windows.chromeAction('control','maximize',undefined,true),{maximized:true});assert.equal(h.windows.chatMode,'full');
 assert.deepEqual(h.windows.chromeAction('control','minimize',undefined,true),{mode:'compact'});
 assert.deepEqual(h.windows.chromeMode('whiteboard','collapsed',undefined,true),{mode:'collapsed'});
 assert.deepEqual(h.windows.chromeState('whiteboard'),{maximized:false,mode:'collapsed',dockWidth:88});
 assert.throws(()=>h.windows.chromeAction('whiteboard','quit'),/Unsupported window action/);
});

test('first run and the training-data window: one page each, focused rather than duplicated',async()=>{
 const h=setup({create:false});
 await h.windows.openSetup();const page=h.window('setup');assert.equal(page.options.backgroundColor,'#101116');
 h.windows.focus();assert.equal(h.calls(page,'focus').length,1,'a second launch focuses the setup page');
 h.windows.openNeuralData();h.windows.openNeuralData();
 const data=h.windows.get('neural-data');
 assert.equal(h.log.filter(e=>e.method==='new'&&e.target.page==='neural-data').length,1);assert.equal(h.calls(data,'focus').length,1);
});

test('Settings decides about its own unsaved changes: a reload asks, a quit already asked',()=>{
 const h=setup(),veto=()=>{const event={prevented:false,preventDefault(){this.prevented=true;}};h.control.webContents.emit('will-prevent-unload',event);return event.prevented;};
 h.settings.dirty=true;h.state.messageBox=1;assert.equal(veto(),false);assert.equal(h.settings.dirty,true,'keep editing');
 h.state.messageBox=0;assert.equal(veto(),true);assert.equal(h.settings.dirty,false,'discard');
 h.settings.dirty=true;h.module.app.isQuitting=true;const asked=h.calls(h.module.dialog,'showMessageBoxSync').length;
 assert.equal(veto(),true);assert.equal(h.calls(h.module.dialog,'showMessageBoxSync').length,asked,'no second question while quitting');
 h.settings.dirty=true;h.control.webContents.emit('render-process-gone');assert.equal(h.settings.dirty,false);
});

test('pages never navigate, and only web links leave for the browser',()=>{
 const h=setup();
 for(const window of [h.overlay,h.control,h.whiteboard,h.effects]){
  const event={prevented:false,preventDefault(){this.prevented=true;}};window.webContents.emit('will-navigate',event);assert.equal(event.prevented,true);
  for(const url of ['https://example.com/a','HTTP://example.com','file:///etc/passwd','javascript:alert(1)'])assert.deepEqual(window.webContents.openHandler({url}),{action:'deny'});
 }
 assert.deepEqual(h.opened,Array(4).fill(['https://example.com/a','HTTP://example.com']).flat());
 const window=new (fakeElectron().module.BrowserWindow)({});protectNavigation(window,()=>Promise.reject(new Error('no browser')));
 assert.deepEqual(window.webContents.openHandler({url:'https://example.com'}),{action:'deny'});
});
