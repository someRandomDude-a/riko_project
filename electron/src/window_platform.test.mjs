import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {windowPlatform,x11Relaunch,WM_LBUTTONDOWN,WM_LBUTTONUP}=createRequire(import.meta.url)('../window_platform.cjs');
const hooked=()=>{const hooks={};return {hooks,hookWindowMessage:(message,fn)=>{hooks[message]=fn;},setBackgroundColor(){},setBackgroundMaterial(value){this.material=value;}};};
test('Windows hooks the left button; only Linux forwards the cursor; only Windows reports native buttons',()=>{
 assert.deepEqual([WM_LBUTTONDOWN,WM_LBUTTONUP],[0x0201,0x0202]);
 for(const [platform,expected] of [['win32',{nativeButtons:true,forwardsPointer:false,overlay:[WM_LBUTTONDOWN,WM_LBUTTONUP],chat:[WM_LBUTTONUP]}],['darwin',{nativeButtons:false,forwardsPointer:false,overlay:[],chat:[]}],['linux',{nativeButtons:false,forwardsPointer:true,overlay:[],chat:[]}]]){
  const os=windowPlatform(platform),overlay=hooked(),chat=hooked(),press=()=>'press',release=()=>'release';
  os.hookOverlay(overlay,{press,release});os.hookChat(chat,release);
  assert.deepEqual({nativeButtons:os.nativeButtons,forwardsPointer:os.forwardsPointer,overlay:Object.keys(overlay.hooks).map(Number),chat:Object.keys(chat.hooks).map(Number)},expected,platform);
  if(platform==='win32')assert.deepEqual([overlay.hooks[WM_LBUTTONDOWN](),overlay.hooks[WM_LBUTTONUP](),chat.hooks[WM_LBUTTONUP]()],['press','release','release']);
  assert.ok(Object.isFrozen(os));
 }
});
test('acrylic is a Windows material; elsewhere the full chat stays transparent',()=>{
 for(const platform of ['win32','darwin','linux']){const window=hooked();assert.equal(windowPlatform(platform).material(window,'full',true).blurred,platform==='win32',platform);assert.equal(window.material,platform==='win32'?'acrylic':undefined);}
});
test('only a Linux Wayland launch without an Ozone platform relaunches, keeping its arguments and the AppImage',()=>{
 const hasSwitch=()=>false,argv=['/opt/riko','.','--trace'];
 assert.deepEqual(x11Relaunch({platform:'linux',env:{WAYLAND_DISPLAY:'wayland-0',APPIMAGE:'/home/me/Riko.AppImage'},argv,hasSwitch}),{execPath:'/home/me/Riko.AppImage',args:['.','--trace','--ozone-platform=x11']});
 assert.deepEqual(x11Relaunch({platform:'linux',env:{WAYLAND_DISPLAY:'wayland-0'},argv,hasSwitch}),{args:['.','--trace','--ozone-platform=x11']});
 assert.equal(x11Relaunch({platform:'linux',env:{WAYLAND_DISPLAY:'wayland-0'},argv,hasSwitch:name=>name==='ozone-platform'}),null);
 assert.equal(x11Relaunch({platform:'linux',env:{},argv,hasSwitch}),null);
 assert.equal(x11Relaunch({platform:'darwin',env:{WAYLAND_DISPLAY:'wayland-0'},argv,hasSwitch:()=>assert.fail('not asked')}),null);
});
