import assert from 'node:assert/strict';
import {test} from 'node:test';
import {awarenessToggles,activeAppLabel,privacyNote,perceptionLine} from './initiative_model.mjs';

const mac={idle:{supported:true,reason:''},active_app:{supported:true,reason:'',fields:['app_name','bundle_id','process_id'],description:"the frontmost app's name, bundle ID and process ID"}};
const wayland={idle:{supported:true,reason:''},active_app:{supported:false,reason:'Wayland does not tell apps which window is focused',fields:['window_title','app_name','process_id']}};

test('toggles follow what the OS can observe, and say why one cannot be used',()=>{
  assert.deepEqual(awarenessToggles(mac),[{key:'observe_idle',label:'Allow computer idle/return awareness',disabled:false,reason:''},
    {key:'observe_active_app',label:'Allow foreground app name/process awareness',disabled:false,reason:''}]);
  const [idle,app]=awarenessToggles(wayland);
  assert.equal(idle.disabled,false);
  assert.deepEqual(app,{key:'observe_active_app',label:'Allow foreground window title/process awareness',disabled:true,reason:'Wayland does not tell apps which window is focused'});
  const none=awarenessToggles({idle:{supported:false,reason:'not available on freebsd14'},active_app:{supported:false,reason:'not available on freebsd14'}});
  assert.ok(none.every(toggle=>toggle.disabled&&toggle.reason==='not available on freebsd14'));
  assert.ok(awarenessToggles(undefined).every(toggle=>!toggle.disabled));  // an older backend without capabilities
  assert.equal(activeAppLabel({active_app:{fields:['window_title','process_id']}}),'Allow foreground window title/process awareness');
});

test('the privacy note names what is shared and the status comes from the last reading',()=>{
  assert.equal(privacyNote(mac),"Idle monitoring reads time since keyboard/mouse activity, not keys. App monitoring shares the frontmost app's name, bundle ID and process ID with your configured model provider. No screenshots are captured.");
  assert.match(privacyNote(wayland),/App monitoring is not available on this computer\./);
  assert.equal(perceptionLine({idle:'available',active_app:'unsupported'}),'Idle awareness: reading · App awareness: not available here');
  assert.equal(perceptionLine(undefined),'');
});
