import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,readdirSync} from 'node:fs';
// A source guard, as no test can prove a timer is absent. The one main-process cursor poll (Linux, standing in for the
// click-through forwarding it lacks) is a setTimeout chain whose behaviour window_manager.test.mjs checks per platform.
test('status consumers and every main-process module have no periodic polling timers',()=>{
  const host=readdirSync(new URL('..',import.meta.url)).filter(name=>name.endsWith('.cjs')).map(name=>'../'+name);
  assert.ok(host.includes('../main.cjs')&&host.includes('../window_manager.cjs'));
  for(const file of ['animation_library.jsx','initiative_settings.jsx','voice_input.jsx','tool_approvals.jsx','gpu_resources.jsx','stream_chat.jsx',...host]){
    assert.doesNotMatch(readFileSync(new URL(file,import.meta.url),'utf8'),/\bsetInterval\s*\(/,file);
  }
});
