import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
test('status consumers and Electron host have no periodic polling timers',()=>{
  for(const file of ['animation_library.jsx','initiative_settings.jsx','voice_input.jsx','tool_approvals.jsx','gpu_resources.jsx','stream_chat.jsx','../main.cjs']){
    const path=file==='../main.cjs'?new URL('../main.cjs',import.meta.url):new URL(file,import.meta.url);
    assert.doesNotMatch(readFileSync(path,'utf8'),/\bsetInterval\s*\(/,file);
  }
});
