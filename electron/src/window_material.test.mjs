import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {borderActive,normalizeFeedback} from './feedback_model.mjs';
const {applyWindowMaterial}=createRequire(import.meta.url)('../window_material.cjs');
test('full-view blur clears material and restores alpha in either small mode',()=>{
 const calls=[],w={setBackgroundColor:v=>calls.push(v),setBackgroundMaterial:v=>calls.push(v)};
 assert.equal(applyWindowMaterial(w,'full',true,'win32').blurred,true);
 assert.equal(calls.at(-1),'acrylic');
 for(const mode of ['compact','collapsed','transparent']){
  calls.length=0;assert.equal(applyWindowMaterial(w,mode,true,'win32').blurred,false);
  assert.deepEqual(calls,['#00000000','none','#00000000']);
 }
});
test('disabled or unsupported full-view blur keeps the native surface transparent',()=>{
 const calls=[],w={setBackgroundColor:v=>calls.push(v),setBackgroundMaterial:()=>{throw Error('unsupported');}};
 assert.equal(applyWindowMaterial(w,'full',true,'win32').supported,false);
 assert.equal(calls.at(-1),'#00000000');
 assert.equal(applyWindowMaterial(w,'full',false,'win32').blurred,false);
 assert.equal(applyWindowMaterial(w,'full',true,'linux').blurred,false);
});
test('transcription border settings are bounded and independent from listening',()=>{
 const p=normalizeFeedback({listenBorder:false,transcribingBorder:true,transcribingBorderStyle:'solid',transcribingBorderAnchor:'screen',transcribingBorderColor:'#112233',transcribingBorderWidth:99});
 assert.equal(p.transcribingBorderWidth,12);assert.equal(p.transcribingBorderColor,'#112233');
 assert.equal(borderActive({enabled:true,phase:'transcribing'},p),true);
 assert.equal(borderActive({enabled:true,phase:'capturing'},p),false);
 assert.equal(borderActive({enabled:true,phase:'transcribing'},{...p,listenBorder:true,transcribingBorder:false}),false);
 assert.equal(borderActive({enabled:false,phase:'transcribing'},p),false);
 const bad=normalizeFeedback({transcribingBorderWidth:NaN,transcribingBorderStyle:'invalid',transcribingBorderColor:'bad'});
 assert.equal(bad.transcribingBorderWidth,3);assert.equal(bad.transcribingBorderStyle,'rainbow');
});
