import test from 'node:test';
import assert from 'node:assert/strict';
import {share,shallowEqual,createRuntimeStore,pick} from './runtime_store.mjs';
import {mergeHistory} from './chat_history.mjs';
import {initialVoice,reduceVoice} from './voice_state.mjs';

test('structural sharing keeps every unchanged part of a new snapshot as the previous object',()=>{
 const before={avatar:{model:'a.vrm',camera:{fov:30}},actions:[{id:1,status:'running'},{id:2,status:'done'}],emotion:{primary:'joy'},n:1};
 const after=share(before,JSON.parse(JSON.stringify({...before,emotion:{primary:'calm'},actions:[{id:1,status:'running'},{id:2,status:'failed'}]})));
 assert.notEqual(after,before);assert.equal(after.avatar,before.avatar);assert.equal(after.actions[0],before.actions[0]);
 assert.notEqual(after.actions[1],before.actions[1]);assert.notEqual(after.emotion,before.emotion);assert.equal(after.n,1);
 assert.equal(share(before,JSON.parse(JSON.stringify(before))),before,'equal throughout: the previous snapshot itself');
 for(const [a,b] of [[{x:1},{x:1,y:2}],[{x:1,y:2},{x:1}],[[1,2],[1]],[[1],{0:1}],[{x:null},{x:{}}],[{x:NaN},{x:0}]])assert.notEqual(share(a,b),a,JSON.stringify([a,b]));
 assert.equal(share({x:NaN},{x:NaN}).x,NaN);
});

test('shallow equality compares keys by reference, or values',()=>{
 const o={};assert.ok(shallowEqual({a:o,b:1},{a:o,b:1}));assert.ok(!shallowEqual({a:{},b:1},{a:{},b:1}));
 assert.ok(!shallowEqual({a:1},{a:1,b:undefined}));assert.ok(shallowEqual('x','x'));assert.ok(!shallowEqual([],{}));assert.ok(shallowEqual(null,null));
});

test('the store opens the socket with its first listener, notifies only changed snapshots, and closes with the last',()=>{
 let deliver=null,opened=0,closed=0;
 const store=createRuntimeStore(onEvent=>{opened++;deliver=onEvent;return()=>{closed++;};});
 const heard=[];const off=store.subscribe(()=>heard.push(store.get()));const off2=store.subscribe(()=>{});assert.equal(opened,1);
 deliver({type:'state.snapshot',payload:{a:{b:1},c:2}});const first=store.get();assert.deepEqual(first,{a:{b:1},c:2});
 deliver({type:'state.snapshot',payload:{a:{b:1},c:2}});deliver({type:'chat.delta',payload:{text:'x'}});
 assert.equal(heard.length,1,'an identical snapshot and other events are not heard');
 deliver({type:'state.snapshot',payload:{a:{b:1},c:3}});assert.equal(heard.length,2);assert.equal(store.get().a,first.a);
 assert.deepEqual(pick(['c','missing'])(store.get()),{c:3,missing:undefined});
 off();assert.equal(closed,0);off2();assert.equal(closed,1);store.subscribe(()=>{});assert.equal(opened,2);
});

test('a history page that repeats a message unchanged keeps its object; a changed one is new',()=>{
 const current=[{id:'a',role:'user',text:'hi',sequence:1,event_sequence:3},{id:'b',role:'assistant',text:'yo',sequence:2,metrics:{tokens_per_second:40}}];
 const merged=mergeHistory(current,JSON.parse(JSON.stringify([{id:'a',role:'user',text:'hi',sequence:1},{id:'b',role:'assistant',text:'yo!',sequence:2,metrics:{tokens_per_second:40}}])));
 assert.equal(merged[0],current[0]);assert.notEqual(merged[1],current[1]);assert.equal(merged[1].text,'yo!');assert.equal(merged[1].metrics,current[1].metrics);
});

test('voice state keeps its object for repeated snapshots, voice resources and silent levels',()=>{
 const snap={type:'state.snapshot',payload:{runtime:{listening:true,microphone_status:'ready',voice_phase:'waiting',wake:{active:false,mode:'wake'},latest_transcript:{text:'hi',final:true}}}};
 const state=reduceVoice(initialVoice,snap);assert.equal(reduceVoice(state,JSON.parse(JSON.stringify(snap))),state);
 const resource={type:'resource.voice',payload:{capture_running:true,microphone_status:'ready',phase:'waiting',wake:{active:false,mode:'wake'},latest_transcript:{text:'hi',final:true}}};
 assert.equal(reduceVoice(state,resource),state);
 const off=reduceVoice(initialVoice,{type:'voice.stopped'});assert.equal(reduceVoice(off,{type:'voice.level',payload:{rms:.3}}),off,'no level while the microphone is off');
 assert.notEqual(reduceVoice(state,{type:'voice.level',payload:{rms:.05}}),state);
});
