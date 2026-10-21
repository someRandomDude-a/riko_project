import {fakeTimers} from './electron_fakes.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {gestureBounds,gestureLoop}=createRequire(import.meta.url)('../window_gesture.cjs');
const area={x:0,y:0,width:1920,height:1080};
const g={kind:'resize',mode:'compact',bounds:{x:300,y:300,width:480,height:600},cursor:{x:780,y:300}};
test('resize is absolute from the start and preserves the lower left anchor',()=>{
 const cursor={x:830,y:250},b=gestureBounds(g,cursor,area);
 assert.deepEqual(b,{x:300,y:250,width:530,height:650});
 for(let i=0;i<100;i++)assert.deepEqual(gestureBounds(g,cursor,area),b);
 assert.deepEqual(gestureBounds(g,g.cursor,area),g.bounds);
});
test('moving never resizes and follows the native cursor after missed frames',()=>{
 const moving={...g,kind:'move'};
 assert.deepEqual(gestureBounds(moving,{x:1080,y:500},area),{x:600,y:480,width:480,height:600});
});
test('collapsed corner scales proportionally along either axis',()=>{
 const dock={...g,mode:'collapsed',bounds:{x:300,y:600,width:88,height:100}};
 assert.deepEqual(gestureBounds(dock,{x:868,y:300},area),{x:300,y:524,width:176,height:176});
 assert.deepEqual(gestureBounds(dock,{x:780,y:200},area),{x:300,y:524,width:176,height:176});
});
test('the gesture loop follows the cursor every 16 ms until ended, replaced, a minute old or the window is gone',()=>{
 const timers=fakeTimers();let cursor={x:0,y:0};const applied=[];
 const screen={getCursorScreenPoint:()=>cursor,getDisplayMatching:()=>({workArea:area})};
 const window=(bounds,destroyed=false)=>({getBounds:()=>({...bounds}),setBounds:next=>{bounds=next;applied.push(next);},isDestroyed:()=>destroyed,destroy(){destroyed=true;}});
 const loop=gestureLoop({screen,setTimer:timers.set,clearTimer:timers.clear,now:timers.now});
 const w=window({x:10,y:10,width:480,height:600}),seen=[];
 const g=loop.begin({id:'a',window:w,kind:'move',mode:'compact',onBounds:(next,gesture)=>seen.push([next.x,gesture.id])});
 assert.equal(loop.active,g);assert.deepEqual(applied,[],'an unmoved cursor applies nothing');
 cursor={x:5,y:7};timers.run(16);assert.deepEqual(applied,[{x:15,y:17,width:480,height:600}]);assert.deepEqual(seen,[[15,'a']]);
 timers.run(16);assert.equal(applied.length,1,'unchanged bounds are not reapplied');
 loop.endIf('b',w);loop.endIf('a',window({}));assert.equal(loop.active,g,'only its own id and window end it');
 loop.endIf('a',w);assert.equal(loop.active,null);assert.equal(timers.pending,0);
 const other=window({x:0,y:0,width:100,height:100});loop.begin({id:'c',window:other,kind:'move'});loop.begin({id:'d',window:other,kind:'move'});
 assert.equal(loop.active.id,'d');assert.equal(timers.pending,1,'a new gesture replaces the old one');
 timers.run(60016);assert.equal(loop.active,null,'a minute at most');
 loop.begin({id:'e',window:other,kind:'move'});other.destroy();timers.run(16);assert.equal(loop.active,null);assert.equal(timers.pending,0);
});
