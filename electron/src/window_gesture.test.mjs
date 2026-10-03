import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {gestureBounds}=createRequire(import.meta.url)('../window_gesture.cjs');
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
