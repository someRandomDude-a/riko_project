import test from 'node:test';
import assert from 'node:assert/strict';
import {scheduleBoardCapture} from './whiteboard_capture.mjs';

test('board capture is scheduled once per mutation and sends only the board rectangle',async()=>{
  let scheduled,sent,rectangle;
  const rect={x:0,y:42,width:900,height:650};
  scheduleBoardCapture({revision:'revision',rect:()=>rect,capture:async value=>{rectangle=value;return new Uint8Array([1,2]);},send:async(...value)=>{sent=value;},schedule:callback=>{scheduled=callback;}});
  await scheduled();
  assert.deepEqual(rectangle,rect);assert.equal(sent[0],'revision');
});

test('superseded captures never upload stale images',async()=>{
  let scheduled,resolve,uploaded=false;
  const stop=scheduleBoardCapture({revision:'old',rect:()=>({}),capture:()=>new Promise(done=>{resolve=done;}),send:()=>{uploaded=true;},schedule:callback=>{scheduled=callback;},unschedule:()=>{}});
  const pending=scheduled();await Promise.resolve();stop();resolve(new Uint8Array([1]));await pending;
  assert.equal(uploaded,false);
});
