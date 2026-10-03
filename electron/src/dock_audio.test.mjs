import test from 'node:test';
import assert from 'node:assert/strict';
import {audioPopover,volumeWriter} from './dock_audio.mjs';
test('hover reveals without muting, first click arms and further clicks toggle',()=>{
 let state={open:false,armed:false};state=audioPopover(state,'reveal');assert.equal(state.open,true);assert.equal(state.toggle,undefined);
 state=audioPopover(state,'click');assert.equal(state.toggle,false);
 state=audioPopover(state,'click');assert.equal(state.toggle,true);
 assert.equal(audioPopover(state,'reveal').toggle,undefined);
 state=audioPopover(state,'click');assert.equal(state.toggle,true);
 state=audioPopover(state,'close');assert.deepEqual(state,{open:false,armed:false});assert.equal(audioPopover(state,'click').toggle,false);
});
test('rapid gain edits serialize and keep only the latest waiting value',async()=>{
 const values=[],resolve=[];const writer=volumeWriter(value=>{values.push(value);return new Promise(r=>resolve.push(r));});
 writer.set(.1);writer.set(.2);writer.set(.8);assert.deepEqual(values,[.1]);assert.equal(writer.pending,true);
 resolve.shift()();await Promise.resolve();await Promise.resolve();assert.deepEqual(values,[.1,.8]);
 resolve.shift()();await Promise.resolve();await Promise.resolve();assert.equal(writer.pending,false);writer.close();writer.set(.9);assert.deepEqual(values,[.1,.8]);
});
test('failed gain writes report errors and still process newer edits',async()=>{
 const values=[],errors=[];const writer=volumeWriter(async value=>{values.push(value);if(value===.1)throw new Error('offline');},e=>errors.push(e.message));
 writer.set(.1);writer.set(.5);await Promise.resolve();await Promise.resolve();await Promise.resolve();assert.deepEqual(values,[.1,.5]);assert.deepEqual(errors,['offline']);writer.close();
});
