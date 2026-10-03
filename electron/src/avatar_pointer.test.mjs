import test from 'node:test';
import assert from 'node:assert/strict';
import {pointerSampler} from './avatar_pointer.mjs';
test('hover coalesces duplicate mouse/pointer events and bounds expensive picking',()=>{
 const sampler=pointerSampler(80);
 sampler.queue({clientX:1,clientY:2});sampler.queue({clientX:3,clientY:4});
 assert.deepEqual(sampler.take(0),{x:3,y:4});
 sampler.queue({clientX:5,clientY:6});assert.equal(sampler.take(16),null);
 assert.deepEqual(sampler.take(80),{x:5,y:6});assert.equal(sampler.take(160),null);
});
test('drag can consume every frame and clearing prevents stale movement after release',()=>{
 const sampler=pointerSampler();
 sampler.queue({clientX:1,clientY:2});sampler.take(0);
 sampler.queue({clientX:3,clientY:4});assert.deepEqual(sampler.take(16,true),{x:3,y:4});
 sampler.queue({clientX:8,clientY:9});sampler.clear();assert.equal(sampler.take(32,true),null);
});
