import test from 'node:test';
import assert from 'node:assert/strict';
import {pointerSampler,pointerReporter} from './avatar_pointer.mjs';
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
test('the backend hears about the pointer when it comes near or goes, and once a second while it keeps moving near',()=>{
 const reports=pointerReporter(1000),far={x:1.9,y:0,near:false},near={x:.2,y:.1,near:true};
 assert.deepEqual([0,100,200].map(t=>reports.due(far,t)),[false,false,false],'moving far away costs nothing');
 assert.equal(reports.due(near,300),true,'it came near');
 assert.deepEqual([400,1200,1299].map(t=>reports.due(near,t)),[false,false,false]);
 assert.equal(reports.due(near,1300),true,'a keep-alive inside the backend\'s 2 s');
 assert.equal(reports.due(far,1350),true,'it left');assert.equal(reports.due(far,5000),false);
 assert.equal(reports.due(null,5100),false,'no pointer is far');
 reports.due(near,6000);reports.reset();assert.equal(reports.due(near,6001),true,'after a leave the next near report goes out');
});
