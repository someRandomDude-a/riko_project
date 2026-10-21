import test from 'node:test';
import assert from 'node:assert/strict';
import {Vector3} from 'three';
import {AvatarInertia,withSpringForce} from './avatar_inertia.mjs';
const dt=1/60,run=(inertia,positions)=>{for(const [x,y] of positions)inertia.track(x,y,dt);return inertia;};
test('springs lag behind the window when it speeds up and swing ahead when it stops',()=>{
 const inertia=new AvatarInertia(),force=new Vector3();
 run(inertia,Array.from({length:10},(_,i)=>[i*i*2,0]));  // speeding up to the right
 assert.ok(inertia.force(.003,1,dt,force).x<0);
 run(inertia,Array.from({length:60},(_,i)=>[200+i*10,0]));  // a steady 600 px/s: only the air pushes back
 assert.ok(Math.abs(inertia.acceleration.x)<5);assert.ok(Math.abs(inertia.velocity.x-600)<5);assert.ok(inertia.force(.003,1,dt,force).x<0);
 run(inertia,Array.from({length:4},()=>[790,0]));  // stopped dead
 assert.ok(inertia.force(.003,1,dt,force).x>0);
 const lift=run(new AvatarInertia(),Array.from({length:10},(_,i)=>[0,-i*i*2]));  // speeding up towards the top of the screen
 assert.ok(lift.force(.003,1,dt,force).y<0,'lifting her pulls her springs down');
 assert.equal(lift.force(.003,0,dt,force).length(),0);  // bounce 0 is off
});
test('a jump to another place is not motion',()=>{
 const inertia=run(new AvatarInertia(),[[0,0],[10,0],[20,0],[5000,900]]);
 assert.deepEqual([inertia.velocity,inertia.acceleration],[{x:0,y:0},{x:0,y:0}]);
 run(inertia,[[5010,900]]);assert.ok(inertia.velocity.x>0&&inertia.velocity.x<600);
});
test('the force is added to each joint\'s gravity for one update, then their own gravity comes back',()=>{
 const joints=[{settings:{gravityPower:.2,gravityDir:new Vector3(0,-1,0)}},{settings:{gravityPower:0,gravityDir:new Vector3(0,-1,0)}}],seen=[];
 withSpringForce(joints,new Vector3(.1,0,0),()=>{for(const {settings} of joints)seen.push(settings.gravityDir.clone().multiplyScalar(settings.gravityPower).toArray().map(v=>+v.toFixed(6)));});
 assert.deepEqual(seen,[[.1,-.2,0],[.1,0,0]]);
 assert.deepEqual(joints.map(({settings})=>[settings.gravityPower,settings.gravityDir.toArray()]),[[.2,[0,-1,0]],[0,[0,-1,0]]]);
 assert.throws(()=>withSpringForce(joints,new Vector3(0,.3,0),()=>{throw new Error('update failed');}),/update failed/);
 assert.deepEqual(joints.map(({settings})=>settings.gravityPower),[.2,0]);
 let calls=0;withSpringForce(joints,new Vector3(),()=>calls++);assert.equal(calls,1);
});
test('a resize about a fixed point is not motion: shifting the tracked position leaves velocity and acceleration as they were',()=>{
 const inertia=run(new AvatarInertia(),Array.from({length:30},(_,i)=>[i*5,0])),velocity={...inertia.velocity};
 inertia.shift(-40,-60);run(inertia,[[30*5-40,-60]]);
 assert.ok(Math.abs(inertia.velocity.x-velocity.x)<1&&Math.abs(inertia.velocity.y)<1e-9,`velocity ${JSON.stringify(inertia.velocity)}`);
 assert.ok(Math.abs(inertia.acceleration.y)<1e-9);
 const fresh=new AvatarInertia();fresh.shift(10,10);assert.equal(fresh.position,null,'nothing to shift before the first sample');
});
