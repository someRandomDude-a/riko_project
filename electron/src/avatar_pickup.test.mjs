import test from 'node:test';
import assert from 'node:assert/strict';
import {Group,Bone,Vector3} from 'three';
import {AvatarPickup} from './avatar_pickup.mjs';
import {boneCatalog} from './avatar_secondary.mjs';
import {normalizeProfile} from './avatar_studio_settings.mjs';
test('feet pickup keeps its anchor, swings under gravity, triggers ragdoll once and recovers',()=>{
 const scene=new Group(),foot=new Bone(),hips=new Bone(),head=new Bone();foot.name='foot';hips.position.y=1;head.position.y=1;scene.add(foot);foot.add(hips);hips.add(head);
 const vrm={scene,humanoid:{getRawBoneNode:()=>foot}},events=[],pickup=new AvatarPickup(vrm,boneCatalog(vrm,'fixture'),kind=>events.push(kind)),settings=normalizeProfile({}).pickup;
 pickup.start('leftFoot',{x:0,y:0});pickup.move({x:500,y:0},1,settings);pickup.move({x:600,y:0},1,settings);assert.deepEqual(events,['ragdoll']);
 for(let i=0;i<300;i++){pickup.restore();pickup.step(settings,1/60);assert.ok(foot.getWorldPosition(new Vector3()).distanceTo(pickup.anchor)<1e-6);assert.ok(Number.isFinite(pickup.angle));}
 assert.ok(Math.abs(pickup.angle)>1);pickup.release();assert.deepEqual(events,['ragdoll','recover']);
 for(let i=0;i<200;i++){pickup.restore();pickup.step(settings,1/60);}assert.equal(pickup.active,false);assert.ok(scene.position.length()<1e-6);
});
test('ragdoll thresholds and disable switch are respected',()=>{
 const scene=new Group(),bone=new Bone();scene.add(bone);const vrm={scene,humanoid:{getRawBoneNode:()=>bone}},events=[],pickup=new AvatarPickup(vrm,boneCatalog(vrm,'fixture'),e=>events.push(e));
 const settings=normalizeProfile({pickup:{distanceThreshold:1000,speedThreshold:10000}}).pickup;
 pickup.start('head',{x:0,y:0});pickup.move({x:100,y:0},1,settings);assert.deepEqual(events,[]);
 pickup.move({x:2000,y:0},1,{...settings,ragdollEnabled:false});assert.deepEqual(events,[]);
 pickup.step({...settings,enabled:false},.1);assert.equal(pickup.active,false);
});
