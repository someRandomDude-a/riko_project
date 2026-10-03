import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {PoseInterpolator,MotionEngine,proceduralPose} from './motion_engine.mjs';
import {DesktopWalker,nearbyPointer} from './motion_geometry.mjs';
import {MotionAssetSampler} from './motion_assets.mjs';

test('interrupted transition starts at current displayed pose',()=>{
  const q=new THREE.Quaternion(),target=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),1);
  const blend=new PoseInterpolator(new Map([['head',q]]));
  blend.step(new Map([['head',target]]),'first',.1,.4);
  const before=blend.displayed.get('head').clone();
  const after=blend.step(new Map([['head',q]]),'second',0,.4).get('head');
  assert.ok(before.angleTo(after)<1e-6);
  for(let i=0;i<10;i++)blend.step(new Map([['head',q]]),'second',.1,.4);
  assert.ok(blend.displayed.get('head').angleTo(q)<1e-6);
});

test('walker clamps bounds, completes once, and hold cancels',()=>{
  const events=[],walker=new DesktopWalker((...args)=>events.push(args));
  const geometry={x:0,y:0,width:100,height:100,screen:0};
  const action={id:'walk',payload:{target:{x:2000,y:0},speed:240}};
  let result;
  for(let i=0;i<100;i++){const next=walker.step(action,geometry,{width:400,height:400},.1);if(next)result=next;}
  assert.equal(result.x,300);assert.equal(result.final,true);
  assert.equal(events.filter(e=>e[1]==='completed').length,1);
  walker.step({...action,id:'second'},geometry,{width:400,height:400},.1);
  walker.step({...action,id:'second'},geometry,{width:400,height:400},.1,true);
  assert.equal(walker.active,null);assert.equal(events.at(-1)[1],'cancelled');
});

test('pointer hysteresis avoids flickering at boundary',()=>{
  const rect={x:0,y:0,width:100,height:100};
  assert.equal(nearbyPointer(rect,{x:125,y:50},false).near,false);
  assert.equal(nearbyPointer(rect,{x:125,y:50},true).near,true);
});

test('procedural held and walking poses differ from idle',()=>{
  assert.ok(proceduralPose('held',1).get('leftUpperArm').angleTo(proceduralPose('idle',1).get('leftUpperArm'))>.1);
  assert.ok(proceduralPose('walking',1).get('leftUpperLeg').angleTo(proceduralPose('idle',1).get('leftUpperLeg'))>.1);
});

test('pose sampling respects masks and reports once',()=>{
  const sampler=new MotionAssetSampler({}),events=[];
  const asset={id:'pose',kind:'pose',mask:['head'],pose:{bones:{head:[.1,0,0],neck:[0,.1,0]},expressions:{happy:.5}}};
  const sample=sampler.sample(asset,0,'action',(...args)=>events.push(args));
  assert.deepEqual([...sample.bones.keys()],['head']);assert.equal(sample.expressions.get('happy'),.5);
  sampler.retain(new Set(['pose']));sampler.sample(asset,1,'action',(...args)=>events.push(args));
  assert.equal(events.length,1);sampler.dispose();
});

test('held authority suppresses imported skeletal overlays',()=>{
  const head=new THREE.Object3D(),arm=new THREE.Object3D();
  const vrm={humanoid:{getNormalizedBoneNode:name=>name==='head'?head:name==='leftUpperArm'?arm:null}};
  let sampled=0;
  const engine=new MotionEngine(vrm,{sample:()=>{sampled++;return null;},retain:()=>{},dispose:()=>{}});
  engine.update({actions:[{id:'base',kind:'motion.base',status:'running',payload:{procedural:'held',asset:{id:'clip'}}}],interaction:{held:true}},.1);
  assert.equal(sampled,0);engine.dispose();
});

test('late asset load is disposed and cannot become active',async()=>{
  let resolve;
  const sampler=new MotionAssetSampler({},{load:()=>new Promise(done=>{resolve=done;})});
  sampler.sample({id:'clip',kind:'vrma'},0,'action',()=>assert.fail('stale report'));
  await Promise.resolve();sampler.dispose();
  resolve({scene:new THREE.Scene(),userData:{}});
  await new Promise(done=>setImmediate(done));
  assert.equal(sampler.cache.size,0);
});

test('VRMA sampling retargets rotations, excludes translations and restores live pose',async()=>{
  const scene=new THREE.Scene(),head=new THREE.Object3D();head.name='Head';scene.add(head);
  const animation={duration:1,humanoidTracks:{rotation:new Map([['head',new THREE.QuaternionKeyframeTrack('head.quaternion',[0,1],[0,0,0,1,0,Math.sin(.5),0,Math.cos(.5)])]]),translation:new Map()},expressionTracks:{preset:new Map(),custom:new Map()},lookAtTrack:null};
  const vrm={scene,meta:{metaVersion:'1'},humanoid:{getNormalizedBoneNode:name=>name==='head'?head:null},expressionManager:{getExpression:()=>null},lookAt:null};
  const sampler=new MotionAssetSampler(vrm,{load:async()=>({scene:new THREE.Scene(),userData:{vrmAnimations:[animation]}})});
  const asset={id:'vrma',kind:'vrma',mask:['head'],loop:false,speed:1};
  const events=[];
  sampler.sample(asset,0,'action',(...args)=>events.push(args));
  await new Promise(done=>setImmediate(done));
  const pose=sampler.sample(asset,.5,'action',(...args)=>events.push(args));
  assert.ok(pose?.bones.get('head').angleTo(new THREE.Quaternion())>.4);
  assert.ok(head.quaternion.angleTo(new THREE.Quaternion())<1e-6);
  sampler.sample(asset,1.1,'action',(...args)=>events.push(args));
  sampler.sample(asset,1.2,'action',(...args)=>events.push(args));
  assert.equal(events.filter(e=>e[0]==='started').length,1);
  assert.equal(events.filter(e=>e[0]==='completed').length,1);
  sampler.dispose();
});
