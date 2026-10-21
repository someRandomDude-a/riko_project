import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {PoseInterpolator,MotionEngine,proceduralPose,gesturePose,flipVRM0,BODY_BONES} from './motion_engine.mjs';
import {DesktopWalker,nearbyPointer} from './motion_geometry.mjs';
import {MotionAssetSampler} from './motion_assets.mjs';
import {heldPose,settlePose} from './avatar_held_pose.mjs';
import {HoldBehaviour} from './avatar_hold.mjs';

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

test('the held pose (the plain dangle) and walking differ from idle; held, settling and recovering are no longer procedural poses',()=>{
  assert.ok(heldPose(null,1).get('leftUpperArm').angleTo(proceduralPose('idle',1).get('leftUpperArm'))>.1);
  assert.ok(heldPose(null,1).get('hips').angleTo(new THREE.Quaternion())>.4,'the scruff tilt');
  assert.ok(proceduralPose('walking',1).get('leftUpperLeg').angleTo(proceduralPose('idle',1).get('leftUpperLeg'))>.1);
  for(const mode of ['held','settling','recovering'])for(const [bone,q] of proceduralPose(mode,1))assert.ok(q.equals(proceduralPose('idle',1).get(bone)),`${mode} ${bone}`);
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
  await new Promise(done=>setTimeout(done));
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
  await new Promise(done=>setTimeout(done));
  const pose=sampler.sample(asset,.5,'action',(...args)=>events.push(args));
  assert.ok(pose?.bones.get('head').angleTo(new THREE.Quaternion())>.4);
  assert.ok(head.quaternion.angleTo(new THREE.Quaternion())<1e-6);
  sampler.sample(asset,1.1,'action',(...args)=>events.push(args));
  sampler.sample(asset,1.2,'action',(...args)=>events.push(args));
  assert.equal(events.filter(e=>e[0]==='started').length,1);
  assert.equal(events.filter(e=>e[0]==='completed').length,1);
  sampler.dispose();
});

// A two-bone arm per side. VRM 1.0 faces +z, so her left arm points to +x; a VRM 0.x model faces -z and its left arm points to -x.
function armRig(version){
  const scene=new THREE.Group(),nodes=new Map(BODY_BONES.map(name=>[name,new THREE.Object3D()])),side=version==='0'?-1:1;
  for(const [arm,lower,sign] of [['leftUpperArm','leftLowerArm',side],['rightUpperArm','rightLowerArm',-side]]){nodes.get(lower).position.x=.25*sign;nodes.get(arm).add(nodes.get(lower));scene.add(nodes.get(arm));}
  return {scene,nodes,vrm:{scene,meta:{metaVersion:version},humanoid:{getNormalizedBoneNode:name=>nodes.get(name)||null}}};
}
const still={sample:()=>null,retain:()=>{},dispose:()=>{}};
const armDirection=(nodes,arm,lower)=>{nodes.get(arm).updateMatrixWorld(true);return nodes.get(lower).getWorldPosition(new THREE.Vector3()).sub(nodes.get(arm).getWorldPosition(new THREE.Vector3())).normalize();};

test('canonical poses are flipped once for VRM 0.x and written unchanged for VRM 1.0, so idle arms hang down on both',()=>{
  for(const version of ['0','1']){
    const {nodes,vrm}=armRig(version),engine=new MotionEngine(vrm,still);
    for(let i=0;i<20;i++)engine.update({},.05);
    for(const bone of ['leftUpperArm','rightUpperArm','head']){
      const canonical=proceduralPose('idle',engine.time).get(bone),expected=version==='0'?flipVRM0(canonical):canonical;
      assert.ok(nodes.get(bone).quaternion.angleTo(expected)<1e-6,`${version} ${bone}`);
    }
    for(const [arm,lower] of [['leftUpperArm','leftLowerArm'],['rightUpperArm','rightLowerArm']])assert.ok(armDirection(nodes,arm,lower).y<-.8,`VRM ${version} ${arm} is raised`);
    engine.dispose();
  }
  const q=new THREE.Quaternion(.1,.2,.3,.927).normalize();assert.ok(flipVRM0(flipVRM0(q)).angleTo(q)<1e-9);
});

test('gestures keep their direction on VRM 0.x: a wave raises the right hand above the shoulder',()=>{
  for(const version of ['0','1']){
    const {nodes,vrm}=armRig(version),engine=new MotionEngine(vrm,still),wave=[{id:'wave',kind:'gesture',status:'running',duration:2,payload:{name:'wave',intensity:.8}}];
    for(let i=0;i<12;i++)engine.update({actions:wave},.05);
    assert.ok(armDirection(nodes,'rightUpperArm','rightLowerArm').y>.3,`VRM ${version} wave lowers the arm`);
    const expected=gesturePose('wave',engine.ages.get('wave'),new Map(),.8).get('rightUpperArm');
    assert.ok(nodes.get('rightUpperArm').quaternion.angleTo(version==='0'?flipVRM0(expected):expected)<1e-6);
    engine.dispose();
  }
});

test('a .pose.json is canonical on VRM 0.x and its expression targets go to the face layer, never to the manager',()=>{
  const {nodes,vrm}=armRig('0');vrm.expressionManager={setValue:()=>assert.fail('MotionEngine wrote an expression'),getValue:()=>0};
  const sampler=new MotionAssetSampler(vrm),asset={id:'pose',kind:'pose',mask:['head'],pose:{bones:{head:[.3,0,.2]},expressions:{happy:.5}}};
  const engine=new MotionEngine(vrm,sampler),actions=[{id:'preview',kind:'motion.preview',status:'running',payload:{asset}}];
  for(let i=0;i<20;i++)engine.update({actions},.05);
  assert.ok(nodes.get('head').quaternion.angleTo(flipVRM0(new THREE.Quaternion().setFromEuler(new THREE.Euler(.3,0,.2,'XYZ'))))<1e-6);
  assert.equal(engine.expressionTargets.get('happy'),.5);engine.dispose();
});

test('a .vrma clip on VRM 0.x reaches its node native, never flipped twice',async()=>{
  const scene=new THREE.Scene(),head=new THREE.Object3D();head.name='Head';scene.add(head);
  const q=new THREE.Quaternion(.3,.2,.4,.84).normalize(),native=flipVRM0(q),values=[...q.toArray(),...q.toArray()];
  const animation={duration:1,humanoidTracks:{rotation:new Map([['head',new THREE.QuaternionKeyframeTrack('head.quaternion',[0,1],values)]]),translation:new Map()},expressionTracks:{preset:new Map(),custom:new Map()},lookAtTrack:null};
  const vrm={scene,meta:{metaVersion:'0'},humanoid:{getNormalizedBoneNode:name=>name==='head'?head:null},expressionManager:{getExpression:()=>null},lookAt:null};
  const sampler=new MotionAssetSampler(vrm,{load:async()=>({scene:new THREE.Scene(),userData:{vrmAnimations:[animation]}})});
  const engine=new MotionEngine(vrm,sampler),actions=[{id:'preview',kind:'motion.preview',status:'running',payload:{asset:{id:'vrma',kind:'vrma',mask:['head'],loop:true,speed:1}}}];
  engine.update({actions},.05);await new Promise(done=>setTimeout(done));
  for(let i=0;i<20;i++)engine.update({actions},.05);
  assert.ok(head.quaternion.angleTo(native)<1e-5,'the clip is not where createVRMAnimationClip puts it');
  assert.ok(head.quaternion.angleTo(q)>.5,'the clip was flipped twice');
  engine.dispose();
});

const hold=(state,pose={},transition=.25)=>({state,transition,pose:{variant:'tuck',startle:0,alarm:0,limp:0,kick:0,kickPhase:0,sway:0,squash:0,mirror:1,...pose},breath:{phase:0,depth:0}});
const expect=(version,q)=>version==='0'?flipVRM0(q):q;

test('interaction.hold picks the held pose, crossfades in its transition time, flips once for VRM 0.x and ignores the pointer head-gaze',()=>{
  for(const version of ['0','1']){
    const {nodes,vrm}=armRig(version),engine=new MotionEngine(vrm,still),interaction={held:true,hold:hold('dangling'),pointer:{x:1,y:1,near:true}};
    engine.update({interaction,held:{pitch:30}},.1);
    assert.ok(nodes.get('hips').quaternion.angleTo(expect(version,heldPose(interaction.hold,engine.time,{pitch:30}).get('hips')))>.05,'still crossfading after 0.1 of 0.25 s');
    for(let i=0;i<3;i++)engine.update({interaction,held:{pitch:30}},.05);
    const target=heldPose(interaction.hold,engine.time,{pitch:30});
    for(const bone of ['hips','head','neck','leftUpperLeg','rightLowerArm','leftFoot'])assert.ok(nodes.get(bone).quaternion.angleTo(expect(version,target.get(bone)))<1e-6,`VRM ${version} ${bone}`);
    // a backend base change while held never restarts the crossfade; a new hold state does, over its own time
    engine.update({interaction,held:{pitch:30},actions:[{id:'b2',kind:'motion.base',status:'running',payload:{procedural:'held'}}]},.05);
    assert.ok(nodes.get('hips').quaternion.angleTo(expect(version,heldPose(interaction.hold,engine.time,{pitch:30}).get('hips')))<1e-6);
    const limp={...interaction,hold:hold('resigned',{limp:1},.4)};for(let i=0;i<2;i++)engine.update({interaction:limp,held:{pitch:30}},.1);
    const halfway=nodes.get('leftUpperArm').quaternion.clone();for(let i=0;i<2;i++)engine.update({interaction:limp,held:{pitch:30}},.1);
    const resigned=expect(version,heldPose(limp.hold,engine.time,{pitch:30}).get('leftUpperArm'));
    assert.ok(halfway.angleTo(resigned)>.05&&nodes.get('leftUpperArm').quaternion.angleTo(resigned)<1e-6,'0.4 s between held states');
    engine.dispose();
  }
});

test('the in-place settle bends the knees on top of idle, a backend or hit-rule settling does too, and a hit rule still overrides the held pose',()=>{
  const {nodes,vrm}=armRig('1'),engine=new MotionEngine(vrm,still);
  for(let i=0;i<10;i++)engine.update({interaction:{hold:hold('recovering',{squash:.4},.35)}},.05);
  assert.ok(nodes.get('leftLowerLeg').quaternion.angleTo(settlePose(proceduralPose('idle',engine.time),.4).get('leftLowerLeg'))<1e-6);
  assert.ok(nodes.get('leftLowerLeg').quaternion.angleTo(new THREE.Quaternion())>15*Math.PI/180);
  const other=armRig('1'),plain=new MotionEngine(other.vrm,still);for(let i=0;i<10;i++)plain.update({interaction:{reaction:'settling'}},.05);
  assert.ok(other.nodes.get('leftLowerLeg').quaternion.angleTo(new THREE.Quaternion())>10*Math.PI/180);
  const ruled=armRig('1'),rule=new MotionEngine(ruled.vrm,still);
  for(let i=0;i<10;i++)rule.update({interaction:{held:true,hold:hold('dangling'),rule:{animation:'thinking'}}},.05);
  assert.ok(ruled.nodes.get('hips').quaternion.angleTo(new THREE.Quaternion())<1e-6,'the rule\'s thinking pose, not the scruff tilt');
  for(const e of [engine,plain,rule])e.dispose();
});

test('a fall shows the falling pose; landing then recovering is one settle whose crossfade never restarts',()=>{
  const {nodes,vrm}=armRig('0'),engine=new MotionEngine(vrm,still);
  for(let i=0;i<10;i++)engine.update({interaction:{hold:hold('falling',{fall:1},.2)},held:{pitch:30}},.05);
  assert.ok(nodes.get('leftUpperArm').quaternion.angleTo(expect('0',heldPose(hold('falling',{fall:1}),engine.time,{pitch:30}).get('leftUpperArm')))<1e-6,'the falling pose, flipped once');
  const landing=hold('landing',{squash:.6},.08);engine.update({interaction:{hold:landing}},.04);const mid=nodes.get('leftLowerLeg').quaternion.clone();
  engine.update({interaction:{hold:{...landing,state:'recovering',pose:{...landing.pose,squash:.6}}}},.04);
  const settled=expect('0',settlePose(proceduralPose('idle',engine.time),.6).get('leftLowerLeg'));
  assert.ok(nodes.get('leftLowerLeg').quaternion.angleTo(settled)<1e-6,'the landing crossfade (0.08 s) finished across the change to recovering');assert.ok(mid.angleTo(settled)>1e-3,'it was still blending at 0.04 s');
  engine.dispose();
});

test('reversing a drag mid-hold crossfades the legs and never moves a displayed bone more than a few degrees in a frame',()=>{
  for(const random of [()=>.8,()=>.5]){  // the reach and tuck variants
    const {nodes,vrm}=armRig('0'),engine=new MotionEngine(vrm,still),hold=new HoldBehaviour({random}),settings={enabled:true,protestSeconds:6,resignSeconds:18,recoverySeconds:2.5};
    let t=0;hold.grab(t,{kind:'body'});const frame=vx=>{t+=1/60;const out=hold.step(t,1/60,{vx},settings);engine.update({interaction:{held:true,hold:out},held:{pitch:30}},1/60);return out;};
    for(let i=0;i<72;i++)frame(0);
    let previous=new Map([...nodes].map(([bone,node])=>[bone,node.quaternion.clone()])),worst=0;
    for(let i=0;i<60;i++){const out=frame(i<20?300:-400);assert.equal(out.state,'dangling','slow enough to stay dangling');
      for(const [bone,node] of nodes){worst=Math.max(worst,node.quaternion.angleTo(previous.get(bone))*180/Math.PI);previous.set(bone,node.quaternion.clone());}}
    assert.equal(hold.mirror,-1);assert.ok(worst<4,`${hold.variant}: a bone moved ${worst.toFixed(1)} degrees in one frame`);
    engine.dispose();
  }
});
