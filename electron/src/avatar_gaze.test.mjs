import test from 'node:test';
import assert from 'node:assert/strict';
import {Bone,Group,PerspectiveCamera,Vector3,Quaternion} from 'three';
import {VRMLookAt,VRMLookAtBoneApplier,VRMLookAtRangeMap} from '@pixiv/three-vrm';
import {AvatarEyeGaze,SaccadeGaze,SACCADE,gazeActive} from './avatar_gaze.mjs';
import {normalizeProfile} from './avatar_studio_settings.mjs';

// The exponential glide's tests run with saccades off (settings.saccades false keeps it exactly); the saccade tests below set it on.
function fixture(boneMode=true,{saccades=false,random=()=>.5}={}){
 const scene=new Group(),head=new Bone(),leftEye=new Bone(),rightEye=new Bone();
 scene.add(head);head.add(leftEye,rightEye);leftEye.position.set(.032,0,.01);rightEye.position.set(-.032,0,.01);scene.updateMatrixWorld(true);
 const raw={head,leftEye,rightEye},normalized={head:new Bone(),leftEye:new Bone(),rightEye:new Bone()};
 const humanoid={getRawBoneNode:name=>raw[name],getNormalizedBoneNode:name=>normalized[name]};
 const shared=[];const applier=boneMode?new VRMLookAtBoneApplier(humanoid,...Array.from({length:4},()=>new VRMLookAtRangeMap(90,90))):{applyYawPitch:(yaw,pitch)=>shared.push({yaw,pitch})};
 const lookAt=new VRMLookAt(humanoid,applier),vrm={humanoid,lookAt};
 const gaze=new AvatarEyeGaze(vrm,{random}),settings={...normalizeProfile({}).gaze,saccades},camera=new PerspectiveCamera(30,1,.01,100);camera.position.z=3;camera.updateMatrixWorld(true);
 function update(target,active=true,dt=.1,override=null){gaze.restore();scene.updateMatrixWorld(true);gaze.update(settings,dt,{x:50,y:50},{left:0,top:0,width:100,height:100},camera,{active:true,initialized:true,position:target},active,override);}
 function settle(target){for(let i=0;i<30;i++)update(target);}
 return {scene,head,leftEye,rightEye,normalized,lookAt,gaze,settings,camera,shared,update,settle};
}
test('gaze converges more for near targets, without changing anatomical eye positions',()=>{
 const f=fixture(),left=f.leftEye.position.clone(),right=f.rightEye.position.clone();
 f.settle(new Vector3(0,0,2));const far=Math.abs(f.gaze.angles.left.yaw-f.gaze.angles.right.yaw);
 f.settle(new Vector3(0,0,.2));const near=Math.abs(f.gaze.angles.left.yaw-f.gaze.angles.right.yaw);
 assert.ok(near>far*5);assert.ok(f.gaze.angles.left.yaw*f.gaze.angles.right.yaw<0);
 assert.ok(f.leftEye.quaternion.angleTo(f.rightEye.quaternion)>.01);
 assert.deepEqual(f.leftEye.position,left);assert.deepEqual(f.rightEye.position,right);
});
test('shared tracking is smoothed, bounded and convergence can be disabled',()=>{
 const f=fixture();f.settings.convergence=false;f.update(new Vector3(100,100,.2));
 assert.ok(f.gaze.angles.left.yaw!==0&&Math.abs(f.gaze.angles.left.yaw)<f.settings.maxYaw);
 assert.deepEqual(f.gaze.angles.left,f.gaze.angles.right);
 f.settle(new Vector3(100,100,-3));
 for(const angle of Object.values(f.gaze.angles)){assert.ok(Math.abs(angle.yaw)<=f.settings.maxYaw);assert.ok(Math.abs(angle.pitch)<=f.settings.maxPitch);}
 assert.ok(f.gaze.target.z>=f.settings.minDistance-1e-8);
});
test('gaze respects rotated heads and targets behind the model remain finite',()=>{
 const f=fixture();f.head.rotation.y=Math.PI/2;f.scene.updateMatrixWorld(true);f.settle(new Vector3(2,0,0));
 assert.ok(Math.abs((f.gaze.angles.left.yaw+f.gaze.angles.right.yaw)/2)<2);
 f.settle(new Vector3(-2,0,0));
 for(const eye of [f.leftEye,f.rightEye])assert.ok(eye.quaternion.toArray().every(Number.isFinite));
});
test('leaving eases toward neutral and restore/dispose remove the gaze layer',()=>{
 const f=fixture();const baseline=new Quaternion().setFromAxisAngle(new Vector3(0,1,0),.1);
 f.leftEye.quaternion.copy(baseline);f.update(new Vector3(.2,0,.2));f.gaze.restore();assert.ok(f.leftEye.quaternion.angleTo(baseline)<1e-7);
 f.settle(new Vector3(.2,0,.2));for(let i=0;i<30;i++)f.update(new Vector3(),false);
 assert.ok(Math.abs(f.gaze.angles.left.yaw)<1e-6);assert.equal(f.lookAt.autoUpdate,false);
 f.settings.enabled=false;f.update(new Vector3());assert.equal(f.lookAt.autoUpdate,true);
 f.settings.enabled=true;f.update(new Vector3(0,0,.2));f.gaze.dispose();assert.equal(f.lookAt.autoUpdate,true);assert.equal(f.gaze.saved.length,0);
});
test('expression-only rigs get shared tracking; missing lookAt safely does nothing',()=>{
 const f=fixture(false);f.settle(new Vector3(.2,.1,.3));assert.ok(f.shared.at(-1).yaw!==0);assert.equal(f.gaze.saved.length,0);
 new AvatarEyeGaze({}).update(f.settings,.1,null,{},null,null);new AvatarEyeGaze({}).dispose();
});
test('camera projection tracks the cursor when the mouse sphere is unavailable',()=>{
 const f=fixture();f.gaze.update(f.settings,.1,{x:75,y:25},{left:0,top:0,width:100,height:100},f.camera,null);
 assert.ok(f.gaze.target.x>0&&f.gaze.target.y>0);assert.ok(f.gaze.angles.left.yaw!==0&&f.gaze.angles.left.pitch!==0);
});
test('gaze and selection settings default on and reject non-finite or excessive values',()=>{
 const p=normalizeProfile({gaze:{strength:Infinity,minDistance:-3,maxConvergence:99,smoothing:0},springPickRadius:99});
 assert.equal(p.gaze.enabled,true);assert.equal(p.gaze.convergence,true);assert.equal(p.gaze.strength,1);assert.equal(p.gaze.minDistance,.05);assert.equal(p.gaze.maxConvergence,30);assert.equal(p.gaze.smoothing,1);assert.equal(p.springPickRadius,.2);
 assert.equal(normalizeProfile({springPickRadius:NaN}).springPickRadius,.025);
});

const D=Math.PI/180;
// A target `degrees` to her left at 1 m, seen from the origin.
const at=degrees=>new Vector3(Math.sin(degrees*D),0,Math.cos(degrees*D));
const sideways=(point,origin=new Vector3())=>Math.atan2(point.x-origin.x,point.z-origin.z)/D;

test('a 10-degree target step is one saccade of (21 + 2.2 A) ms with a smoothstep profile, then a fixation of at least 0.12 s',()=>{
 const gaze=new SaccadeGaze({random:()=>.5}),origin=new Vector3(),dt=1/240;
 for(let i=0;i<240;i++)gaze.step(at(0),origin,dt);  // fixed on the first target
 const track=[];for(let i=0;i<240&&!(track.length&&!gaze.saccading);i++)track.push(sideways(gaze.step(at(10),origin,dt)));  // until the jump ends
 const start=track.findIndex(v=>v>.01),end=track.findIndex(v=>Math.abs(v-10)<1e-6);
 assert.equal(start,0,'the fixation was long enough: it jumps at once');
 assert.ok(Math.abs((end+1)*dt-(.021+.0022*10))<=dt+1e-9,`done after ${((end+1)*dt*1000).toFixed(1)} ms (43 expected)`);
 assert.ok(track.slice(0,end+1).every((v,i)=>!i||v>=track[i-1]-1e-9),'it only moves toward the target');
 const mid=track[Math.round(end/2)];assert.ok(mid>3.5&&mid<6.5,`halfway through it is halfway there (${mid.toFixed(2)})`);
 // a new target right as the jump ends waits for the fixation to reach 0.12 s
 const next=[];for(let i=0;i<120;i++)next.push(sideways(gaze.step(at(-10),origin,dt)));
 const moved=next.findIndex(v=>v<9.99);assert.ok(moved*dt>=SACCADE.fixation-2*dt&&moved*dt<=SACCADE.fixation+dt,`it left the fixation after ${(moved*dt*1000).toFixed(0)} ms`);
});

test('micro-saccades during a fixation stay under 0.5 degrees and come every 0.6-1.5 s; small target drift under 1.5 degrees never jumps',()=>{
 let seed=7;const random=()=>{seed=(seed*16807)%2147483647;return seed/2147483647;};
 const gaze=new SaccadeGaze({random}),origin=new Vector3(),dt=1/60;gaze.step(at(0),origin,dt);
 let changes=0,last={...gaze.micro},worst=0;const times=[];
 for(let i=0;i<60*20;i++){gaze.step(at(1.2*Math.sin(i/200)),origin,dt);worst=Math.max(worst,Math.hypot(gaze.micro.yaw,gaze.micro.pitch));
  if(Math.hypot(gaze.micro.yaw-last.yaw,gaze.micro.pitch-last.pitch)>.05&&(!times.length||i*dt-times.at(-1)>.1)){changes++;times.push(i*dt);}last={...gaze.micro};assert.equal(gaze.saccading,false);}
 assert.ok(worst>.1&&worst<=SACCADE.microMax+1e-9,`micro-saccades reach ${worst.toFixed(3)} degrees`);
 assert.ok(changes>=13&&changes<=34,`${changes} micro-saccades in 20 s`);
 for(let i=1;i<times.length;i++)assert.ok(times[i]-times[i-1]>=.6-dt-1e-9&&times[i]-times[i-1]<=1.5+.1,`gap ${(times[i]-times[i-1]).toFixed(2)} s`);
});

test('a target moving more than 5 degrees mid-jump re-plans it; a 30-degree shift sets bigShift for one step; a turning head keeps the eyes on the point',()=>{
 const gaze=new SaccadeGaze({random:()=>.5}),origin=new Vector3(),dt=1/240;for(let i=0;i<60;i++)gaze.step(at(0),origin,dt);
 gaze.step(at(20),origin,dt);assert.ok(gaze.saccading);const to=gaze.to.clone();gaze.step(at(22),origin,dt);assert.ok(gaze.to.equals(to),'2 degrees: the jump goes on');
 gaze.step(at(-10),origin,dt);assert.ok(Math.abs(sideways(gaze.to)+10)<1e-9,'32 degrees away: re-planned');
 for(let i=0;i<120;i++)gaze.step(at(-10),origin,dt);assert.ok(Math.abs(sideways(gaze.current)+10)<1e-9);
 const flags=[];for(let i=0;i<120;i++){gaze.step(at(25),origin,dt);flags.push(gaze.bigShift);}assert.equal(flags.filter(Boolean).length,1,'35 degrees: one big shift');
 const small=[];for(let i=0;i<120;i++){gaze.step(at(5),origin,dt);small.push(gaze.bigShift);}assert.equal(small.some(Boolean),false,'20 degrees is not big');
 const point=gaze.current.clone();for(let i=0;i<60;i++)gaze.step(at(5),new Vector3(.001*i,0,0),dt);assert.ok(gaze.current.equals(point),'a fixation is a world point: moving eyes do not drag it');
});

test('a big shift (a gaze-evoked blink) starts only from a fixation: a re-plan mid-jump is no new shift',()=>{
 const gaze=new SaccadeGaze({random:()=>.5}),origin=new Vector3(),dt=1/240,flags=[];let t=0;const step=deg=>{gaze.step(at(deg),origin,dt);t+=dt;if(gaze.bigShift)flags.push(t);};
 for(let i=0;i<60;i++)step(0);
 for(let i=0;i<240*2;i++)step(Math.sin(2*Math.PI*t)>0?20:-20);  // 40-degree jumps back and forth twice a second
 assert.ok(flags.length>=3&&flags.length<=5,`${flags.length} big shifts in 2 s (AvatarFace.gazeShift keeps their blinks 2 s apart)`);
 const replan=new SaccadeGaze({random:()=>.5});for(let i=0;i<60;i++)replan.step(at(0),origin,dt);replan.step(at(10),origin,dt);assert.equal(replan.bigShift,false);
 replan.step(at(-30),origin,dt);assert.ok(replan.saccading&&Math.abs(sideways(replan.to)+30)<1e-9,'re-planned 40 degrees away');assert.equal(replan.bigShift,false,'a re-plan mid-jump is no new shift');
});

test('gazeActive: a reaction\'s look decides; with no look she tracks the cursor when it is near, as without a reaction',()=>{
 assert.equal(gazeActive('cursor',false),true);assert.equal(gazeActive('viewer',true),false);assert.equal(gazeActive('ahead',true),false,'dizzy: the eyes rest');
 assert.equal(gazeActive(null,true),true,'landing and recovering with no look: normal tracking');assert.equal(gazeActive(null,false),false);assert.equal(gazeActive(undefined,undefined),false);
});

test('with saccades on, the eyes jump to the cursor, an override target wins even when the pointer is not near, and a big shift is reported',()=>{
 const f=fixture(true,{saccades:true});
 for(let i=0;i<60;i++)f.update(new Vector3(0,0,2),true,1/60);
 const yaw=()=>(f.gaze.angles.left.yaw+f.gaze.angles.right.yaw)/2;assert.ok(Math.abs(yaw())<1,'looking ahead');
 const seen=[];for(let i=0;i<30;i++){f.update(new Vector3(2,0,2),true,1/240);seen.push(yaw());}
 const end=seen.findIndex(v=>Math.abs(v-seen.at(-1))<.01);assert.ok(end>0&&end<=Math.ceil((.021+.0022*45)*240)+1,`the 45-degree jump took ${end} steps at 240 Hz`);
 let big=false;const g=fixture(true,{saccades:true});for(let i=0;i<60;i++)g.update(new Vector3(0,0,2),true,1/60);
 for(let i=0;i<60;i++){g.update(new Vector3(),false,1/60,new Vector3(-2,0,1));big||=g.gaze.bigShift;}
 assert.ok((g.gaze.angles.left.yaw+g.gaze.angles.right.yaw)/2<-30,'the override (far to her right) wins with active false');assert.ok(big,'a 60-degree shift is big');
 assert.equal(g.gaze.tracking,true);
 const r=g.gaze.reach;assert.ok(r.yaw>0&&r.pitch>0);
});

test('given a measure, sizes are the eyes\' own clamped turn: a target sweeping far past the clamp is a short saccade',()=>{
 const measure=p=>({yaw:Math.max(-45,Math.min(45,sideways(p))),pitch:0}),origin=new Vector3(),dt=1/240;
 const gaze=new SaccadeGaze({random:()=>.5});for(let i=0;i<60;i++)gaze.step(at(-80),origin,dt,measure);
 gaze.step(at(80),origin,dt,measure);assert.ok(gaze.saccading);assert.ok(Math.abs(gaze.amplitude-90)<1e-6,`160 degrees of target, 90 of eye turn (${gaze.amplitude.toFixed(1)})`);
 assert.ok(Math.abs(gaze.duration-(.021+.0022*90))<1e-9);
 const still=new SaccadeGaze({random:()=>.5});for(let i=0;i<60;i++)still.step(at(60),origin,dt,measure);still.step(at(80),origin,dt,measure);assert.equal(still.saccading,false,'past the clamp the eyes have nowhere to go');
});
