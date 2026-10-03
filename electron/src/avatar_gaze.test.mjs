import test from 'node:test';
import assert from 'node:assert/strict';
import {Bone,Group,PerspectiveCamera,Vector3,Quaternion} from 'three';
import {VRMLookAt,VRMLookAtBoneApplier,VRMLookAtRangeMap} from '@pixiv/three-vrm';
import {AvatarEyeGaze} from './avatar_gaze.mjs';
import {normalizeProfile} from './avatar_studio_settings.mjs';

function fixture(boneMode=true){
 const scene=new Group(),head=new Bone(),leftEye=new Bone(),rightEye=new Bone();
 scene.add(head);head.add(leftEye,rightEye);leftEye.position.set(.032,0,.01);rightEye.position.set(-.032,0,.01);scene.updateMatrixWorld(true);
 const raw={head,leftEye,rightEye},normalized={head:new Bone(),leftEye:new Bone(),rightEye:new Bone()};
 const humanoid={getRawBoneNode:name=>raw[name],getNormalizedBoneNode:name=>normalized[name]};
 const shared=[];const applier=boneMode?new VRMLookAtBoneApplier(humanoid,...Array.from({length:4},()=>new VRMLookAtRangeMap(90,90))):{applyYawPitch:(yaw,pitch)=>shared.push({yaw,pitch})};
 const lookAt=new VRMLookAt(humanoid,applier),vrm={humanoid,lookAt};
 const gaze=new AvatarEyeGaze(vrm),settings=normalizeProfile({}).gaze,camera=new PerspectiveCamera(30,1,.01,100);camera.position.z=3;camera.updateMatrixWorld(true);
 function update(target,active=true){gaze.restore();scene.updateMatrixWorld(true);gaze.update(settings,.1,{x:50,y:50},{left:0,top:0,width:100,height:100},camera,{active:true,initialized:true,position:target},active);}
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
