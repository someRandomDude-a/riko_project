import test from 'node:test';
import assert from 'node:assert/strict';
import {Bone,Group,PerspectiveCamera,Vector3,Quaternion} from 'three';
import {AvatarMouseSphere,endpointPosition,nearestEndpoint} from './avatar_mouse_sphere.mjs';
import {normalizeProfile} from './avatar_studio_settings.mjs';
import {VRMSpringBoneJoint,VRMSpringBoneManager} from '@pixiv/three-vrm';

function fixture(length=.04,virtual=false){
 const scene=new Group(),root=new Bone(),joint=new Bone(),tail=new Bone();scene.add(root);root.add(joint);joint.position.y=.1;
 if(!virtual){joint.add(tail);tail.position.y=length;}
 const entry={id:'root/0/0',node:joint,joint:{child:virtual?null:tail,initialLocalChildPosition:new Vector3(0,length,0)}};
 const catalog={entries:[entry]},sphere=new AvatarMouseSphere(catalog,scene),camera=new PerspectiveCamera(30,1,.01,10);camera.position.z=2;camera.updateMatrixWorld(true);scene.updateMatrixWorld(true);
 const rect={left:0,top:0,width:1000,height:1000};
 const screen=position=>{const p=position.clone().project(camera);return {x:(p.x+1)*500,y:(1-p.y)*500};};
 function begin(){const point=screen(endpointPosition({node:joint,localTail:new Vector3(0,length,0)}));const prepared=sphere.prepare(entry,point,rect,camera);sphere.start(entry,prepared);return {point,prepared};}
 function settle(profile,frames=120){for(let i=0;i<frames;i++){sphere.restore();scene.updateMatrixWorld(true);sphere.update(profile,1/60);scene.updateMatrixWorld(true);}}
 return {scene,root,joint,tail,entry,sphere,camera,rect,screen,begin,settle,length};
}
test('initial click captures the endpoint and depth, independent of stale smoothed sphere state',()=>{
 const f=fixture();f.sphere.position.set(99,0,99);
 const position=f.tail.getWorldPosition(new Vector3()),point=f.screen(position.clone().add(new Vector3(.005,0,0)));
 const prepared=f.sphere.prepare(f.entry,point,f.rect,f.camera);assert.equal(prepared.endpoint.node,f.tail);assert.equal(prepared.endpoint.kind,'tail');
 f.sphere.start(f.entry,prepared);f.sphere.aim(point,f.rect,f.camera,.5);
 assert.ok(f.sphere.drag.target.distanceTo(position)<1e-8);
 f.sphere.aim({x:point.x+100,y:point.y},f.rect,f.camera,.5);
 assert.equal(f.sphere.drag.endpoint,prepared.endpoint);assert.ok(f.sphere.drag.target.x>position.x+.05);assert.ok(Math.abs(f.sphere.drag.target.z-position.z)<1e-8);
});
test('short spring segments visibly follow dragging, with length-independent strength and a fixed attachment',()=>{
 const angles=[];
 for(const length of [.02,.2]){
  const f=fixture(length);f.begin();f.sphere.drag.target.add(new Vector3(length,0,0));
  const p=normalizeProfile({mouseSphere:{enabled:false,dragSpring:25,maxAngle:90}});f.settle(p);
  const angle=f.joint.quaternion.angleTo(new Quaternion());assert.ok(angle>.3);angles.push(angle);
  assert.equal(f.root.quaternion.angleTo(new Quaternion()),0);assert.equal(f.joint.position.y,.1);assert.equal(f.tail.position.y,length);assert.equal(f.scene.position.length(),0);
  f.sphere.restore();assert.ok(f.joint.quaternion.angleTo(new Quaternion())<1e-8);
 }
 assert.ok(Math.abs(angles[0]-angles[1])<1e-6);
});
test('drag strength is live, increases follow-through and zero stops pulling',()=>{
 const f=fixture();f.begin();f.sphere.drag.target.x=.04;
 const profile=strength=>normalizeProfile({mouseSphere:{enabled:false,dragSpring:strength,maxAngle:90}});
 f.settle(profile(5));const weak=f.joint.quaternion.angleTo(new Quaternion());
 f.settle(profile(100));const strong=f.joint.quaternion.angleTo(new Quaternion());assert.ok(strong>weak*2);
 f.settle(profile(0));assert.ok(f.joint.quaternion.angleTo(new Quaternion())<1e-5);
 assert.equal(normalizeProfile({mouseSphere:{dragSpring:999}}).mouseSphere.dragSpring,100);
 assert.equal(normalizeProfile({mouseSphere:{dragSpring:NaN}}).mouseSphere.dragSpring,25);
});
test('virtual spring tails can be selected and dragged without translating the bone',()=>{
 const f=fixture(.04,true),position=f.joint.getWorldPosition(new Vector3()).add(new Vector3(0,.04,0));
 const endpoint=nearestEndpoint(f.entry,position);assert.equal(endpoint.kind,'tail');assert.ok(endpoint.localTail);assert.ok(endpointPosition(endpoint).distanceTo(position)<1e-8);
 f.begin();f.sphere.drag.target.x=.04;f.settle(normalizeProfile({mouseSphere:{enabled:false}}));
 assert.ok(f.joint.quaternion.angleTo(new Quaternion())>.3);assert.equal(f.joint.position.y,.1);assert.equal(f.entry.joint.initialLocalChildPosition.y,.04);
});
test('dragging a fixed root head does not rotate or move the attachment',()=>{
 const f=fixture();f.sphere.position.copy(f.joint.getWorldPosition(new Vector3()));assert.equal(f.sphere.start(f.entry).kind,'head');
 f.sphere.target.x=.1;f.sphere.active=true;f.sphere.initialized=true;
 f.settle(normalizeProfile({mouseSphere:{enabled:false}}));assert.equal(f.joint.quaternion.angleTo(new Quaternion()),0);assert.equal(f.joint.position.y,.1);
});
test('high drag strength remains bounded on rotated rigs and release blends out',()=>{
 const f=fixture();f.scene.rotation.set(.3,.5,.1);f.joint.rotation.set(.1,-.2,.3);f.scene.updateMatrixWorld(true);const baseline=f.joint.quaternion.clone();
 f.begin();f.sphere.drag.target.add(new Vector3(10,-5,2));const p=normalizeProfile({mouseSphere:{enabled:false,dragSpring:100,maxAngle:20}});
 for(let i=0;i<120;i++){f.sphere.restore();f.scene.updateMatrixWorld(true);f.sphere.update(p,.1);assert.ok(f.joint.quaternion.toArray().every(Number.isFinite));assert.ok(f.joint.quaternion.angleTo(baseline)<=20*Math.PI/180+1e-6);}
 f.sphere.release();f.settle(p);assert.ok(f.joint.quaternion.angleTo(baseline)<1e-5);
});
test('the drag layer remains effective after the imported VRM spring simulation updates',()=>{
 const f=fixture(),joint=new VRMSpringBoneJoint(f.joint,f.tail,{stiffness:2,dragForce:.4}),manager=new VRMSpringBoneManager();
 manager.addJoint(joint);manager.setInitState();f.entry.joint=joint;f.begin();f.sphere.drag.target.x=.04;
 const profile=normalizeProfile({mouseSphere:{enabled:false,dragSpring:50}});
 for(let i=0;i<120;i++){f.sphere.restore();f.scene.updateMatrixWorld(true);manager.update(1/60);f.scene.updateMatrixWorld(true);f.sphere.update(profile,1/60);}
 assert.ok(f.joint.quaternion.angleTo(new Quaternion())>.3);assert.equal(joint.settings.stiffness,2);assert.equal(f.joint.position.y,.1);
 f.sphere.restore();assert.ok(f.joint.quaternion.angleTo(new Quaternion())<1e-6);
});
test('dragging a joint head pulls its movable upstream owner, not the joint itself',()=>{
 const f=fixture(),owner={id:'root/0',node:f.root,joint:{child:f.joint}};f.sphere.catalog.entries.unshift(owner);
 const point=f.screen(f.joint.getWorldPosition(new Vector3())),prepared=f.sphere.prepare(f.entry,point,f.rect,f.camera);assert.equal(prepared.endpoint.kind,'head');
 f.sphere.start(f.entry,prepared);f.sphere.drag.target.x=.1;f.settle(normalizeProfile({mouseSphere:{enabled:false}}));
 assert.ok(f.root.quaternion.angleTo(new Quaternion())>.3);assert.ok(f.joint.quaternion.angleTo(new Quaternion())<1e-8);assert.equal(f.root.position.length(),0);
});
