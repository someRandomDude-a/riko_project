import test from 'node:test';
import assert from 'node:assert/strict';
import {Object3D,Bone,Group,Ray,Vector3} from 'three';
import {AvatarBoneBVH} from './avatar_bvh.mjs';

function avatar(){
 const root=new Object3D(),bones={};
 for(const [name,x,y] of [['hips',0,1],['spine',0,1.2],['neck',0,1.6],['head',0,1.8],['leftUpperArm',.3,1.5],['leftLowerArm',.6,1.5],['leftHand',.9,1.5]]){
  const node=new Object3D();node.position.set(x,y,0);root.add(node);bones[name]=node;
 }
 root.updateMatrixWorld(true);
 return {root,bones,bvh:AvatarBoneBVH.fromVRM({humanoid:{getNormalizedBoneNode:name=>bones[name]}},2)};
}
const ray=(x,y,z=3)=>new Ray(new Vector3(x,y,z),new Vector3(0,0,-1));
test('bone BVH picks torso, head and limbs but leaves surrounding canvas transparent',()=>{
 const {bvh}=avatar();
 for(const [x,y] of [[0,1.1],[0,1.5],[0,1.8],[.45,1.5],[.9,1.5]])assert.equal(bvh.intersects(ray(x,y)),true);
 assert.equal(bvh.intersects(ray(2,1)),false);
 assert.equal(bvh.intersects(ray(.45,1.1)),false);
 assert.equal(bvh.intersects(new Ray(new Vector3(0,1.5,3),new Vector3(0,0,1))),false);
 assert.equal(bvh.hit(ray(.9,1.5)).region,'arms');
 assert.ok(['leftHand','leftLowerArm'].includes(bvh.hit(ray(.9,1.5)).bone));
});
test('hit preferences validate rules, bone filters and expression intensity',async()=>{
 const {normalizeHitSettings}=await import('./avatar_hit_settings.mjs');
 const {HIT_BONES}=await import('./avatar_bvh.mjs');
 const settings=normalizeHitSettings({avatarHitOutlines:'yes',avatarHitBones:['head','invalid','head'],avatarHitRules:{head:{hold:{animation:'held',expression:'happy',intensity:9},click:{animation:'invalid',expression:'invalid',intensity:NaN}},invalid:{hold:{}}}},HIT_BONES);
 assert.equal(settings.avatarHitOutlines,false);assert.deepEqual(settings.avatarHitBones,['head']);
 assert.deepEqual(settings.avatarHitRules.head.hold,{animation:'held',expression:'happy',intensity:1});
 assert.deepEqual(settings.avatarHitRules.head.click,{animation:'default',expression:'default',intensity:.7});
 assert.equal(settings.avatarHitRules.invalid,undefined);
});
test('debug bone filtering does not alter pick volumes',async()=>{
 const {AvatarHitOutlines}=await import('./avatar_hit_outlines.mjs');
 const {Group}=await import('three');
 const {bvh}=avatar(),scene=new Group(),debug=new AvatarHitOutlines(bvh,scene);
 debug.update({avatarHitOutlines:true,avatarHitBones:['head']},{bone:'head'});
 assert.equal(debug.group.visible,true);
 assert.equal(debug.parts.find(part=>part.volume.bone==='head').first.visible,true);
 assert.equal(debug.parts.find(part=>part.volume.bone==='hips').first.visible,false);
 assert.equal(bvh.intersects(ray(0,1.1)),true);
 debug.update({avatarHitOutlines:false},null);assert.equal(debug.group.visible,false);
});
test('refit follows animated bones without rebuilding the BVH',()=>{
 const {root,bones,bvh}=avatar(),tree=bvh.root;
 bones.leftLowerArm.position.y=2.5;bones.leftHand.position.y=3;
 root.updateMatrixWorld(true);bvh.refit();
 assert.equal(bvh.root,tree);
 assert.equal(bvh.intersects(ray(.75,2.75)),true);
 assert.equal(bvh.intersects(ray(.75,1.5)),false);
});
test('empty skeleton safely has no pickable volume',()=>{
 const bvh=AvatarBoneBVH.fromVRM({humanoid:{getNormalizedBoneNode:()=>null}},2);
 bvh.refit();assert.equal(bvh.intersects(ray(0,0)),false);
});
test('every spring has an animated pick volume, including humanoid and virtual terminal joints',async()=>{
 const {boneCatalog}=await import('./avatar_secondary.mjs');
 const scene=new Group(),head=new Bone(),hair=new Bone(),tip=new Bone();head.name='head';hair.name=tip.name='hair';head.position.set(0,1.8,0);hair.position.set(.4,0,0);tip.position.y=-.2;scene.add(head);head.add(hair);hair.add(tip);
 const joints=[{bone:head,child:hair},{bone:hair,child:tip},{bone:tip,child:null,initialLocalChildPosition:new Vector3(0,-.15,0)}];
 const vrm={scene,humanoid:{normalizedHumanBones:{head:{}},getRawBoneNode:name=>name==='head'?head:null},springBoneManager:{joints:new Set(joints)}};
 scene.updateMatrixWorld(true);const catalog=boneCatalog(vrm,'spring.vrm'),bvh=AvatarBoneBVH.fromVRM(vrm,2,catalog),springs=bvh.volumes.filter(v=>v.spring);
 assert.equal(springs.length,joints.length);assert.equal(new Set(springs.map(v=>v.entry.id)).size,joints.length);
 assert.equal(springs.find(v=>v.first===head).bone,'head');assert.equal(bvh.hit(ray(0,1.8)).bone,'head');
 const terminal=springs.find(v=>v.first===tip);assert.ok(terminal.b.distanceTo(new Vector3(.4,1.45,0))<1e-8);
 assert.equal(bvh.hit(ray(.4,1.47)).bone,terminal.bone);
 assert.equal(AvatarBoneBVH.fromVRM(vrm,2).volumes.filter(v=>v.spring).length,3);
 const root=bvh.root;tip.rotation.z=Math.PI/2;scene.updateMatrixWorld(true);bvh.refit(.06);
 assert.equal(bvh.root,root);assert.ok(terminal.b.distanceTo(new Vector3(.55,1.6,0))<1e-8);assert.equal(terminal.radius,.06);
 assert.equal(bvh.hit(ray(.56,1.6)).bone,terminal.bone);assert.equal(joints[2].initialLocalChildPosition.y,-.15);
});
test('spring selection padding and outline filters are independent of physics and picking',async()=>{
 const {AvatarHitOutlines}=await import('./avatar_hit_outlines.mjs');const {normalizeHitSettings}=await import('./avatar_hit_settings.mjs');
 const scene=new Group(),node=new Bone(),child=new Bone();node.position.x=1;child.position.y=.2;scene.add(node);node.add(child);scene.updateMatrixWorld(true);
 const entry={id:'root/0',node,joint:{child,settings:{hitRadius:.001}}},bvh=AvatarBoneBVH.fromVRM({},2,{entries:[entry]});
 assert.equal(bvh.hit(ray(1.04,.1)),null);bvh.refit(.06);assert.equal(bvh.hit(ray(1.04,.1)).bone,entry.id);assert.equal(entry.joint.settings.hitRadius,.001);
 const debug=new AvatarHitOutlines(bvh,scene),p=normalizeHitSettings({avatarHitOutlines:true,avatarHitBones:[entry.id,'not-a-bone']},['head']);
 assert.deepEqual(p.avatarHitBones,[entry.id]);debug.update(p,{bone:entry.id});assert.equal(debug.parts[0].body.visible,true);assert.equal(debug.parts[0].body.material,debug.selected);assert.equal(debug.parts[0].first.scale.x,.06);
 debug.update({avatarHitOutlines:true,avatarHitBones:['head']},null);assert.equal(debug.parts[0].first.visible,false);assert.equal(bvh.hit(ray(1.04,.1)).bone,entry.id);
});
