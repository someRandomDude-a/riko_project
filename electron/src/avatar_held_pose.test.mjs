import test from 'node:test';
import assert from 'node:assert/strict';
import {Group,Bone,Vector3,Quaternion,Euler} from 'three';
import {heldTable,heldPose,mirrorPose,landingPose,settlePose,toQuaternions,kickPose,tuckPose,fallPose,HELD_BONES,JOINT_LIMITS} from './avatar_held_pose.mjs';

const D=Math.PI/180;
// A canonical VRM 1.0 chain (faces +z, her left is +x) in the style of avatar_pickup.test.mjs.
function chain(){
 const scene=new Group(),nodes=new Map(),add=(name,parent,x,y,z)=>{const bone=new Bone();bone.position.set(x,y,z);(parent?nodes.get(parent):scene).add(bone);nodes.set(name,bone);};
 add('hips',null,0,.9,0);add('spine','hips',0,.1,0);add('chest','spine',0,.1,0);add('upperChest','chest',0,.1,0);add('neck','upperChest',0,.12,0);add('head','neck',0,.1,0);add('nose','head',0,.05,.1);
 for(const [s,x] of [['left',1],['right',-1]]){add(s+'Shoulder','upperChest',.05*x,.08,0);add(s+'UpperArm',s+'Shoulder',.1*x,0,0);add(s+'LowerArm',s+'UpperArm',.25*x,0,0);add(s+'Hand',s+'LowerArm',.25*x,0,0);add(s+'Fingers',s+'Hand',.08*x,0,0);
  add(s+'UpperLeg','hips',.08*x,-.05,0);add(s+'LowerLeg',s+'UpperLeg',0,-.4,0);add(s+'Foot',s+'LowerLeg',0,-.4,0);add(s+'Toes',s+'Foot',0,-.03,.1);}
 const apply=pose=>{for(const [name,node] of nodes)node.quaternion.copy(pose.get(name)||new Quaternion());scene.updateMatrixWorld(true);};
 const at=name=>nodes.get(name).getWorldPosition(new Vector3()),dir=(a,b)=>at(b).sub(at(a)).normalize();
 return {apply,at,dir,down:(a,b)=>Math.acos(-dir(a,b).y)/D};
}
const hold=(state,pose={})=>({state,pose:{variant:'tuck',...pose}});

test('the scruff tuck pitches the spine as one piece, hangs the thighs and points the toes; the limp arms hang too',()=>{
 const rig=chain();
 for(const P of [25,30,35]){
  const table=heldTable(hold('dangling'),0,{pitch:P});assert.deepEqual([table.hips[0],table.spine[0],table.chest[0],table.upperChest[0]],[P,0,0,0],'only the hips pitch');
  rig.apply(heldPose(hold('dangling'),0,{pitch:P}));
  assert.ok(Math.abs(Math.acos(rig.dir('hips','upperChest').y)/D-P)<1e-6,`spine tilt at P=${P}`);
  for(const s of ['left','right'])assert.ok(rig.down(s+'UpperLeg',s+'LowerLeg')<10,`${s} thigh ${rig.down(s+'UpperLeg',s+'LowerLeg').toFixed(1)} degrees from down`);
  for(const s of ['left','right'])assert.ok(rig.dir(s+'Foot',s+'Toes').y<-.6,'toes pointed below horizontal');
  for(const s of ['left','right'])assert.ok(rig.at(s+'Hand').z-rig.at('upperChest').z>.1,`${s} hand in front of the chest`);
  const face=rig.dir('head','nose');assert.ok(face.z>.85,`head faces the viewer (${face.toArray().map(v=>v.toFixed(2))})`);
 }
 rig.apply(heldPose(hold('resigned',{limp:1}),0,{pitch:30}));
 for(const s of ['left','right']){assert.ok(rig.down(s+'UpperArm',s+'LowerArm')<12,`limp ${s} arm ${rig.down(s+'UpperArm',s+'LowerArm').toFixed(1)}`);assert.ok(rig.down(s+'UpperLeg',s+'LowerLeg')<10);}
 rig.apply(heldPose(hold('dangling',{variant:'reach'}),0,{pitch:30}));assert.ok(rig.at('rightHand').y>rig.at('upperChest').y&&rig.at('rightHand').y>rig.at('leftHand').y+.15,'reach: one hand goes up toward the holder');assert.ok(rig.at('leftHand').y<rig.at('upperChest').y);
});

test('kicks are thighs in antiphase with lagging knees that never hyperextend, and every pose stays inside the joint limits',()=>{
 for(let phase=0;phase<2*Math.PI;phase+=.05){
  const k=kickPose(phase,22);assert.ok(k.leftUpperLeg[0]*k.rightUpperLeg[0]<=1e-9,'antiphase');assert.ok(k.leftLowerLeg[0]>=0&&k.rightLowerLeg[0]>=0);
 }
 const cycle=Array.from({length:720},(_,i)=>kickPose(i/720*2*Math.PI,22)),peak=Math.max(...cycle.map(k=>k.leftLowerLeg[0]));assert.ok(Math.abs(peak-55)<.1,'the knees lead: up to 2.5 times the kick size');
 const back=Math.max(...cycle.map(k=>k.leftUpperLeg[0])),forward=-Math.min(...cycle.map(k=>k.leftUpperLeg[0]));assert.ok(Math.abs(back-.45*22)<.01&&Math.abs(forward-.18*22)<.01,'the thighs swing 0.45 of it back and 0.18 forward, so they never split far');
 const lag=[...Array(360).keys()].reduce((best,d)=>kickPose(d*D,22).leftLowerLeg[0]>kickPose(best*D,22).leftLowerLeg[0]?d:best,0);assert.equal(lag,160,'the knee peaks 70 degrees after the thigh');
 // Measured on the composed rotations (the whole quaternion a bone gets, whatever its Euler parts), and the bend's direction: a knee
 // bends with x+, an elbow with y- on the left and y+ on the right, a hip flexes with x-.
 const check=(table,label)=>{const q=toQuaternions(table),angle=b=>2*Math.acos(Math.min(1,Math.abs(q.get(b).w)))/D,axis=(b,i)=>Math.sign([q.get(b).x,q.get(b).y,q.get(b).z][i]*Math.sign(q.get(b).w||1));
  for(const s of ['left','right']){
   const knee=angle(s+'LowerLeg'),elbow=angle(s+'LowerArm'),hip=new Euler().setFromQuaternion(q.get(s+'UpperLeg'),'XYZ').x/D;
   assert.ok(knee<=JOINT_LIMITS.knee[1]+1e-6&&(knee<1e-6||axis(s+'LowerLeg',0)>0),`${label} ${s} knee ${knee}`);
   assert.ok(elbow<=JOINT_LIMITS.elbow[1]+1e-6&&(elbow<1e-6||axis(s+'LowerArm',1)===(s==='left'?-1:1)),`${label} ${s} elbow ${elbow}`);
   assert.ok(-hip<=JOINT_LIMITS.hipFlex[1]+1e-6&&hip<=JOINT_LIMITS.hipExtend[1]+1e-6,`${label} ${s} hip ${hip}`);}};
 for(const variant of ['tuck','reach','swing'])for(const state of ['grabbed','dangling','carried','protesting','dizzy','resigned'])for(const P of [0,30,60])for(let phase=0;phase<2*Math.PI;phase+=.3)for(const mirror of [1,0,-1])for(const startle of [0,1])
  check(heldTable({state,pose:{variant,startle,alarm:state==='carried'?1:0,limp:state==='dizzy'||state==='resigned'?1:0,kick:state==='protesting'?1:0,kickPhase:phase,sway:1,mirror},breath:{phase,depth:3}},phase,{pitch:P,kickAmplitude:45}),`${variant} ${state} P${P}`);
 for(let s=-.15;s<=1.2;s+=.05)check({...tuckPose(0),...landingPose(s)},`landing ${s.toFixed(2)}`);
 for(let t=0;t<2;t+=.02)for(const mirror of [1,-1])check(heldTable({state:'falling',pose:{variant:'tuck',fall:1,mirror},breath:{phase:t,depth:1.5}},t,{pitch:30}),`falling ${t.toFixed(2)}`);
});

test('mirror swaps only the legs, continuously: -1 puts the more bent knee on the other side, 0 is halfway, and the arms never swap',()=>{
 const table=heldTable(hold('dangling'),0,{pitch:30});assert.deepEqual(mirrorPose(mirrorPose(table)),table);
 const mirrored=heldTable(hold('dangling',{mirror:-1}),0,{pitch:30}),half=heldTable(hold('dangling',{mirror:0}),0,{pitch:30});
 assert.ok(table.rightLowerLeg[0]>table.leftLowerLeg[0]);assert.equal(mirrored.leftLowerLeg[0],table.rightLowerLeg[0]);assert.equal(mirrored.rightLowerLeg[0],table.leftLowerLeg[0]);
 assert.deepEqual([mirrored.leftUpperLeg,mirrored.rightUpperLeg],[mirrorPose(table).leftUpperLeg,mirrorPose(table).rightUpperLeg]);
 assert.ok(Math.abs(half.leftLowerLeg[0]-half.rightLowerLeg[0])<1e-9,'halfway the knees match');
 const reach=heldTable(hold('dangling',{variant:'reach'}),0,{pitch:30}),flipped=heldTable(hold('dangling',{variant:'reach',mirror:-1}),0,{pitch:30});
 for(const bone of ['leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','head','hips'])assert.deepEqual(flipped[bone],reach[bone],`${bone} stays: the raised hand never jumps sides`);
 // a mirror flip spread over 0.4 s moves no bone more than a few degrees a frame, kicks included
 let previous=null,worst=0;for(let i=0;i<=24;i++){const u=i/24,m=1-2*u*u*(3-2*u),pose=heldPose({state:'protesting',pose:{variant:'reach',kick:1,kickPhase:1,mirror:m}},0,{pitch:30});
  if(previous)for(const [bone,q] of pose)worst=Math.max(worst,q.angleTo(previous.get(bone))/D);previous=pose;}
 assert.ok(worst<6,`the worst bone moves ${worst.toFixed(1)} degrees in a frame`);
 for(const bone of HELD_BONES)assert.ok(bone in mirrored,`${bone} kept`);
});

test('the landing settle flexes knees, hips and ankles together and turns into a small stretch below 0; settlePose multiplies it on',()=>{
 let last=-1;for(const s of [0,.2,.4,.8,1]){const l=landingPose(s);assert.ok(l.leftLowerLeg[0]>last||s===0);last=l.leftLowerLeg[0];assert.ok(Math.abs(l.leftUpperLeg[0]+l.leftLowerLeg[0]+l.leftFoot[0])<1e-9,'the feet stay flat');}
 const stretch=landingPose(-.15);assert.ok(stretch.head[0]<0&&stretch.leftFoot[0]>0&&stretch.leftLowerLeg[0]===0);
 const base=new Map([['head',new Quaternion().setFromAxisAngle(new Vector3(0,1,0),.2)]]),settled=settlePose(new Map(base),.5);
 assert.ok(settled.get('leftLowerLeg').angleTo(new Quaternion())>20*D);assert.ok(settled.get('head').angleTo(base.get('head'))>2*D);
 assert.equal(settlePose(base,0),base);
});

test('startle, alarm, sway and breath layer onto the variant; the swing variant swings the legs',()=>{
 const plain=heldTable(hold('dangling'),0,{pitch:30}),startle=heldTable(hold('grabbed',{startle:1}),0,{pitch:30});
 assert.ok(startle.leftUpperLeg[0]<plain.leftUpperLeg[0]-9&&startle.head[0]>plain.head[0]&&startle.leftShoulder[2]>plain.leftShoulder[2]);
 const alarm=heldTable(hold('carried',{alarm:1}),0,{pitch:30});assert.equal(alarm.leftUpperArm[2]-plain.leftUpperArm[2],20);assert.equal(alarm.rightUpperArm[2]-plain.rightUpperArm[2],-20);
 const swayA=heldTable(hold('dangling',{sway:1}),.5,{pitch:30}),swayB=heldTable(hold('dangling',{sway:1}),1.6,{pitch:30});assert.ok(Math.abs(swayA.leftUpperLeg[0]-swayB.leftUpperLeg[0])>2,'the idle sway moves the legs');
 const breath=heldTable({...hold('dangling'),breath:{phase:Math.PI/2,depth:1.5}},0,{pitch:30});assert.ok(Math.abs(breath.chest[0]-plain.chest[0]+1.5)<1e-9);assert.ok(breath.leftShoulder[2]>plain.leftShoulder[2]);
 const a=heldTable(hold('dangling',{variant:'swing'}),.3125,{pitch:30}),b=heldTable(hold('dangling',{variant:'swing'}),.9375,{pitch:30});assert.ok(Math.abs(a.leftUpperLeg[0]-b.leftUpperLeg[0])>15,'the playful swing moves the legs 10 degrees each way');
 assert.ok(toQuaternions({hips:[30,0,0]}).get('hips').angleTo(new Quaternion())-30*D<1e-9);
});

test('falling: arms flung out just below level with bent elbows, knees bent unevenly, toes pointed, a flail that moves, and the held tuck gone',()=>{
 const rig=chain();
 for(const t of [0,.1,.3]){
  const table=heldTable({state:'falling',pose:{variant:'tuck',fall:1}},t,{pitch:30});const reach=heldTable({state:'falling',pose:{variant:'reach',fall:1}},t,{pitch:30});for(const bone in table)assert.ok(table[bone].every((v,i)=>Math.abs(v-reach[bone][i])<1e-9),`the variant no longer matters (${bone})`);
  assert.ok(table.hips[0]<0,'no scruff tilt: the back arches a little');rig.apply(heldPose({state:'falling',pose:{variant:'tuck',fall:1}},t,{pitch:30}));
  for(const s of ['left','right']){const arm=rig.dir(s+'UpperArm',s+'LowerArm');assert.ok(Math.abs(arm.x)>.8&&arm.y<0&&arm.y>-.5,`${s} arm out, just below level (${arm.toArray().map(v=>v.toFixed(2))})`);
   assert.ok(rig.at(s+'Hand').y>rig.at(s+'LowerArm').y-.02,`${s} forearm not hanging`);assert.ok(rig.dir(s+'Foot',s+'Toes').y<-.3,'toes pointed');}
  assert.ok(table.rightLowerLeg[0]-table.leftLowerLeg[0]>0,'uneven knees');
 }
 const a=fallPose(0),cycle=Array.from({length:50},(_,i)=>fallPose(i/110).leftUpperArm[2]);assert.ok(Math.max(...cycle)-Math.min(...cycle)>10,'the arms flail');
 for(let t=0;t<1;t+=.01){const f=fallPose(t);assert.ok(f.leftUpperArm[2]<=-24+1e-9&&f.leftUpperArm[2]>=-36-1e-9&&f.rightUpperArm[2]>=24-1e-9,'never raised past 24 degrees below level (skin stretch)');}
 const half=heldTable({state:'falling',pose:{variant:'tuck',fall:.5}},0,{pitch:30}),tuck=heldTable({state:'dangling',pose:{variant:'tuck'}},0,{pitch:30});assert.ok(Math.abs(half.hips[0]-(tuck.hips[0]+a.hips[0])/2)<1e-9,'fall weights blend from the held pose');
});
