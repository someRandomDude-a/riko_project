import test from 'node:test';
import assert from 'node:assert/strict';
import {Group,Bone,Vector3,Quaternion,PerspectiveCamera} from 'three';
import {AvatarPickup} from './avatar_pickup.mjs';
import {boneCatalog} from './avatar_secondary.mjs';
import {normalizeProfile} from './avatar_studio_settings.mjs';
import {heldPose} from './avatar_held_pose.mjs';
import {flipVRM0} from './motion_engine.mjs';

const dt=1/60,D=Math.PI/180,WPP=.0025;  // metres per pixel at the avatar
// A canonical VRM 1.0 skeleton (faces +z, her left is +x); raw and normalized bones are the same nodes, so humanoid.update is a no-op.
function rig({version='1',hair=true}={}){
 const scene=new Group(),nodes=new Map(),side=version==='0'?-1:1;
 const add=(name,parent,x,y,z)=>{const bone=new Bone();bone.name=name;bone.position.set(x,y,z);(parent?nodes.get(parent):scene).add(bone);nodes.set(name,bone);return bone;};
 add('hips',null,0,.9,0);add('spine','hips',0,.1,0);add('chest','spine',0,.1,0);add('upperChest','chest',0,.1,0);add('neck','upperChest',0,.12,0);add('head','neck',0,.1,0);
 for(const [s,x] of [['left',side],['right',-side]]){
  add(s+'Shoulder','upperChest',.05*x,.08,0);add(s+'UpperArm',s+'Shoulder',.1*x,0,0);add(s+'LowerArm',s+'UpperArm',.25*x,0,0);add(s+'Hand',s+'LowerArm',.25*x,0,0);
  add(s+'UpperLeg','hips',.08*x,-.05,0);add(s+'LowerLeg',s+'UpperLeg',0,-.4,0);add(s+'Foot',s+'LowerLeg',0,-.4,0);add(s+'Toes',s+'Foot',0,-.03,.1*side);
 }
 if(hair){const strand=new Bone();strand.name='hair';strand.position.set(.05,.1,-.05);strand.quaternion.setFromAxisAngle(new Vector3(1,0,0),.3);nodes.get('head').add(strand);}
 if(version==='0')scene.rotation.y=Math.PI;
 const humanoid={normalizedHumanBones:Object.fromEntries([...nodes].map(([name,node])=>[name,{node}])),getRawBoneNode:name=>nodes.get(name)||null,getNormalizedBoneNode:name=>nodes.get(name)||null};
 const vrm={scene,meta:{metaVersion:version},humanoid};scene.updateMatrixWorld(true);
 return {scene,nodes,vrm,catalog:boneCatalog(vrm,'fixture')};
}
const world=node=>node.getWorldPosition(new Vector3());
const downAngle=(a,b)=>Math.acos(Math.max(-1,Math.min(1,-world(b).sub(world(a)).normalize().y)))/D;
// One renderer frame: restore, MotionEngine's write (the tuck, flipped for VRM 0.x), the carry rig, (humanoid.update), the anchor.
function frame(pickup,{nodes,vrm},settings,motion={},pose=heldPose({pose:{variant:'tuck'}},0,{pitch:30})){
 pickup.restore();const vrm0=vrm.meta.metaVersion==='0';
 for(const [name,node] of nodes)if(name!=='hair')node.quaternion.copy(pose.has(name)?(vrm0?flipVRM0(pose.get(name)):pose.get(name)):new Quaternion());
 pickup.pose(settings,dt,{worldPerPixel:WPP,...motion});pickup.anchor(settings);
}
// The design's carry: accelerate right at 6 m/s^2 for 0.25 s, cruise 0.5 s, stop at -15 m/s^2 over 0.1 s; window motion in px.
function carry(t){const a=t<.25?6:t<.75?0:t<.85?-15:0,v=t<.25?6*t:t<.75?1.5:t<.85?1.5-15*(t-.75):0;return {ax:a/WPP,vx:v/WPP};}

test('the scruff anchor holds the upper chest on its grab point every frame while she swings, and only the hips and limbs turn',()=>{
 const fixture=rig(),{nodes,catalog}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,catalog);
 const before=new Map(catalog.entries.filter(e=>!e.human).map(e=>[e.node,e.node.quaternion.clone()]));assert.ok(before.size>=1);
 pickup.start('leftFoot',{x:0,y:0},null,{style:'scruff'});const anchor=pickup.anchorPoint.clone();
 assert.ok(world(nodes.get('upperChest')).distanceTo(anchor)<1e-9,'S is the upper chest');
 let peak=0;for(let i=0;i<180;i++){frame(pickup,fixture,settings,carry(i*dt));peak=Math.max(peak,Math.abs(pickup.angle));
  assert.ok(world(nodes.get('upperChest')).distanceTo(anchor)<1e-6,`frame ${i}: S is ${world(nodes.get('upperChest')).distanceTo(anchor)} m off its anchor`);
  for(const [node,q] of before)assert.ok(node.quaternion.equals(q),`the pickup turned ${node.name}`);
  assert.ok(Math.abs(pickup.angle)<=settings.swingLimit*D+1e-9);}
 assert.ok(peak>15*D,`the swing peaked at ${(peak/D).toFixed(1)} degrees`);
});

test('a constant drag settles to the air lean only, and a rightward start makes her body lag left',()=>{
 const fixture=rig(),settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
 for(let i=0;i<300;i++)frame(pickup,fixture,settings,{vx:600,ax:0});
 assert.ok(Math.abs(pickup.angle)<=settings.airLean*D+.5*D,`${(pickup.angle/D).toFixed(2)} degrees at a steady 600 px/s`);
 assert.ok(Math.abs(pickup.angle/D+6)<.6,'leaning back 6 degrees (0.6 of the 10-degree air lean) against the drag');
 const lag=rig(),again=new AvatarPickup(lag.vrm,lag.catalog);again.start('chest',{x:0,y:0},null);
 for(let i=0;i<15;i++)frame(again,lag,settings,{ax:6/WPP,vx:6*i*dt/WPP});
 assert.ok(again.angle<-5*D,`swing ${(again.angle/D).toFixed(1)} degrees`);
 const feet=world(lag.nodes.get('leftFoot')).add(world(lag.nodes.get('rightFoot'))).multiplyScalar(.5);assert.ok(feet.x<again.anchorPoint.x-.05,'her legs trail to the left');
});

test('after a hard stop she swings past centre two to four times, never beyond the soft limit, and settles',()=>{
 const fixture=rig(),settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
 const after=[];for(let i=0;i<300;i++){const t=i*dt;frame(pickup,fixture,settings,carry(t));if(t>=.85)after.push(pickup.angle/D);assert.ok(Math.abs(pickup.angle)<=settings.swingLimit*D);}
 const peaks=after.filter((v,i)=>i>0&&i<after.length-1&&Math.abs(v)>=Math.abs(after[i-1])&&Math.abs(v)>Math.abs(after[i+1])&&Math.abs(v)>3);
 assert.ok(peaks.length>=2&&peaks.length<=4,`peaks ${peaks.map(v=>v.toFixed(1)).join(' ')}`);
 assert.ok(peaks[0]>25&&peaks[0]<=35,`first peak ${peaks[0].toFixed(1)} (about 30)`);assert.ok(peaks.every((v,i)=>!i||Math.sign(v)!==Math.sign(peaks[i-1])),'peaks alternate');
 assert.ok(after.slice(-20).every(v=>Math.abs(v)<2),'below 2 degrees by 4.2 s');
});

test('during a 25-degree swing the legs catch up to within 15 degrees of world-down, the head stays steadier than the body, and the hit-stop mutes the first 0.08 s',()=>{
 const fixture=rig(),{nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
 frame(pickup,fixture,settings,{ax:-8/WPP});frame(pickup,fixture,settings,{ax:-8/WPP});assert.equal(pickup.angle,0,'no swing inside the hit-stop');
 const a=9.8*Math.tan(35*Math.atanh(25/35)*D);  // the steady acceleration whose displayed swing is 25 degrees
 for(let i=0;i<240;i++)frame(pickup,fixture,settings,{ax:-a/WPP});
 assert.ok(Math.abs(pickup.angle/D-25)<.5,`swing ${(pickup.angle/D).toFixed(2)}`);
 for(const s of ['left','right'])assert.ok(downAngle(nodes.get(s+'UpperLeg'),nodes.get(s+'LowerLeg'))<15,`${s} thigh ${downAngle(nodes.get(s+'UpperLeg'),nodes.get(s+'LowerLeg')).toFixed(1)} degrees from down`);
 const roll=node=>{const q=node.getWorldQuaternion(new Quaternion()),up=new Vector3(0,1,0).applyQuaternion(q);return Math.atan2(-up.x,up.y)/D;};
 assert.ok(Math.abs(roll(nodes.get('head')))<.4*25,`head roll ${roll(nodes.get('head')).toFixed(1)} against the body's 25`);
 assert.ok(Math.abs(roll(nodes.get('spine'))-25)<1,'the spine rolls with the hips as one piece');
});

test('the legs follow the swing a beat late: they first swing with the hips, then overshoot toward world-down',()=>{
 const fixture=rig(),settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
 const legs=[];for(let i=0;i<240;i++){frame(pickup,fixture,settings,{ax:4/WPP});legs.push(pickup.legs.x);}
 const final=-.85*pickup.angle;assert.ok(Math.abs(legs.at(-1)-final)<.2*D,'they end on the counter-swing');
 assert.ok(legs[12]/final<.3,`after 0.2 s the legs have followed only ${(legs[12]/final*100).toFixed(0)}% of it`);
 assert.ok(Math.max(...legs.map(v=>v/final))>1.1,'and overshoot on the way');
});

test('lifted fast she straightens and points her toes; lowered fast her knees come up; a knee never bends backwards',()=>{
 const knee=(pickup,fixture,ay)=>{for(let i=0;i<20;i++)frame(pickup,fixture,normalizeProfile({}).pickup,{ay});const n=fixture.nodes,thigh=world(n.get('leftLowerLeg')).sub(world(n.get('leftUpperLeg'))),shin=world(n.get('leftFoot')).sub(world(n.get('leftLowerLeg')));return thigh.angleTo(shin)/D;};
 const make=()=>{const fixture=rig(),pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);return [pickup,fixture];};
 const rest=knee(...make(),0),up=knee(...make(),-9.8/WPP),down=knee(...make(),9.8/WPP);
 assert.ok(up<rest-3&&up>=-1e-6,`lifting: knee ${up.toFixed(1)} against ${rest.toFixed(1)}`);assert.ok(down>rest+3,`lowering: knee ${down.toFixed(1)}`);
 const straight=new Map([['leftLowerLeg',new Quaternion()]]),[pickup,fixture]=make();
 for(let i=0;i<30;i++){frame(pickup,fixture,normalizeProfile({}).pickup,{ay:-30/WPP},straight);assert.ok(fixture.nodes.get('leftLowerLeg').quaternion.angleTo(new Quaternion())<1e-9,'a straight knee is never straightened further');}
});

test('the scruff slide reaches project(G) - project(S) in 0.25-0.6 s; the bone style anchors the grabbed bone and never slides',()=>{
 const camera=new PerspectiveCamera(30,480/720,.01,100);camera.position.set(0,.9,4);camera.lookAt(0,.9,0);camera.updateMatrixWorld(true);
 for(const [bone,longest] of [['leftFoot',.6],['head',.45],['upperChest',.25]]){
  const fixture=rig(),settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog),g=world(fixture.nodes.get(bone)).project(camera),s=world(fixture.nodes.get('upperChest')).project(camera);
  pickup.start(bone,{x:0,y:0},camera,{style:'scruff',pointer:{x:g.x,y:g.y}});assert.ok(pickup.slideTime>=.25&&pickup.slideTime<=longest,`${bone}: ${pickup.slideTime.toFixed(2)} s`);
  let t=0;while(t<pickup.slideTime-1e-9){frame(pickup,fixture,settings);t+=dt;}frame(pickup,fixture,settings);
  const slide=pickup.slide();assert.ok(Math.abs(slide.x-(g.x-s.x))<1e-9&&Math.abs(slide.y-(g.y-s.y))<1e-9,`${bone} slide ${JSON.stringify(slide)}`);
 }
 const fixture=rig(),pickup=new AvatarPickup(fixture.vrm,fixture.catalog),foot=fixture.nodes.get('leftFoot');pickup.start('leftFoot',{x:0,y:0},camera,{style:'bone',pointer:{x:.5,y:-.5}});
 const anchor=world(foot);for(let i=0;i<40;i++){frame(pickup,fixture,normalizeProfile({}).pickup);assert.deepEqual(pickup.slide(),{x:0,y:0});assert.ok(world(foot).distanceTo(anchor)<1e-6);}
});

test('after release the translation eases out without a pop, the feet plant, the camera eases back and the rig ends with the root exactly restored',()=>{
 const camera=new PerspectiveCamera(30,480/720,.01,100);camera.position.set(0,.9,2.2);camera.lookAt(0,.9,0);camera.updateMatrixWorld(true);const rest=camera.position.clone();
 const fixture=rig(),{scene,nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog),root=scene.position.clone(),hips=nodes.get('hips').position.clone();
 pickup.start('chest',{x:0,y:0},camera);
 const offsets=[],cameras=[];for(let i=0;i<60;i++){frame(pickup,fixture,settings,carry(i*dt));offsets.push(nodes.get('hips').position.clone());assert.ok(scene.position.equals(root),'the anchor moves the hips, never the scene');}
 assert.ok(camera.position.z>rest.z+.05,'the safety net zooms the camera out to keep her in frame');
 pickup.release();let frames=0;
 while(pickup.active&&frames<600){frame(pickup,fixture,settings);offsets.push(nodes.get('hips').position.clone());cameras.push(camera.position.z);frames++;}
 const step=i=>offsets[i].distanceTo(offsets[i-1]);
 assert.ok(Math.abs(step(60)-step(59))<.002,`release changed the translation's per-frame step by ${((step(60)-step(59))*1000).toFixed(2)} mm`);
 assert.ok(Math.max(...offsets.slice(60).map((_,i)=>step(60+i)))<.03,'no teleport: under 3 cm a frame while the hold hands over to the feet');
 assert.ok(cameras.every((z,i)=>!i||z<=cameras[i-1]+1e-9),'the camera only eases back');assert.ok(cameras.slice(2).every((z,i)=>Math.abs(z-2*cameras[i+1]+cameras[i])<.01),'without a snap: it starts from rest and its speed changes smoothly');
 assert.equal(pickup.active,false);assert.ok(frames/60>=settings.recoverySeconds&&frames/60<settings.recoverySeconds+1.5,`ended after ${(frames/60).toFixed(2)} s`);
 pickup.restore();assert.ok(nodes.get('hips').position.distanceTo(hips)<1e-12);assert.ok(scene.position.equals(root));assert.ok(camera.position.distanceTo(rest)<1e-12);
});

test('released, the feet plant on the line they stood on while the settle bends her knees',()=>{
 const fixture=rig(),{nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);
 const toe=()=>Math.min(world(nodes.get('leftToes')).y,world(nodes.get('rightToes')).y),standing=toe();
 pickup.start('chest',{x:0,y:0},null);for(let i=0;i<30;i++)frame(pickup,fixture,settings);pickup.release();
 const squat=new Map([...heldPose({pose:{variant:'tuck'}},0,{pitch:0})].map(([b])=>[b,new Quaternion()]));
 for(const [bone,degrees] of [['leftUpperLeg',-22],['rightUpperLeg',-22],['leftLowerLeg',45],['rightLowerLeg',45],['leftFoot',-23],['rightFoot',-23]])squat.set(bone,new Quaternion().setFromAxisAngle(new Vector3(1,0,0),degrees*D));
 for(let i=0;i<60;i++)frame(pickup,fixture,settings,{},squat);
 assert.ok(Math.abs(toe()-standing)<.001,`toes ${((toe()-standing)*1000).toFixed(2)} mm off the floor line`);
 assert.ok(world(nodes.get('hips')).y<.9-.03,'the knee bend drops her hips instead of lifting her feet');
});

test('the opt-in ragdoll triggers once past a threshold, makes her limp, and recovers once on release or after 1 s slow',()=>{
 const fixture=rig(),events=[],settings=normalizeProfile({pickup:{ragdollEnabled:true}}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog,kind=>events.push(kind));
 pickup.start('chest',{x:0,y:0},null);pickup.move({x:100,y:0},1,settings);assert.deepEqual(events,[]);
 pickup.move({x:500,y:0},1,settings);pickup.move({x:600,y:0},1,settings);assert.deepEqual(events,['ragdoll']);assert.equal(pickup.ragdoll,true);
 for(let i=0;i<30;i++)frame(pickup,fixture,settings,{vx:2000});assert.deepEqual(events,['ragdoll'],'still fast: still limp');
 for(let i=0;i<66;i++)frame(pickup,fixture,settings,{vx:0});assert.deepEqual(events,['ragdoll','recover']);assert.equal(pickup.ragdoll,false);
 pickup.move({x:620,y:0},1,settings);assert.deepEqual(events,['ragdoll','recover'],'the distance threshold re-arms from where she recovered');
 pickup.move({x:5000,y:0},.01,settings);pickup.release();assert.deepEqual(events,['ragdoll','recover','ragdoll','recover']);
 const quiet=[],off=new AvatarPickup(fixture.vrm,fixture.catalog,kind=>quiet.push(kind));off.start('chest',{x:0,y:0},null);
 off.move({x:5000,y:0},.001,{...settings,ragdollEnabled:false});assert.deepEqual(quiet,[]);
 off.pose({...settings,enabled:false},.1);assert.equal(off.active,false);
});

test('a VRM 0.x root turned by PI keeps facing the camera while held and after recovery',()=>{
 const fixture=rig({version:'0'}),{scene,nodes}=fixture,pickup=new AvatarPickup(fixture.vrm,fixture.catalog),settings=normalizeProfile({}).pickup;
 const facing=()=>new Vector3(0,0,-1).applyQuaternion(scene.quaternion).z;  // a VRM 0.x model faces -z, so turned by PI it faces +z
 const hips=()=>new Vector3(0,0,-1).applyQuaternion(nodes.get('hips').getWorldQuaternion(new Quaternion())).z;
 const step=(time,held)=>{pickup.restore();scene.rotation.y=Math.PI+Math.sin(time*.7)*.025;scene.updateMatrixWorld(true);frame(pickup,fixture,settings,held?carry(time%1.2):{});};
 scene.rotation.y=Math.PI;pickup.start('chest',{x:0,y:0},null);
 for(let i=0;i<240;i++){step(i/60,true);assert.ok(facing()>0,`frame ${i} faces away while held`);assert.ok(hips()>.7,`frame ${i}: her body turned away (${hips().toFixed(2)})`);}
 pickup.release();for(let i=240;i<600&&pickup.active;i++){step(i/60,false);assert.ok(facing()>0,`frame ${i} faces away while recovering`);assert.ok(hips()>.7);}
 assert.equal(pickup.active,false);pickup.restore();scene.rotation.y=Math.PI;assert.ok(facing()>.99);  // the renderer's next yaw write leaves her facing front
});

test('a shake at the limbs\' own frequency never folds a hip, shoulder or neck: the lag rolls ease into soft limits',()=>{
 const fixture=rig(),{nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog),pose=heldPose({pose:{variant:'tuck'}},0,{pitch:30});pickup.start('chest',{x:0,y:0},null);
 const roll=bone=>nodes.get(bone).quaternion.angleTo(pose.get(bone)||new Quaternion())/D;  // the pickup premultiplies its roll onto the written pose
 const worst={leftUpperLeg:0,rightUpperArm:0,neck:0,head:0};let spring=0;
 for(let i=0;i<360;i++){const t=i*dt,w=2*Math.PI*settings.limbFrequency;frame(pickup,fixture,settings,{ax:Math.sin(w*t)*40/WPP,vx:-Math.cos(w*t)*40/w/WPP},pose);spring=Math.max(spring,Math.abs(pickup.legs.x)/D);for(const bone in worst)worst[bone]=Math.max(worst[bone],roll(bone));}
 assert.ok(spring>28,`the shake drives the leg spring to ${spring.toFixed(0)} degrees`);
 assert.ok(worst.leftUpperLeg<=25+1e-6&&worst.rightUpperArm<=45+1e-6&&worst.neck<=.4*30+1e-6&&worst.head<=.6*30+1e-6,`rolls past their limits: ${JSON.stringify(Object.fromEntries(Object.entries(worst).map(([k,v])=>[k,+v.toFixed(1)])))}`);
});

test('set down mid-swing she is upright inside the 0.35 s handover, never rocks back, and her planted feet never slide',()=>{
 const fixture=rig(),{nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
 let i=0,last=0;for(;i<300;i++){frame(pickup,fixture,settings,carry(i*dt));if(i*dt>.85&&Math.abs(pickup.angle)<Math.abs(last))break;last=pickup.angle;}  // the first peak after the hard stop
 assert.ok(Math.abs(pickup.angle)>25*D,`released at a ${(pickup.angle/D).toFixed(1)}-degree peak`);
 pickup.release();const toes=()=>world(nodes.get('leftToes')).add(world(nodes.get('rightToes'))).multiplyScalar(.5),angles=[],feet=[];
 for(let k=0;k<150;k++){frame(pickup,fixture,settings,{});angles.push(Math.abs(pickup.angle)/D);feet.push(toes());}
 const handover=Math.round(.35/dt);assert.ok(angles[handover-1]<1,`still ${angles[handover-1].toFixed(2)} degrees at 0.35 s`);
 assert.ok(angles.slice(angles.findIndex(a=>a<2)).every(a=>a<2),'once upright she stays upright');
 assert.ok(Math.max(...[pickup.legs,pickup.arms,pickup.head].map(s=>Math.abs(s.x)))<1*D,'the limb lag has settled too');
 const slide=Math.max(...feet.slice(handover).map(p=>Math.abs(p.x-feet[handover].x)));assert.ok(slide<.002,`the planted feet slid ${(slide*1000).toFixed(1)} mm`);
 let jerk=0;for(let k=2;k<handover;k++)jerk=Math.max(jerk,Math.abs(angles[k]-2*angles[k-1]+angles[k-2]));assert.ok(jerk<2,`no jolt: the swing's per-frame step changes at most ${jerk.toFixed(2)} degrees`);
});

test('an arm hanging at her side rolls in toward her body at most 6 degrees; held forward it gets the full 45',()=>{
 const a=9.8*Math.tan(35*Math.atanh(30/35)*D);  // a steady 30-degree swing: the arm lag wants 27 degrees
 for(const [label,arms,inward] of [['hanging',{leftUpperArm:[0,0,-80],rightUpperArm:[0,0,80]},6],['forward',{leftUpperArm:[0,-90,0],rightUpperArm:[0,90,0]},45]]){
  const fixture=rig(),{nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
  const pose=new Map(Object.entries(arms).map(([bone,[x,y,z]])=>[bone,new Quaternion().setFromAxisAngle(new Vector3(x?1:0,y?1:0,z?1:0).normalize(),(x||y||z)*D)]));
  for(let i=0;i<300;i++)frame(pickup,fixture,settings,{ax:-a/WPP},pose);
  const roll=bone=>nodes.get(bone).quaternion.angleTo(pose.get(bone))/D;
  assert.ok(pickup.arms.x<-25*D,`the lag drives the arm spring to ${(pickup.arms.x/D).toFixed(1)} (her left arm, on screen right, rolls in)`);
  assert.ok(roll('leftUpperArm')<=inward+1e-6&&roll('leftUpperArm')>(label==='hanging'?4:20),`${label}: the inward arm rolls ${roll('leftUpperArm').toFixed(1)} degrees`);
  assert.ok(roll('rightUpperArm')>20,`${label}: the outward arm keeps its lag (${roll('rightUpperArm').toFixed(1)})`);
 }
});

test('thrown, nothing drives the swing and it relaxes; the shoe tips hand over to the line they stood on in the air; land() plants them and the recovery counts from the contact',()=>{
 const fixture=rig(),{nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);
 // the lowest shoe tip: the toes bone carried on along the foot by 40% (pointed toes put it below the toes bone, so she lands tip first)
 const tip=side=>{const t=world(nodes.get(side+'Toes')).y,f=world(nodes.get(side+'Foot')).y;return t+.4*(t-f);},toe=()=>Math.min(tip('left'),tip('right')),standing=toe();
 pickup.start('chest',{x:0,y:0},null);let i=0;for(;i<300;i++){frame(pickup,fixture,settings,carry(i*dt));if(i*dt>.9&&Math.abs(pickup.angle)>20*D)break;}
 assert.ok(Math.abs(pickup.angle)>20*D,'thrown mid-swing');
 pickup.release({flight:true});assert.equal(pickup.flying,true);
 const angles=[];for(let k=0;k<60;k++){frame(pickup,fixture,settings,{ax:20/WPP,vx:3000,ay:2500});angles.push(Math.abs(pickup.angle)/D);}
 assert.ok(angles.at(-1)<1.5,`the swing relaxed to ${angles.at(-1).toFixed(2)} degrees in 1 s whatever the window does`);
 assert.ok(Math.abs(pickup.lift.x)<1e-9,'no lift reaction in free fall');
 // the feet are held clear of the line by half their remaining sideways swing (as after a set-down), so the contact is continuous
 assert.ok(Math.abs(toe()-standing)<.01,`in the air her shoe tips are ${((toe()-standing)*1000).toFixed(1)} mm from the line they stood on`);
 for(let k=0;k<60;k++)frame(pickup,fixture,settings,{});assert.ok(Math.abs(toe()-standing)<.003,`and ${((toe()-standing)*1000).toFixed(1)} mm once the legs have settled`);
 for(let k=0;k<60;k++)frame(pickup,fixture,settings,{});assert.equal(pickup.active,true,'still flying 3 s later: the rig waits for the contact');
 pickup.land();assert.equal(pickup.flying,false);
 const squat=new Map([['leftUpperLeg',-22],['rightUpperLeg',-22],['leftLowerLeg',45],['rightLowerLeg',45],['leftFoot',-23],['rightFoot',-23]].map(([bone,degrees])=>[bone,new Quaternion().setFromAxisAngle(new Vector3(1,0,0),degrees*D)]));
 for(let k=0;k<30;k++)frame(pickup,fixture,settings,{},squat);
 assert.ok(Math.abs(toe()-standing)<.001,`landed: shoe tips ${((toe()-standing)*1000).toFixed(2)} mm off the floor line`);assert.ok(world(nodes.get('hips')).y<.9-.03,'the knee bend drops her hips');
 let frames=30;while(pickup.active&&frames<600){frame(pickup,fixture,settings,{});frames++;}
 assert.ok(frames*dt>=settings.recoverySeconds&&frames*dt<settings.recoverySeconds+1.5,`the rig ended ${(frames*dt).toFixed(2)} s after the contact`);
});

test('the head look turns the head toward a world point by what the eyes cannot reach, at most 45 degrees of yaw and 30 of pitch, trailing the eyes',()=>{
 const none=new Map();  // every bone at rest, so the look is all there is
 const facing=(fixture,version='1')=>{const q=fixture.nodes.get('head').getWorldQuaternion(new Quaternion()),f=new Vector3(0,0,version==='0'?-1:1).applyQuaternion(q);return {yaw:Math.atan2(f.x,f.z)/D,pitch:Math.atan2(-f.y,Math.hypot(f.x,f.z))/D};};
 const settle=(look,version='1',frames=150,reach)=>{const fixture=rig({version}),settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
  const seen=[];for(let i=0;i<frames;i++){frame(pickup,fixture,settings,{look:look(fixture),eyeReach:reach},none);seen.push(facing(fixture,version));}return {fixture,seen,last:seen.at(-1)};};
 const head=fixture=>world(fixture.nodes.get('head'));
 let r=settle(f=>head(f).add(new Vector3(10,0,1)));assert.ok(Math.abs(r.last.yaw-45)<1,`far to her left: yaw ${r.last.yaw.toFixed(2)}`);
 r=settle(f=>head(f).add(new Vector3(0,-10,1)));assert.ok(Math.abs(r.last.pitch-30)<1,`far below: pitch ${r.last.pitch.toFixed(2)}`);
 r=settle(f=>head(f).add(new Vector3(Math.tan(30*D),0,1)));assert.ok(Math.abs(r.last.yaw-18)<.5,`30 degrees away the head takes 18 and the eyes 12 (${r.last.yaw.toFixed(2)})`);
 assert.ok(r.seen[5].yaw<.5*18,`after 0.1 s the head has turned only ${r.seen[5].yaw.toFixed(1)} degrees: it trails the eyes`);
 r=settle(f=>head(f).add(new Vector3(Math.tan(8*D),0,1)));assert.ok(Math.abs(r.last.yaw)<1e-6,'inside the eyes\' reach the head stays');
 r=settle(f=>head(f).add(new Vector3(Math.tan(30*D),0,1)),'1',150,{yaw:20,pitch:10});assert.ok(Math.abs(r.last.yaw-10)<.5,'eyeReach comes from the model');
 const vrm0=settle(()=>new Vector3(5,1.42,1),'0');assert.ok(vrm0.last.yaw>30,`a VRM 0.x head turns toward the point too (${vrm0.last.yaw.toFixed(1)})`);
 const back=settle(f=>head(f).add(new Vector3(10,0,1))),fixture=back.fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
 for(let i=0;i<150;i++)frame(pickup,fixture,settings,{look:null},none);assert.ok(Math.abs(facing(fixture).yaw)<.1,'no look: back to the front');
});

test('looking at the viewer while the body swings 25 degrees, the head\'s world roll stays within 0.4 of the swing',()=>{
 const fixture=rig(),{nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
 const a=9.8*Math.tan(35*Math.atanh(25/35)*D),viewer=new Vector3(0,1.2,3);
 for(let i=0;i<240;i++)frame(pickup,fixture,settings,{ax:-a/WPP,look:viewer});
 const roll=node=>{const q=node.getWorldQuaternion(new Quaternion()),up=new Vector3(0,1,0).applyQuaternion(q);return Math.atan2(-up.x,up.y)/D;};
 assert.ok(Math.abs(pickup.angle/D-25)<.5);assert.ok(Math.abs(roll(nodes.get('head')))<.4*25,`head roll ${roll(nodes.get('head')).toFixed(1)} against the body's 25`);
 const f=new Vector3(0,0,1).applyQuaternion(nodes.get('head').getWorldQuaternion(new Quaternion())),toViewer=viewer.clone().sub(world(nodes.get('head'))).normalize();
 assert.ok(f.angleTo(toViewer)/D<15,`she faces the viewer within the eyes' reach (${(f.angleTo(toViewer)/D).toFixed(1)} degrees off)`);
});

test('after a flight nothing drives her: the limbs relax onto the body in the air, the floor\'s stop at the contact never kicks the swing, and a short drop hands over to the feet by the contact',()=>{
 const fixture=rig(),{nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);
 const tip=side=>{const t=world(nodes.get(side+'Toes')).y,f=world(nodes.get(side+'Foot')).y;return t+.4*(t-f);},toe=()=>Math.min(tip('left'),tip('right')),standing=toe();
 pickup.start('chest',{x:0,y:0},null);for(let i=0;i<300;i++){frame(pickup,fixture,settings,carry(i*dt));if(i*dt>.9&&Math.abs(pickup.angle)>20*D)break;}
 pickup.release({flight:true,seconds:.5});assert.ok(Math.abs(pickup.airFrequency-2)<1e-9,'it relaxes at 1 / 0.5 s');assert.equal(pickup.handover,.35);
 for(let k=0;k<30;k++)frame(pickup,fixture,settings,{ax:0,vx:1500});
 assert.ok(Math.abs(pickup.angle)<2*D,`the swing is ${(pickup.angle/D).toFixed(2)} degrees at the contact`);
 assert.ok(Math.abs(pickup.legs.x)<2*D&&Math.abs(pickup.arms.x)<2*D,`the limbs relaxed onto the body (legs ${(pickup.legs.x/D).toFixed(2)}, arms ${(pickup.arms.x/D).toFixed(2)}), not toward world-down`);
 pickup.land();const angles=[];frame(pickup,fixture,settings,{ax:-1500*60,vx:0,ay:-2000*60});  // the window stops dead at the contact
 for(let k=0;k<30;k++){frame(pickup,fixture,settings,{});angles.push(Math.abs(pickup.angle)/D);}
 assert.ok(Math.max(...angles)<2,`no kick: at most ${Math.max(...angles).toFixed(2)} degrees after the contact`);assert.ok(Math.abs(pickup.lift.x)<1e-9,'and no lift reaction');
 const short=rig(),hold=new AvatarPickup(short.vrm,short.catalog),shortTip=side=>{const t=world(short.nodes.get(side+'Toes')).y,f=world(short.nodes.get(side+'Foot')).y;return t+.4*(t-f);},shortToe=()=>Math.min(shortTip('left'),shortTip('right')),rest=shortToe();
 hold.start('chest',{x:0,y:0},null);for(let k=0;k<30;k++)frame(hold,short,settings,{ay:-9000});hold.release({flight:true,seconds:.06});assert.equal(hold.handover,.1);assert.equal(hold.airFrequency,6);
 hold.land();for(let k=0;k<5;k++)frame(hold,short,settings);  // it touched down at once: the hand's hold lets go within 0.08 s of the contact
 assert.ok(Math.abs(shortToe()-rest)<1e-6,`a 0.06 s drop: the shoe tips are ${((shortToe()-rest)*1000).toFixed(3)} mm off the line 0.08 s after the contact`);
 assert.ok(Math.abs(toe()-standing)<.003);
});

test('the head look aims from where the head shows, with the anchor\'s translation the normalized rig never gets',()=>{
 const fixture=rig(),pickup=new AvatarPickup(fixture.vrm,fixture.catalog);pickup.start('chest',{x:0,y:0},null);
 const head=fixture.nodes.get('head'),point=world(head).add(new Vector3(0,.3,1));
 assert.ok(Math.abs(pickup.aim(head,point).pitch-Math.atan2(-.3,1))<1e-9,'30 cm up at 1 m: 17 degrees up');
 pickup.offset.set(0,.3,0);assert.ok(Math.abs(pickup.aim(head,point).pitch)<1e-9,'the head shows 30 cm higher: the point is level with it');
});

test('the plant puts the shoe tip lowest on screen onto the floor line seen through the camera: feet swung toward the camera never dip below it',()=>{
 const camera=new PerspectiveCamera(30,480/720,.01,100);camera.position.set(0,.9,4);camera.lookAt(0,.9,0);camera.updateMatrixWorld(true);
 const fixture=rig(),{nodes}=fixture,settings=normalizeProfile({}).pickup,pickup=new AvatarPickup(fixture.vrm,fixture.catalog);
 const tip=side=>{const t=world(nodes.get(side+'Toes')),f=world(nodes.get(side+'Foot'));return t.clone().add(t.clone().sub(f).multiplyScalar(.4));};
 const lowest=()=>Math.min(...['left','right'].map(side=>tip(side).project(camera).y)),line=lowest(),standing=Math.min(tip('left').y,tip('right').y);
 pickup.start('chest',{x:0,y:0},camera);for(let i=0;i<30;i++)frame(pickup,fixture,settings);pickup.release({flight:true,seconds:.3});
 const lunge=new Map([['leftUpperLeg',-45],['rightUpperLeg',-45],['leftLowerLeg',30],['rightLowerLeg',30]].map(([bone,degrees])=>[bone,new Quaternion().setFromAxisAngle(new Vector3(1,0,0),degrees*D)]));
 for(let i=0;i<20;i++)frame(pickup,fixture,settings,{},lunge);pickup.land();for(let i=0;i<120;i++)frame(pickup,fixture,settings,{},lunge);
 assert.equal(pickup.cameraActive,false);assert.ok(Math.abs(lowest()-line)<1e-6,`on screen the tip is ${((lowest()-line)*360).toFixed(3)} px off the line`);
 const z=Math.max(tip('left').z,tip('right').z);assert.ok(z>.3,'her feet are well in front of her');
 assert.ok(Math.min(tip('left').y,tip('right').y)>standing+.003,'so in the world they stand a little higher, where the camera sees them on the line');
});
