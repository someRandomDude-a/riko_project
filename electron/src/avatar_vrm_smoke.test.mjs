// The avatar on a real model, gated: set RIKO_TEST_VRM to a .vrm file (it is never committed), e.g. from electron/:
//   RIKO_TEST_VRM=../character_files/models/<your model>.vrm node --test src/avatar_vrm_smoke.test.mjs
// Without it every check reports skipped, never passed. The model loads in Node with stubbed textures and runs through the renderer's
// frame order (avatar_renderer.jsx): the hold, MotionEngine, the carry rig with its head look, humanoid.update, the anchor, the springs
// with the window's inertial force, the face and the saccadic eyes (gazeActive, gazeShift). Each check prints
// a one-line summary so the numbers can be compared when tuning. The release runs through the renderer's own Release (avatar_drop.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin,VRMUtils} from '@pixiv/three-vrm';
import {AvatarPickup,shoeTips} from './avatar_pickup.mjs';
import {HoldBehaviour,HoldFace,HELD_STATES,DROP_STATES} from './avatar_hold.mjs';
import {MotionEngine,flipVRM0} from './motion_engine.mjs';
import {AvatarInertia,withSpringForce} from './avatar_inertia.mjs';
import {AvatarFace,registerFaceParts,resolveExpression,MAX_EXPRESSIONS} from './avatar_face.mjs';
import {AvatarEyeGaze,gazeActive} from './avatar_gaze.mjs';
import {boneCatalog} from './avatar_secondary.mjs';
import {normalizeProfile} from './avatar_studio_settings.mjs';
import {cameraDistance} from './avatar_geometry.mjs';
import {keepAvatarDrawn} from './avatar_model.mjs';
import {Release,floorTop} from './avatar_drop.mjs';
import {JOINT_LIMITS} from './avatar_held_pose.mjs';

const FILE=process.env.RIKO_TEST_VRM,skip=FILE?false:'set RIKO_TEST_VRM to a .vrm file to run the real-model smoke test';
const D=Math.PI/180,DT=1/60,W=480,H=720,WORK={left:0,top:0,right:1920,bottom:1040};
const say=line=>console.log('  [vrm smoke] '+line);

async function load(){
 const buf=fs.readFileSync(path.resolve(FILE)),loader=new GLTFLoader();
 loader.register(()=>({name:'stubTextures',loadTexture:()=>Promise.resolve(new THREE.Texture())}));
 loader.register(parser=>new VRMLoaderPlugin(parser));
 const gltf=await new Promise((ok,no)=>loader.parse(buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength),'',ok,no));
 return gltf.userData.vrm;
}
/** The renderer's frame order on the loaded model, with a canvas at `geometry` (overlay px) and the press, drag and release it handles. */
async function stage({geometry={x:700,y:100},profile:raw={},dt=DT}={}){
 const vrm=await load();VRMUtils.rotateVRM0(vrm);keepAvatarDrawn(vrm.scene);
 const scene=new THREE.Scene();scene.add(vrm.scene);const vrm0=vrm.meta?.metaVersion==='0',spin=vrm0?Math.PI:0;
 const catalog=boneCatalog(vrm,FILE);
 const bounds=new THREE.Box3().setFromObject(vrm.scene),size=bounds.getSize(new THREE.Vector3());vrm.scene.position.y-=bounds.min.y;vrm.scene.updateMatrixWorld(true);
 const center=new THREE.Box3().setFromObject(vrm.scene).getCenter(new THREE.Vector3());
 const camera=new THREE.PerspectiveCamera(30,W/H,.01,100),distance=cameraDistance(size,camera.aspect,camera.fov);
 camera.position.set(center.x,center.y,center.z+distance);camera.lookAt(center);camera.updateProjectionMatrix();camera.updateMatrixWorld(true);
 const profile=normalizeProfile(raw),added=registerFaceParts(vrm),face=new AvatarFace(vrm.expressionManager,{random:()=>.999}),holdFace=new HoldFace(face);
 const hold=new HoldBehaviour({random:()=>.5}),motion=new MotionEngine(vrm,{sample:()=>null,retain:()=>{},dispose:()=>{}}),pickup=new AvatarPickup(vrm,catalog),gaze=new AvatarEyeGaze(vrm,{random:()=>.5});
 const inertia=new AvatarInertia(),cursor=new AvatarInertia(),force=new THREE.Vector3(),interaction={held:false,reaction:'',until:0,rule:null,hold:null};
 const raw_=name=>vrm.humanoid.getRawBoneNode(name),world=name=>raw_(name).getWorldPosition(new THREE.Vector3());
 const shoes=shoeTips(vrm.humanoid),front=shoes.reduce((sum,tip)=>sum+tip.z,0)/shoes.length;
 const s={dt,vrm,camera,catalog,profile,added,face,hold,motion,pickup,gaze,interaction,world,time:10,geometry:{...geometry,width:W,height:H},drag:null,out:null,patches:[],wpp:0,frames:0,faceAway:0,limitBreaks:[],interacts:[]};
 const screen=v=>{const p=v.clone().project(camera);return {x:s.geometry.x+(p.x+1)/2*W,y:s.geometry.y+(1-p.y)/2*H};};
 const rest=new THREE.PerspectiveCamera(),footLine=(atRest=false)=>{let view=camera;if(atRest&&pickup.cameraActive){view=rest.copy(camera);view.position.copy(pickup.cameraRest.position);view.quaternion.copy(pickup.cameraRest.quaternion);view.updateMatrixWorld(true);}const p=new THREE.Vector3(center.x,0,front).project(view);return (1-p.y)/2*H;};
 const BOX=['head','neck','hips','leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','leftHand','rightHand','leftFoot','rightFoot','leftToes','rightToes'];
 const bodyBox=()=>{const box={left:W,right:0,top:H};for(const name of BOX){const p=world(name).project(camera),x=(p.x+1)/2*W,y=(1-p.y)/2*H;box.left=Math.min(box.left,x);box.right=Math.max(box.right,x);box.top=Math.min(box.top,y);}const pad=s.wpp>0?.08/s.wpp:0;return {left:box.left-pad,right:box.right+pad,top:box.top-3*pad};};
 s.screen=screen;s.floor=()=>floorTop(WORK,footLine());s.box=bodyBox;s.blinks=[];
 // the screen-lowest shoe tip (overlay px), and every blink with the hold state it came in
 s.soles=()=>Math.max(...shoeTips(vrm.humanoid).map(t=>screen(t).y));const blink=face.blink.bind(face);face.blink=(kind,delay)=>{s.blinks.push({t:s.time,kind,state:hold.state});return blink(kind,delay);};
 const release=new Release({place:(g,final)=>{s.geometry=g;if(final)s.patches.push({...g});},interact:kind=>s.interacts.push(kind),hold,pickup,measure:{area:()=>WORK,foot:footLine,box:bodyBox,worldPerPixel:()=>s.wpp,height:()=>H}});s.releaseCtl=release;
 const slidePx=()=>{const v=s.drag&&pickup.held?pickup.slide():null;return v?{x:Math.round(v.x*W/2),y:Math.round(-v.y*H/2)}:{x:0,y:0};};
 const place=()=>{const d=s.drag,v=slidePx();d.slid=v;s.geometry={...s.geometry,x:d.left+d.point.x-d.x+v.x,y:d.top+d.point.y-d.y+v.y};};
 s.press=bone=>{vrm.scene.updateMatrixWorld(true);const p=screen(world(bone)),n=world(bone).project(camera);s.drag={x:p.x,y:p.y,point:{...p},left:s.geometry.x,top:s.geometry.y,bone,ndc:{x:n.x,y:n.y}};hold.press(s.time);return p;};
 s.grab=()=>{interaction.held=true;hold.grab(s.time,{kind:'body'});pickup.start(s.drag.bone,{x:s.drag.x,y:s.drag.y},camera,{style:profile.pickup.grabStyle,pointer:s.drag.ndc});};
 s.move=(x,y)=>{s.drag.point={x,y};pickup.move({x,y},DT,profile.pickup);place();};
 // pointerup: the renderer's letGo (the cursor tracker takes the slide in, then Release decides, flies or sets her down)
 s.release=()=>{const d=s.drag,v=slidePx(),final={...s.geometry,x:d.left+d.point.x-d.x+v.x,y:d.top+d.point.y-d.y+v.y},velocity={x:cursor.velocity.x,y:cursor.velocity.y};s.drag=null;interaction.held=false;cursor.shift(d.slid?.x||0,d.slid?.y||0);
  return release.letGo(s.time,final,velocity,{slid:d.slid,settings:profile.pickup});};
 const lookPoint=new THREE.Vector3();
 s.frame=()=>{
  s.time+=s.dt;const now=s.time;s.landed=null;s.frames++;
  gaze.restore();pickup.restore();vrm.scene.rotation.y=spin+Math.sin(now*.7)*.025;vrm.scene.position.x=Math.sin(now*.45)*.006;
  if(release.active){const step=release.step(now,s.dt);if(step.landed)s.landed=step;}
  if(s.drag&&pickup.held)place();
  inertia.track(s.geometry.x,s.geometry.y,s.dt);const slid=s.drag?.slid||{x:0,y:0};cursor.track(s.geometry.x-slid.x,s.geometry.y-slid.y,s.dt);
  s.wpp=2*Math.tan(camera.fov*D/2)*camera.position.distanceTo(center)/H;
  const held=hold.step(now,s.dt,{vx:cursor.velocity.x,vy:cursor.velocity.y,ax:cursor.acceleration.x,ay:cursor.acceleration.y},{...profile.reactions,recoverySeconds:profile.pickup.recoverySeconds,landingSquash:profile.pickup.landingSquash});s.out=held;
  interaction.hold=held?.pose?held:null;if(held?.pose)interaction.reaction=held.state;else if(HELD_STATES.has(interaction.reaction)||DROP_STATES.has(interaction.reaction))interaction.reaction='';
  holdFace.apply(held);
  const look=held?.state?held.look:null,head=world('head');
  const lookAt=look==='viewer'?lookPoint.copy(camera.position):look==='holder'?lookPoint.copy(pickup.active?pickup.anchorPoint:head).add(new THREE.Vector3(0,.3,.35)):look==='floor'?lookPoint.copy(head).add(new THREE.Vector3(0,-1,.6)):null;
  motion.update({interaction,settings:{enabled:true},held:{pitch:profile.pickup.scruffPitch,kickAmplitude:profile.reactions.kickAmplitude}},s.dt);
  s.before=new Map(catalog.entries.filter(e=>!e.human).map(e=>[e.node,e.node.quaternion.clone()]));
  pickup.pose(profile.pickup,s.dt,{ax:inertia.acceleration.x,ay:inertia.acceleration.y,vx:inertia.velocity.x,vy:inertia.velocity.y,worldPerPixel:s.wpp,hold:interaction.hold,look:lookAt||(look==='cursor'&&gaze.tracking?gaze.target:null),eyeReach:gaze.reach});
  s.limits();
  vrm.humanoid.update();vrm.nodeConstraintManager?.update();pickup.anchor(profile.pickup);
  s.untouched=[...s.before].every(([node,q])=>node.quaternion.equals(q));
  withSpringForce(vrm.springBoneManager.joints,inertia.force(s.wpp,profile.pickup.bounce,s.dt,force),()=>vrm.springBoneManager.update(s.dt));
  s.expressions=face.step(s.dt,{breathPhase:held?.breath.phase,settings:profile.face});for(const [name,value] of s.expressions)vrm.expressionManager.setValue(name,value);vrm.expressionManager.update();
  vrm.humanoid.getRawBoneNode('head').updateWorldMatrix(true,false);  // as the renderer: the eyes need only the head's chain
  gaze.update({...profile.gaze},s.dt,null,{left:0,top:0,width:W,height:H},camera,null,gazeActive(look,false),lookAt);if(gaze.bigShift)face.gazeShift();
  vrm.expressionManager.update();vrm.scene.updateMatrixWorld(true);  // the frame's one full pass (render() makes it in the renderer)
  const facing=new THREE.Vector3(0,0,vrm0?-1:1).applyQuaternion(vrm.humanoid.getNormalizedBoneNode('head').getWorldQuaternion(new THREE.Quaternion())).z;if(!(facing>0))s.faceAway++;
 };
 // The composed rotation each limb joint gets after the carry rig, in canonical axes: knee and elbow flexion (and its direction), hip.
 s.limits=()=>{const q=name=>{const n=vrm.humanoid.getNormalizedBoneNode(name).quaternion;return vrm0?flipVRM0(n):n.clone();},angle=r=>2*Math.acos(Math.min(1,Math.abs(r.w)))/D,axis=(r,i)=>Math.sign([r.x,r.y,r.z][i]*Math.sign(r.w||1));
  for(const side of ['left','right']){const knee=q(side+'LowerLeg'),elbow=q(side+'LowerArm'),hip=new THREE.Euler().setFromQuaternion(q(side+'UpperLeg'),'XYZ').x/D;
   if(angle(knee)>JOINT_LIMITS.knee[1]+.5||angle(knee)>1&&axis(knee,0)<0)s.limitBreaks.push(`${side} knee ${angle(knee).toFixed(1)} at ${s.time.toFixed(2)}`);
   if(angle(elbow)>JOINT_LIMITS.elbow[1]+.5||angle(elbow)>1&&axis(elbow,1)!==(side==='left'?-1:1))s.limitBreaks.push(`${side} elbow ${angle(elbow).toFixed(1)} at ${s.time.toFixed(2)}`);
   if(-hip>JOINT_LIMITS.hipFlex[1]+.5||hip>JOINT_LIMITS.hipExtend[1]+.5)s.limitBreaks.push(`${side} hip ${hip.toFixed(1)} at ${s.time.toFixed(2)}`);}};
 s.run=(seconds,each)=>{for(let i=0,n=Math.round(seconds/s.dt);i<n;i++){s.frame();each?.(i);}};
 // the two lowest hair spring tips (the twintails on Miku) and their chain roots
 const joints=[...vrm.springBoneManager.joints],tips=joints.filter(j=>!joints.some(k=>k.bone.parent===j.bone)).map(j=>({j,y:j.bone.getWorldPosition(new THREE.Vector3()).y})).sort((a,b)=>a.y-b.y);
 const rootOf=j=>{let b=j.bone;while(b.parent&&joints.some(k=>k.bone===b.parent))b=b.parent;return b;};
 s.hair=tips.slice(0,2).map(t=>({tip:t.j.bone,root:rootOf(t.j)}));
 s.audit=label=>{say(`${label}: ${s.frames} frames, face-away frames ${s.faceAway}, joint-limit breaks ${s.limitBreaks.length}${s.limitBreaks.length?' ('+s.limitBreaks.slice(0,3).join('; ')+')':''}`);assert.equal(s.faceAway,0,`${label}: she faced away`);assert.deepEqual(s.limitBreaks,[],`${label}: joint limits`);};
 return s;
}
const angleBetween=(a,b)=>Math.acos(Math.max(-1,Math.min(1,a.clone().normalize().dot(b.clone().normalize()))))/D;
const downAngle=v=>angleBetween(v,new THREE.Vector3(0,-1,0));
/** A world vector's angle from world-down in the screen plane (degrees; positive toward screen right): a swing keeps its sign. */
const planeAngle=v=>Math.atan2(v.x,-v.y)/D;

test('real model: idle arms hang down; the face parts register and drive their morphs; the face reacts and its surprise resolves',{skip},async()=>{
 const s=await stage();s.run(1);
 for(const side of ['left','right']){const d=s.world(side+'LowerArm').sub(s.world(side+'UpperArm')).normalize();say(`idle ${side} upper arm direction y ${d.y.toFixed(3)}`);assert.ok(d.y<-.8,`${side} arm raised (${d.y.toFixed(2)})`);}
 const manager=s.vrm.expressionManager,names=Object.keys(manager.expressionMap);
 say(`expressions ${names.length} (face parts added ${s.added.length}), surprised -> ${resolveExpression(manager,'surprised')}`);
 assert.ok(resolveExpression(manager,'surprised'));assert.ok(s.added.length>=10);assert.ok(names.length<=MAX_EXPRESSIONS);
 const meshes=[];s.vrm.scene.traverse(o=>{if(o.isMesh&&o.morphTargetDictionary?.Fcl_BRW_Surprised!==undefined)meshes.push(o);});
 for(const name of names)manager.setValue(name,0);manager.setValue('browUp',.7);manager.update();
 const weights=meshes.map(m=>m.morphTargetInfluences[m.morphTargetDictionary.Fcl_BRW_Surprised]);say(`browUp 0.7 -> Fcl_BRW_Surprised on ${meshes.length} primitives: ${[...new Set(weights.map(w=>w.toFixed(3)))].join(', ')}`);
 assert.ok(meshes.length>0&&weights.every(w=>Math.abs(w-.7)<1e-6));
 const p=s.press('upperChest');s.run(.05);s.grab();s.move(p.x+8,p.y);const at=[];s.run(2.2,i=>{if(i===30||i===130)at.push(new Map(s.expressions));});
 const browUp=resolveExpression(manager,'browUp'),browWorry=resolveExpression(manager,'browWorry');
 say(`grab face: at 0.5 s browUp ${(at[0].get(browUp)||0).toFixed(2)}; at 2.2 s browUp ${(at[1].get(browUp)||0).toFixed(2)}, browWorry ${(at[1].get(browWorry)||0).toFixed(2)}`);
 assert.ok(at[0].get(browUp)>.85,'surprised');assert.ok(!(at[1].get(browUp)>.1)&&at[1].get(browWorry)>.4,'the surprise resolved into worry');
 s.audit('idle and grab');
});

test('real model: the dangling silhouette, and a carry that swings her as one piece with the hair, then decays',{skip},async()=>{
 const s=await stage();s.run(1);const p=s.press('upperChest');s.run(.05);s.grab();s.move(p.x+8,p.y);s.run(2);
 const spine=angleBetween(s.world('upperChest').sub(s.world('hips')),new THREE.Vector3(0,1,0)),thighs=['left','right'].map(x=>downAngle(s.world(x+'LowerLeg').sub(s.world(x+'UpperLeg')))),toes=['left','right'].map(x=>s.world(x+'Toes').sub(s.world(x+'Foot')).normalize().y);
 const hands=['left','right'].map(x=>s.world(x+'Hand').z-s.world('upperChest').z),gap=s.world('leftFoot').distanceTo(s.world('rightFoot'));
 say(`dangling (${s.hold.variant}, scruff ${s.profile.pickup.scruffPitch}): spine tilt ${spine.toFixed(1)}, thighs ${thighs.map(v=>v.toFixed(1))} from down, toe direction y ${toes.map(v=>v.toFixed(2))}, hands ${hands.map(v=>(v*100).toFixed(0))} cm in front of the chest, feet ${(gap*100).toFixed(0)} cm apart`);
 assert.ok(spine>=25&&spine<=35);assert.ok(thighs.every(v=>v<10));assert.ok(toes.every(v=>v<-.6));assert.ok(hands.every(v=>v>=.1));assert.ok(gap<.25);
 // the design's carry: accelerate at 6 m/s^2 for 0.25 s, cruise 0.5 s, stop at -15 m/s^2 over 0.1 s, as cursor motion in px
 const x0=s.drag.point.x,y0=s.drag.point.y,px=m=>m/s.wpp,xs=t=>t<.25?3*t*t:t<.75?.1875+1.5*(t-.25):t<.85?.9375+1.5*(t-.75)-7.5*(t-.75)**2:1.0125;
 const anchor=s.pickup.anchorPoint.clone(),S=s.vrm.humanoid.getRawBoneNode('upperChest');let worstAnchor=0,worstSwing=0,untouched=true;const after=[],hair=[];
 for(let i=0;i<Math.round(4/DT);i++){const t=i*DT;s.move(x0+px(xs(t)),y0);s.frame();worstAnchor=Math.max(worstAnchor,S.getWorldPosition(new THREE.Vector3()).distanceTo(anchor));worstSwing=Math.max(worstSwing,Math.abs(s.pickup.angle));untouched&&=s.untouched;
  if(t>=.85)after.push(s.pickup.angle/D);hair.push(planeAngle(s.hair[0].tip.getWorldPosition(new THREE.Vector3()).sub(s.hair[0].root.getWorldPosition(new THREE.Vector3())))-planeAngle(s.world('hips').sub(s.world('head'))));}
 const peaks=after.filter((v,i)=>i>0&&i<after.length-1&&Math.abs(v)>=Math.abs(after[i-1])&&Math.abs(v)>Math.abs(after[i+1])&&Math.abs(v)>3),hairRange=Math.max(...hair)-Math.min(...hair);
 say(`carry: anchor error ${(worstAnchor*1000).toFixed(3)} mm, swing max ${(worstSwing/D).toFixed(1)} (limit ${s.profile.pickup.swingLimit}), peaks after the stop ${peaks.map(v=>v.toFixed(1)).join(' ')}, last 0.5 s under ${Math.max(...after.slice(-30).map(Math.abs)).toFixed(2)} deg; hair tip (${s.hair[0].tip.name}) swings ${hairRange.toFixed(1)} deg against the head-hips axis; non-humanoid bones untouched by the rig: ${untouched}`);
 assert.ok(worstAnchor<.001);assert.ok(worstSwing<=s.profile.pickup.swingLimit*D+1e-9);assert.ok(peaks.length>=2&&peaks.length<=4);assert.ok(Math.max(...after.slice(-30).map(Math.abs))<2,'it decays');
 assert.ok(hairRange>=20);assert.ok(untouched);
 s.audit('dangle and carry');
});

test('real model: a throw flies, lands on the given floor with a squash that drops the hips with the toes planted, and recovers; a gentle foot-grab release ends on the floor',{skip},async()=>{
 const s=await stage({geometry:{x:700,y:-1100}});s.run(1);
 const tip=()=>Math.min(...shoeTips(s.vrm.humanoid).map(t=>t.y)),standTip=tip(),standHips=s.world('hips').y,standSoles=s.soles()-s.geometry.y;  // the shoe tips stand about 8 px above the soles' line
 const p=s.press('upperChest');s.run(.05);s.grab();s.move(p.x+8,p.y);s.run(1);
 let x=s.drag.point.x;for(let i=0;i<8;i++){x+=1600/60*Math.min(1,(i+1)/3);s.move(x,p.y);s.frame();}
 const decision=s.release(),t0=s.time;let landed=null,peak={squash:0},hipsAtPeak=0,tipAtPeak=0,solesAtPeak=0,end=null;
 s.run(5,()=>{if(s.landed)landed={...s.landed,t:s.time-t0,y:s.geometry.y,floor:s.floor()};
  if(s.out?.pose?.squash>peak.squash){peak={squash:s.out.pose.squash,t:s.time-t0};hipsAtPeak=s.world('hips').y;tipAtPeak=tip();solesAtPeak=s.soles()-s.geometry.y;}if(landed&&!end&&!s.out)end=s.time-t0;});
 const goal=Math.min(1.2,Math.max(.5,landed.impact/2500));
 say(`throw: ${decision}; landed after ${landed.t.toFixed(2)} s at ${landed.impact.toFixed(0)} px/s, canvas y ${landed.y.toFixed(1)} against the floor top ${landed.floor.toFixed(1)}; squash peak ${peak.squash.toFixed(3)} (${goal.toFixed(3)} expected) ${(peak.t-landed.t).toFixed(2)} s after contact; hips dropped ${((standHips-hipsAtPeak)*100).toFixed(1)} cm, shoe tips ${((tipAtPeak-standTip)*1000).toFixed(1)} mm off the line they stood on (on screen ${(solesAtPeak-standSoles).toFixed(2)} px from where they stood); reaction over ${(end-landed.t).toFixed(2)} s after contact; rig active ${s.pickup.active}; PATCH y ${s.patches.at(-1)?.y.toFixed(1)}`);
 assert.equal(decision,'fall');assert.ok(Math.abs(landed.y-landed.floor)<1e-6);assert.ok(Math.abs(peak.squash-goal)<.02);assert.ok(goal>.95,'a landing at s = 1');
 assert.ok(standHips-hipsAtPeak>=.04&&standHips-hipsAtPeak<=.09,'hips drop 4-9 cm');assert.ok(Math.abs(solesAtPeak-standSoles)<1,'on screen the lowest shoe tip stays where it stood on the floor');assert.ok(Math.abs(tipAtPeak-standTip)<.01,'toes within 1 cm of the line they stood on');
 assert.ok(Math.abs(end-landed.t-s.profile.pickup.recoverySeconds)<.05);assert.equal(s.pickup.active,false);assert.ok(Math.abs(s.patches.at(-1).y-landed.floor)<1e-6);
 s.audit('throw and landing');
 const f=await stage({geometry:{x:700,y:0}});f.geometry.y=f.floor();f.run(1);const q=f.press('leftFoot');f.run(.05);f.grab();for(let i=1;i<=10;i++){f.move(q.x+i,q.y);f.frame();}f.run(1.5);
 const slid=f.drag.slid.y,held=f.geometry.y,choice=f.release();let early=0;f.run(3,()=>{if(f.out?.pose?.squash>0&&f.geometry.y>f.floor()+.5)early++;});
 say(`foot grab on the floor: slid ${slid} px down while held (canvas ${held.toFixed(0)}), released gently (${choice}) -> canvas y ${f.geometry.y.toFixed(1)}, floor top ${f.floor().toFixed(1)}, PATCH y ${f.patches.at(-1)?.y.toFixed(1)}`);
 assert.ok(slid>200);assert.equal(choice,'place');assert.equal(early,0,'no squash before her feet reach the floor');assert.ok(Math.abs(f.geometry.y-f.floor())<1e-6&&Math.abs(f.patches.at(-1).y-f.floor())<1e-6,'her feet end on the floor, not below it');
 f.audit('foot grab set-down');
});

test('real model: off the taskbar a head grab let go lands back on it, a sideways flick by the hips hops her off it, the walls follow her flung arms and her feet stay put after the landing',{skip},async()=>{
 const h=await stage({geometry:{x:700,y:0}});h.geometry.y=h.floor();h.run(1);const stood=h.soles(),hp=h.press('head');h.run(.05);h.grab();h.move(hp.x+3,hp.y);h.run(1.2);
 const lifted=h.drag.slid.y,choice=h.release();h.run(2.5);
 say(`head grab on the taskbar: the slide lifted the canvas ${-lifted} px; let go (${choice}) -> canvas y ${h.geometry.y.toFixed(1)}, floor top ${h.floor().toFixed(1)}, shoe tips at ${h.soles().toFixed(1)} px (${stood.toFixed(1)} before the grab)`);
 assert.ok(lifted<-30);assert.equal(choice,'place');assert.ok(Math.abs(h.patches.at(-1).y-h.floor())<1e-6,'back on the taskbar, not in mid-air');assert.ok(Math.abs(h.soles()-stood)<1.5);
 h.audit('head grab set-down');
 const s=await stage({geometry:{x:300,y:0}});s.geometry.y=s.floor();s.run(1);const p=s.press('hips');s.run(.05);s.grab();s.move(p.x+3,p.y-6);s.run(.8);
 let x=s.drag.point.x;const start=s.geometry.y;for(let i=0;i<7;i++){x+=2500/60*Math.min(1,(i+1)/3);s.move(x,p.y-6);s.frame();}
 const from=s.geometry.x,decision=s.release(),t0=s.time;let top=Infinity,out=0,landed=null,toe0=null,slide=0,blinks=0;
 s.run(3,()=>{if(s.out?.state==='falling'){const b=s.box();out=Math.max(out,WORK.left-(s.geometry.x+b.left),s.geometry.x+b.right-WORK.right);top=Math.min(top,s.geometry.y);}
  if(s.landed)landed={t:s.time-t0,x:s.geometry.x};const toes=shoeTips(s.vrm.humanoid).map(t=>s.screen(t).x),mid=(toes[0]+toes[1])/2;if(landed){toe0??=mid;slide=Math.max(slide,Math.abs(mid-toe0));}});
 blinks=s.blinks.filter(b=>b.t>=t0&&b.kind==='gaze'&&b.state==='falling').length;
 say(`hips grab on the taskbar (canvas ${(start-s.floor()).toFixed(0)} px below its floor top while held), flick right: ${decision}, rose to ${(s.floor()-top).toFixed(0)} px above the floor, landed after ${landed?.t.toFixed(2)} s ${(landed?.x-from).toFixed(0)} px to the right; her bone box at most ${out.toFixed(1)} px past the screen sides in the air; shoe tips slid ${slide.toFixed(1)} px after the contact; gaze blinks while falling ${blinks}; PATCH y ${s.patches.at(-1)?.y.toFixed(1)}`);
 assert.equal(decision,'fall');assert.ok(s.floor()-top>30,'a hop off the taskbar');assert.ok(landed&&landed.x-from>300);assert.ok(out<=1,'the walls follow her body');
 assert.ok(slide<8,'her feet stay where they touched down');assert.equal(blinks,0,'her eyes stay wide on the way down');assert.ok(Math.abs(s.patches.at(-1).y-s.floor())<1e-6);
 s.audit('flick off the taskbar');
});

test('real model: standing still at the adaptive 30 fps her hair settles where it does at 60 fps and her face stays forward',{skip},async()=>{
 const tips=s=>s.hair.map(h=>h.tip.getWorldPosition(new THREE.Vector3()));
 const fast=await stage(),slow=await stage({dt:1/30});fast.run(3);slow.run(3);
 const gap=Math.max(...tips(fast).map((p,i)=>p.distanceTo(tips(slow)[i])));
 const wander=[];slow.run(1,()=>wander.push(tips(slow)[0].clone()));const drift=Math.max(...wander.map(p=>p.distanceTo(wander[0])));
 say(`idle 30 fps against 60 fps: hair tips ${(gap*1000).toFixed(1)} mm apart after 3 s; the 30 fps tip moves ${(drift*1000).toFixed(1)} mm over the next second`);
 assert.ok(gap<.02,'the hair rests in the same place');assert.ok(drift<.02,'and stays settled');
 slow.audit('idle at 30 fps');
});
