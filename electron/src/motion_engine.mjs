import * as THREE from 'three';
import {heldPose,settlePose} from './avatar_held_pose.mjs';

export const BODY_BONES=['hips','spine','chest','upperChest','neck','head','leftShoulder','rightShoulder','leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','leftHand','rightHand','leftUpperLeg','rightUpperLeg','leftLowerLeg','rightLowerLeg','leftFoot','rightFoot','leftToes','rightToes'];
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const quaternion=rotation=>new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation,'XYZ'));
/** A VRM 0.x normalized rig negates x and z against VRM 1.0 (three-vrm-animation's createVRMAnimationClip flips .vrma the same way).
 * Its own inverse: canonical (VRM 1.0) to native and back. */
export const flipVRM0=q=>new THREE.Quaternion(-q.x,q.y,-q.z,q.w);

export function proceduralPose(mode,time,strength=.65,walkSpeed=1) {
  const pose=Object.fromEntries(BODY_BONES.map(b=>[b,[0,0,0]]));
  pose.leftUpperArm=[0,0,-1.1];pose.rightUpperArm=[0,0,1.1];
  pose.head=[Math.sin(time*.8)*.025,0,Math.sin(time*.55)*.018];
  pose.chest=[Math.sin(time*1.7)*.012,0,0];
  if(mode==='listening'){pose.head=[-.08,0,.08];pose.chest=[-.04,0,0];}
  if(mode==='thinking'||mode==='tool'){pose.head=[.12,Math.sin(time*.6)*.1,-.12];pose.rightUpperArm=[-.5,0,.45];pose.rightLowerArm=[-1.25,0,0];}
  if(mode==='sleeping'){pose.head=[.3,0,.2];pose.chest=[.15,0,0];}
  if(mode==='speaking'||mode==='playful'){pose.head=[Math.sin(time*3)*.07*strength,0,Math.sin(time*1.7)*.08*strength];pose.rightUpperArm=[-.12,0,.85+Math.sin(time*2)*.15*strength];pose.rightLowerArm=[-.2,0,0];}
  if(mode==='clicked'){pose.head=[-.18,0,.1];pose.leftUpperArm=[-.2,0,-.8];pose.rightUpperArm=[-.2,0,.8];}
  if(mode==='walking'){
    const swing=Math.sin(time*7*clamp(walkSpeed,.25,2))*.32;
    pose.leftUpperLeg=[swing,0,0];pose.rightUpperLeg=[-swing,0,0];
    pose.leftLowerLeg=[Math.max(0,-swing)*.8,0,0];pose.rightLowerLeg=[Math.max(0,swing)*.8,0,0];
    pose.leftUpperArm=[-swing*.7,0,-1.05];pose.rightUpperArm=[swing*.7,0,1.05];
    pose.head=[Math.abs(swing)*.06,0,0];
  }
  return new Map(Object.entries(pose).map(([name,value])=>[name,quaternion(value)]));
}

export function gesturePose(name,time,base,amount) {
  const result=new Map([...base].map(([key,value])=>[key,value.clone()]));
  const swing=Math.sin(time*9)*amount;
  if(name==='nod')result.set('head',quaternion([swing*.2,0,0]));
  if(name==='shake')result.set('head',quaternion([0,swing*.3,0]));
  if(name==='wave'){result.set('rightUpperArm',quaternion([0,0,-amount*.5]));result.set('rightLowerArm',quaternion([0,0,-amount*1.2+swing*.3]));}
  return result;
}

/** Blends from the displayed pose even when a transition is interrupted. */
export class PoseInterpolator {
  constructor(initial=new Map()) {this.displayed=new Map([...initial].map(([k,v])=>[k,v.clone()]));this.from=new Map();this.key=null;this.age=0;this.duration=.3;}
  /** Mid-crossfade (the avatar's frame pacing draws every display frame meanwhile). */
  get blending() {return this.age<Math.max(.05,this.duration);}
  step(target,key,dt,duration=.3) {
    this.duration=duration;
    if(key!==this.key){this.key=key;this.from=new Map([...this.displayed].map(([k,v])=>[k,v.clone()]));this.age=0;}
    this.age+=Math.max(0,Math.min(.1,dt));
    const t=clamp(this.age/Math.max(.05,duration),0,1),weight=t*t*(3-2*t);
    for(const [bone,value]of target){const from=this.from.get(bone)||this.displayed.get(bone)||value;this.displayed.set(bone,from.clone().slerp(value,weight).normalize());}
    return this.displayed;
  }
}

/** Sole writer of normalized skeletal motion; the carry rig (AvatarPickup.pose) and secondary VRM updates run afterward.
 * Procedural, gesture, held (avatar_held_pose.mjs), .pose.json and gaze targets, the rest pose and the blend are all in canonical
 * VRM 1.0 axes; a VRM 0.x model is flipped once on write, and samples read back from its nodes (.vrma, sample.native) are flipped to
 * canonical before they merge. While she is held, interaction.hold (HoldBehaviour's step) picks the held pose and its crossfade time,
 * and the pointer head-gaze stops: the carry rig owns the head then. 'held', 'settling' and 'recovering' modes from a hit rule or the
 * backend map to the plain dangle and an in-place settle. Falling is the held pose with fall 1 (avatar_held_pose.mjs fallPose). Asset expression targets are exposed as expressionTargets for the face
 * layer, which owns every expression write. */
export class MotionEngine {
  constructor(vrm,sampler,report=()=>{}) {
    this.vrm=vrm;this.sampler=sampler;this.report=report;this.time=0;this.ages=new Map();this.vrm0=vrm.meta?.metaVersion==='0';
    const initial=new Map(BODY_BONES.map(name=>[name,vrm.humanoid.getNormalizedBoneNode(name)?.quaternion]).filter(([,value])=>value).map(([name,value])=>[name,this.canonical(value)]));
    this.blender=new PoseInterpolator(initial);this.gaze=new THREE.Quaternion();
    this.rest=new Map(initial);this.expressionTargets=new Map();this.mode='idle';
  }
  /** Crossfading between poses (avatar_frame_rate.mjs motionBusy). */
  get blending(){return this.blender.blending;}
  canonical(q){return this.vrm0?flipVRM0(q):q.clone();}
  age(action,dt){if(!action)return 0;const next=(this.ages.get(action.id)||0)+dt;this.ages.set(action.id,next);return next;}
  /** held: {pitch, kickAmplitude} for the held pose (the studio's scruff tilt and kick size). */
  update({actions=[],emotion={},interaction={},walking=false,walkSpeed=1,settings={},held={}},dt) {
    this.time+=dt;
    const base=actions.find(a=>a.kind==='motion.base'&&a.status==='running');
    const preview=actions.find(a=>a.kind==='motion.preview'&&a.status==='running');
    const wake=actions.find(a=>a.kind==='wake_animation'&&a.status==='running');
    const gesture=actions.find(a=>a.kind==='gesture'&&a.status==='running');
    const active=actions.filter(a=>a.status==='running');
    const ids=new Set(active.map(a=>a.id));for(const id of this.ages.keys())if(!ids.has(id))this.ages.delete(id);
    const enabled=settings.enabled!==false,hold=enabled&&interaction.hold?.pose?interaction.hold:null,holding=!!(hold||interaction.held);
    const custom=(holding||interaction.reaction)&&interaction.rule?.animation!=='default'?interaction.rule?.animation:null;
    const mode=this.mode=enabled?(custom|| (hold?'hold':interaction.held?'held':interaction.reaction|| (walking?'walking':base?.payload.procedural||'idle'))):'idle';
    let target,modeKey=mode;
    // Landing and recovering are one settle (idle with the squash): one key, so the crossfade into it never restarts between them.
    if(mode==='hold'){const settle=hold.state==='landing'||hold.state==='recovering';target=settle?settlePose(proceduralPose('idle',this.time),hold.pose.squash):heldPose(hold,this.time,held);modeKey=settle?'hold:settle':'hold:'+hold.state+':'+hold.pose.variant;}
    else if(mode==='held'){target=heldPose(hold,this.time,held);modeKey='held:'+(hold?hold.state+':'+hold.pose.variant:'');}
    else if(mode==='settling'||mode==='recovering')target=settlePose(proceduralPose('idle',this.time),.3);
    else target=proceduralPose(mode,this.time,emotion.intensity??.65,walkSpeed);
    const connected=interaction.rule?.assetId&&preview?.payload.asset?.id===interaction.rule.assetId?preview:null;
    const overlays=holding||interaction.reaction?[connected]:[base?.payload.procedural===mode?base:null,preview,wake];
    const assetKeys=[];
    const expressionTargets=new Map();
    for(const action of overlays.filter(Boolean)){
      const age=this.age(action,dt);
      const asset=action.payload.asset||(action.kind==='wake_animation'?{id:action.id,kind:'vrma',path:action.payload.path,mask:[],loop:false,speed:1}:null);
      if(!asset)continue;
      const sample=this.sampler.sample(asset,age,action.id,(status,error)=>this.report(action.id,status,error));
      if(sample){for(const [bone,rotation]of sample.bones)target.set(bone,sample.native?this.canonical(rotation):rotation);for(const [name,value]of sample.expressions||[])expressionTargets.set(name,value);assetKeys.push(asset.id);}
    }
    if(gesture&&!holding){
      const age=this.age(gesture,dt),envelope=Math.max(0,Math.min(1,age*5,(gesture.duration-age)*5));
      target=gesturePose(gesture.payload.name,age,target,envelope*(gesture.payload.intensity??.65));
    }
    if(enabled&&!holding&&settings.mouse_tracking!==false&&interaction.pointer?.near){
      const pointer=interaction.pointer;
      this.gaze.slerp(quaternion([clamp(pointer.y,-1,1)*.16,clamp(pointer.x,-1,1)*.35,0]),1-Math.exp(-dt*8));
    }else this.gaze.slerp(new THREE.Quaternion(),1-Math.exp(-dt*8));
    if(target.has('head'))target.get('head').multiply(this.gaze);
    // While held only the hold (and a hit rule's connected clip) decides the key, so a backend base change never restarts the crossfade.
    const key=(holding?[connected?.id,modeKey]:[base?.id,preview?.id,wake?.id,gesture?.id,modeKey]).concat(assetKeys).join(':');
    for(const bone of this.blender.displayed.keys())if(!target.has(bone))target.set(bone,this.rest.get(bone)||new THREE.Quaternion());
    const pose=this.blender.step(target,key,dt,mode==='hold'?hold.transition:preview?.payload.transition_seconds||base?.payload.transition_seconds||settings.transition_seconds||.3);
    for(const [bone,rotation]of pose){const node=this.vrm.humanoid.getNormalizedBoneNode(bone);if(node)node.quaternion.copy(this.vrm0?flipVRM0(rotation):rotation);}
    this.expressionTargets=expressionTargets;
    this.sampler.retain(new Set(overlays.filter(Boolean).map(a=>a.payload.asset?.id||a.id)));
  }
  dispose(){this.sampler.dispose();}
}
