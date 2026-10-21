import {Vector3,Quaternion,Raycaster,Vector2,Plane} from 'three';
const clamp=(value,limit)=>Math.max(-limit,Math.min(limit,value));
const D=Math.PI/180;
/** Saccade timing: the main sequence ((21 + 2.2 A) ms for an A-degree jump), the smallest error that starts one, the shortest fixation
 * between jumps, how far a target may move mid-jump before the jump is re-planned, a big shift (it comes with a blink: gaze-evoked
 * blinks go with almost every shift over 33 degrees; AvatarFace.gazeShift keeps them 2 s apart), and micro-saccades every 0.6-1.5 s of
 * 0.15-0.3 degrees, never past 0.5. */
export const SACCADE={base:.021,slope:.0022,threshold:1.5,fixation:.12,replan:5,big:25,microEvery:[.6,1.5],microSize:[.15,.3],microMax:.5};
/** Whether the eyes follow the cursor this frame: a reaction's look decides ('cursor' tracks it even when it is not near; any other
 * look has a point of its own or rests ahead), and with no look (null) she tracks it when it is near her, as without a reaction. */
export const gazeActive=(look,near)=>look?look==='cursor':!!near;
const smooth=u=>{u=Math.max(0,Math.min(1,u));return u*u*(3-2*u);};

/** Saccadic gaze on world points (Lee and Badler, 'Eyes Alive'). The eyes hold a fixation point and jump to a new one with a smoothstep
 * profile in (21 + 2.2 A) ms when the target is more than 1.5 degrees away and the fixation has lasted 0.12 s; a target that moves more
 * than 5 degrees from a jump's destination re-plans it. The fixation is a point in the world, so a turning head keeps the eyes on it
 * (the vestibulo-ocular reflex) instead of dragging them along. Sizes are the angle between points seen from the eyes, or, given a
 * measure (point -> {yaw, pitch} in degrees, as far as the eyes can turn), the eyes' own turn: a cursor sweeping 90 degrees past her
 * face moves eyes clamped at 45 only that far. micro is the current micro-saccade offset {yaw, pitch} in degrees. bigShift is true for
 * the step in which a jump over 25 degrees starts from a fixation (a re-plan mid-jump is no new shift). Pure (three.js vectors only). */
export class SaccadeGaze{
 constructor({random=Math.random}={}){this.random=random;this.current=null;this.from=new Vector3();this.to=new Vector3();this.saccading=false;this.age=0;this.duration=0;this.fixation=0;this.amplitude=0;this.bigShift=false;
  this.micro={yaw:0,pitch:0};this.microFrom={yaw:0,pitch:0};this.microTo={yaw:0,pitch:0};this.microAge=Infinity;this.nextMicro=this.microInterval();this.a=new Vector3();this.b=new Vector3();this.q=new Quaternion();this.q0=new Quaternion();}
 microInterval(){const [a,b]=SACCADE.microEvery;return a+(b-a)*this.random();}
 /** The angle (degrees) between two world points as seen from origin, or the eyes' turn between them under this step's measure. */
 angle(p,q,origin){
  if(this.measure){const a=this.measure(p),b=this.measure(q);return Math.hypot(a.yaw-b.yaw,a.pitch-b.pitch);}
  const a=this.a.copy(p).sub(origin),b=this.b.copy(q).sub(origin);const l=a.length()*b.length();return l>0?Math.acos(Math.max(-1,Math.min(1,a.dot(b)/l)))/D:0;
 }
 start(target,origin){
  const fixed=!this.saccading;this.from.copy(this.current);this.to.copy(target);this.amplitude=this.angle(this.from,this.to,origin);
  this.duration=SACCADE.base+SACCADE.slope*this.amplitude;this.age=0;this.saccading=true;this.fixation=0;
  if(fixed&&this.amplitude>SACCADE.big)this.bigShift=true;
  this.micro={yaw:0,pitch:0};this.microAge=Infinity;this.nextMicro=this.microInterval();  // micro-saccades ride on a fixation
 }
 /** desired: the target world point; origin: the eyes' world position; dt seconds; measure (optional, see the class). Returns the point
  * the eyes look at (reused). */
 step(desired,origin,dt,measure=null){
  const h=Math.max(0,Math.min(.1,dt||0));this.bigShift=false;this.measure=measure;
  if(!this.current){this.current=desired.clone();return this.current;}
  if(this.saccading&&this.angle(desired,this.to,origin)>SACCADE.replan)this.start(desired,origin);
  else if(!this.saccading){
   this.fixation+=h;
   if(this.fixation>=SACCADE.fixation-1e-9&&this.angle(desired,this.current,origin)>SACCADE.threshold)this.start(desired,origin);  // it moves in this step too
   else{
    this.nextMicro-=h;
    if(this.nextMicro<=0){const [a,b]=SACCADE.microSize,size=a+(b-a)*this.random(),turn=2*Math.PI*this.random();this.microFrom={...this.micro};this.microTo={yaw:size*Math.cos(turn),pitch:size*Math.sin(turn)};this.microAge=0;this.nextMicro=this.microInterval();}
    if(this.microAge<Infinity){this.microAge+=h;const length=SACCADE.base+SACCADE.slope*Math.hypot(this.microTo.yaw-this.microFrom.yaw,this.microTo.pitch-this.microFrom.pitch),u=smooth(this.microAge/length);
     for(const k of ['yaw','pitch'])this.micro[k]=clamp(this.microFrom[k]+(this.microTo[k]-this.microFrom[k])*u,SACCADE.microMax);if(u>=1)this.microAge=Infinity;}
   }
  }
  if(this.saccading){
   this.age+=h;
   if(this.age>=this.duration){this.saccading=false;this.fixation=this.age-this.duration;this.current.copy(this.to);}
   else{  // along the great circle seen from the eyes, with the distance in between
    const a=this.a.copy(this.from).sub(origin),b=this.b.copy(this.to).sub(origin),la=a.length(),lb=b.length(),u=smooth(this.age/this.duration);
    if(la>0&&lb>0){this.q.setFromUnitVectors(a.divideScalar(la),b.divideScalar(lb));this.q0.identity().slerp(this.q,u);this.current.copy(a.applyQuaternion(this.q0).multiplyScalar(la+(lb-la)*u)).add(origin);}
    else this.current.lerpVectors(this.from,this.to,u);
   }
  }
  return this.current;
 }
}

/** Binocular convergence uses rotations only: anatomical eye positions never change. With settings.saccades (the default) the eyes jump
 * between fixations (SaccadeGaze); without it they glide exponentially at settings.smoothing. update() takes an optional world point
 * (override) that replaces the cursor target, even when the pointer is not near: the held reactions' viewer, holder and floor. */
export class AvatarEyeGaze{
 constructor(vrm,{random=Math.random}={}){this.vrm=vrm;this.lookAt=vrm.lookAt;this.originalAuto=this.lookAt?.autoUpdate;this.controlled=false;this.saved=[];this.angles={left:{yaw:0,pitch:0},right:{yaw:0,pitch:0}};this.ray=new Raycaster();this.plane=new Plane();this.target=new Vector3();this.tracking=false;this.bigShift=false;this.saccade=new SaccadeGaze({random});this.ahead=new Vector3();}
 /** How far (degrees) the eye bones turn at most (their lookAt range maps); the head joins past it. Miku: 12 sideways, 10 up and down. */
 get reach(){const a=this.lookAt?.applier;return a?.constructor?.type==='bone'?{yaw:Math.max(a.rangeMapHorizontalOuter?.outputScale||0,a.rangeMapHorizontalInner?.outputScale||0)||12,pitch:Math.max(a.rangeMapVerticalUp?.outputScale||0,a.rangeMapVerticalDown?.outputScale||0)||10}:{yaw:12,pitch:10};}
 restore(){for(const [node,q] of this.saved)node.quaternion.copy(q);this.saved=[];}
 update(settings,dt,point,rect,camera,sphere,active=true,override=null){
  const look=this.lookAt;this.bigShift=false;if(!look)return;
  if(!settings.enabled){if(this.controlled){look.autoUpdate=this.originalAuto;look.reset();this.controlled=false;}this.tracking=false;look.update(dt);return;}
  this.controlled=true;look.autoUpdate=false;
  const head=look.getLookAtWorldPosition(new Vector3());let tracking=false;
  const forward=new Vector3(0,0,1).applyQuaternion(look.getFaceFrontQuaternion(new Quaternion())).applyQuaternion(look.getLookAtWorldQuaternion(new Quaternion()));
  if(override){this.target.copy(override);tracking=true;}
  else if(active&&point&&rect.width&&rect.height){
   if(sphere?.active&&sphere.initialized)this.target.copy(sphere.position);
   else {this.ray.setFromCamera(new Vector2((point.x-rect.left)/rect.width*2-1,1-(point.y-rect.top)/rect.height*2),camera);this.plane.set(new Vector3(0,0,1),-(head.z+settings.minDistance));if(!this.ray.ray.intersectPlane(this.plane,this.target)){this.tracking=false;return;}}
   tracking=true;
  }
  if(tracking){  // Keep the synthetic target in front of the face, including during pickup.
   const depth=this.target.clone().sub(head).dot(forward);if(depth<settings.minDistance)this.target.addScaledVector(forward,settings.minDistance-depth);
  }
  this.tracking=tracking;
  if(settings.saccades===false)this.glide(settings,dt,head,tracking);
  else{
   // No target: a point a metre ahead of the face, so the eyes come back to centre with a jump too.
   const measure=point=>{look.lookAt(point);return {yaw:clamp(look.yaw,settings.maxYaw),pitch:clamp(look.pitch,settings.maxPitch)};};
   const fixation=this.saccade.step(tracking?this.target:this.ahead.copy(head).addScaledVector(forward,1),head,dt,measure),micro=this.saccade.micro;this.bigShift=this.saccade.bigShift;
   look.lookAt(fixation);const base={yaw:clamp(look.yaw,settings.maxYaw),pitch:clamp(look.pitch,settings.maxPitch)},center={yaw:clamp(base.yaw+micro.yaw,settings.maxYaw),pitch:clamp(base.pitch+micro.pitch,settings.maxPitch)};
   for(const side of ['left','right'])this.angles[side]=this.eye(settings,side,fixation,head,center,base);
  }
  const left=this.vrm.humanoid.getRawBoneNode('leftEye'),right=this.vrm.humanoid.getRawBoneNode('rightEye');
  const boneMode=look.applier?.constructor?.type==='bone'&&left&&right;
  if(boneMode){
   for(const name of ['leftEye','rightEye'])for(const node of [this.vrm.humanoid.getRawBoneNode(name),this.vrm.humanoid.getNormalizedBoneNode(name)])if(node&&!this.saved.some(([n])=>n===node))this.saved.push([node,node.quaternion.clone()]);
   look.yaw=(this.angles.left.yaw+this.angles.right.yaw)/2;look.pitch=(this.angles.left.pitch+this.angles.right.pitch)/2;look.update(dt);
   look.applier.applyYawPitch(this.angles.left.yaw,this.angles.left.pitch);
   const leftPose=left.quaternion.clone(),normalized=this.vrm.humanoid.getNormalizedBoneNode('leftEye'),normalizedPose=normalized?.quaternion.clone();
   look.applier.applyYawPitch(this.angles.right.yaw,this.angles.right.pitch);left.quaternion.copy(leftPose);if(normalizedPose)normalized.quaternion.copy(normalizedPose);
  }else{ // Expression-only rigs support shared gaze, but not independent pupil spacing.
   look.yaw=(this.angles.left.yaw+this.angles.right.yaw)/2;look.pitch=(this.angles.left.pitch+this.angles.right.pitch)/2;look.update(dt);
  }
 }
 /** One eye's yaw and pitch toward point: the shared centre plus its own convergence (the point seen from that eye, measured against
  * base, the centre without a micro-saccade, which every eye shares). */
 eye(settings,side,point,head,center,base=center){
  const eye=this.vrm.humanoid.getRawBoneNode(side+'Eye'),look=this.lookAt;
  if(!settings.convergence||!eye)return {...center};
  look.lookAt(point.clone().add(head).sub(eye.getWorldPosition(new Vector3())));
  return {yaw:clamp(center.yaw+clamp(look.yaw-base.yaw,settings.maxConvergence)*settings.strength,settings.maxYaw),pitch:clamp(center.pitch+look.pitch-base.pitch,settings.maxPitch)};
 }
 /** The exponential glide (settings.saccades false). */
 glide(settings,dt,head,tracking){
  const look=this.lookAt,desired={left:{yaw:0,pitch:0},right:{yaw:0,pitch:0}};
  if(tracking){look.lookAt(this.target);const center={yaw:clamp(look.yaw,settings.maxYaw),pitch:clamp(look.pitch,settings.maxPitch)};for(const side of ['left','right'])desired[side]=this.eye(settings,side,this.target,head,center);}
  const weight=1-Math.exp(-settings.smoothing*Math.max(0,Math.min(.1,dt)));
  for(const side of ['left','right'])for(const key of ['yaw','pitch'])this.angles[side][key]+=(desired[side][key]-this.angles[side][key])*weight;
 }
 dispose(){this.restore();if(this.lookAt){this.lookAt.autoUpdate=this.originalAuto;this.lookAt.reset();}}
}
