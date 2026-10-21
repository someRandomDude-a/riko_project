import {Vector3,Quaternion,Box3,Matrix4,Euler} from 'three';
import {flipVRM0} from './motion_engine.mjs';
const D=Math.PI/180,G=9.8,TAU=2*Math.PI,SUBSTEPS=4,HIT_STOP=.08,LETGO=.35,CONTACT=.08,SETTLE=1.678/.1,SET_DOWN=1.678/.06,SET_RAMP=.1,VIEW=new Vector3(0,0,1);  // s, s, s; critically damped rates for 0.1 s and 0.06 s half-lives; s
const clamp=(v,a,b)=>Math.max(a,Math.min(b,Number.isFinite(v)?v:0));
const smooth=u=>{u=clamp(u,0,1);return u*u*(3-2*u);};
const smoother=u=>{u=clamp(u,0,1);return u*u*u*(u*(6*u-15)+10);};  // zero speed and acceleration at both ends
const soft=(x,limit)=>{const knee=.6*limit,a=Math.abs(x);return a<=knee?x:Math.sign(x)*(knee+(limit-knee)*Math.tanh((a-knee)/(limit-knee)));};  // linear, then easing into the limit
/** Soft limits on the lag rolls relative to the parent (radians): a shake near the limbs' own frequency must never fold a hip, shoulder or neck.
 * An arm rolling in toward her body stops at armIn while it hangs at her side (it would pass through her torso: on Miku a 36-degree lag
 * put the elbow 45 mm inside it, 12 degrees still 7 mm, 6 degrees leaves it 20 mm clear) and gets the full arms limit once it is held
 * forward or raised clear of her. */
const ROLL={legs:25*D,arms:45*D,armIn:6*D,head:30*D};
/** The head look (held, falling, landing): the head and neck turn toward a world point by what the eyes cannot reach, through a 2 Hz,
 * damping 0.9 spring that trails the fast eyes, at most 45 degrees of yaw and 30 of pitch, 40% at the neck and 60% at the head. Mate
 * Engine tracks the cursor with the head at +-45/30 and the eyes at +-12; gaze shifts under 20 degrees are mostly the eyes. */
const LOOK={frequency:2,damping:.9,yaw:45*D,pitch:30*D,eyes:{yaw:12,pitch:10}};
/** In the air (thrown), the swing and the limb lag relax critically damped: in free fall nothing swings her, and legs still swung out
 * at the contact would skate along the floor as they settle. The rate is 0.8 / the expected flight time (1.5-6 Hz), which leaves about
 * 5% of the swing at the contact (a 1.6 s flight relaxes at 1.5 Hz, a 0.2 s one at 4 Hz). */
const FLIGHT={frequency:1.5,damping:1,reach:1,max:6};
/** Where the shoe tips end: each toes bone carried on along its foot by this share of the foot-to-toes length (on Miku the screen-lowest
 * shoe vertex standing). Pointed toes put the tip below the toes bone, so planting it lands her tip first. */
export const SHOE_TIP=.4;
/** The shoe tips (new world points, one per foot with bones; without a toes bone, the toes or foot bone itself). */
export function shoeTips(humanoid){
 const tips=[];
 for(const side of ['left','right']){const toes=humanoid?.getRawBoneNode?.(side+'Toes'),foot=humanoid?.getRawBoneNode?.(side+'Foot');
  if(toes&&foot){const t=toes.getWorldPosition(new Vector3());tips.push(t.addScaledVector(foot.getWorldPosition(new Vector3()).sub(t),-SHOE_TIP));}else if(toes||foot)tips.push((toes||foot).getWorldPosition(new Vector3()));}
 return tips;
}
/** The camera safety net's margin around the skeleton's joints (m): the skin, shoes and hair reach at most 4 cm past them on Miku (the
 * catalog has the hair and finger joints). The net still zooms out whenever her legs and twintails swing past the 480 px canvas: on Miku
 * a 600-900 px carry in 0.4 s shows her at 84-85% for about 0.6 s, and a dizzying shake at 73-80% for up to 1.6 s (0.1 m was worse).
 * Avoiding it needs a canvas wider than her swing while held, or a pan of the canvas against the camera; neither exists yet. */
const MARGIN=.04;
/** A second-order follower in frequency (Hz) and damping ratio: x'' = w^2 (target - x) - 2 zeta w x'. */
class Spring{
 constructor(){this.x=0;this.v=0;}
 step(target,frequency,damping,h){const w=TAU*frequency;this.v+=(w*w*(target-this.x)-2*damping*w*this.v)*h;this.x+=this.v*h;return this.x;}
 reset(){this.x=0;this.v=0;}
 get still(){return Math.abs(this.x)<.05*D&&Math.abs(this.v)<.5*D;}
}
/** One critically damped step of x toward target (exact for any h). Returns [x, v]. */
function settle(x,v,target,w,h){const d=x-target,j=v+w*d,e=Math.exp(-w*h);return [target+(d+j*h)*e,(v-j*w*h)*e];}

/** The carry rig. Held by the scruff, her body swings as one piece from the grab point like something with weight:
 * - anchor: the grab point S (her upper chest, or the grabbed bone with style 'bone') keeps its world position, and with style 'scruff'
 *   the window slides over 0.25-0.6 s (slide(), in NDC) so S comes under the cursor. The anchor moves the raw hips, never vrm.scene:
 *   VRoid springs are simulated relative to a centre node under vrm.scene, so a scene translation would hide the body's real motion
 *   from the hair (on Miku a brisk drag flung her twintails over her head);
 * - one acceleration-driven pendulum: theta'' = -w^2 sin(theta) - 2 zeta w theta' - (a_x w^2 / g) gain cos(theta) + w^2 lean, shown
 *   through a tanh soft limit and applied as a roll about the view axis on the normalized hips BEFORE humanoid.update and the springs,
 *   so the hair and bust swing with the body (on Miku a carry swings her twintails 38-55 degrees, trailing the body by 20-28; the old
 *   root swing after the springs moved them about 6);
 * - limb lag springs (legs and arms catch up toward world-down a beat late), head stabilisation (a counter-roll), and a lift spring
 *   (lifted fast she straightens and points her toes; lowered fast her knees come up);
 * - set down, the swing, limb lag and lift turn critically damped over 0.1 s (0.06 s half-life: upright inside the handover), and over
 *   0.35 s with no jolt the hold on S hands over to the feet, which plant on the line she stood on until recoverySeconds after the
 *   contact (held clear of it while they still swing in under her, so they never slide along the floor); then the translation settles
 *   to 0 with a 0.1 s half-life. The plant is seen through the camera: the shoe tip lowest on screen stays on the floor line. In the air
 *   (release({flight:true}): a throw, or a set-down gliding onto the floor) the handover runs within the expected flight (0.1-0.35 s,
 *   and at the latest 0.08 s after the contact) while the swing and the limbs relax onto the body, critically damped and undriven, by
 *   the contact (airborne() re-times it after a bounce), so her lowest toe meets the line with no clearance (lifting her clear of
 *   swinging feet would bob her) and nothing swings her afterwards: the stop at the contact is the floor's, not a hand's;
 * - a head look toward a world point (the viewer, the holder, the floor), on top of the head stabilisation.
 * There is no per-bone gravity: the held pose is authored (avatar_held_pose.mjs). Only normalized humanoid bones and the raw hips
 * position change (humanoid.update rewrites that every frame; restore() puts it back too); vrm.scene, and so a VRM 0.x root turned by
 * PI, is never touched. */
export class AvatarPickup{
 constructor(vrm,catalog,onTrigger=()=>{}){
  this.vrm=vrm;this.catalog=catalog;this.onTrigger=onTrigger;this.vrm0=vrm.meta?.metaVersion==='0';
  this.active=false;this.held=false;this.ragdoll=false;this.saved=null;this.bone=null;this.restToe=null;
  this.theta=0;this.omega=0;this.displayed=0;this.legs=new Spring();this.arms=new Spring();this.head=new Spring();this.lift=new Spring();
  this.offset=new Vector3();this.offsetVelocity=new Vector3();this.slideTarget={x:0,y:0};this.slideTime=.25;this.age=0;this.releaseAge=0;this.groundAge=0;this.flying=false;this.h=0;this.handover=LETGO;this.airFrequency=FLIGHT.frequency;this.restFront=0;this.pv=new Vector3();this.pw=new Vector3();
  this.lookYaw=new Spring();this.lookPitch=new Spring();this.lq=new Quaternion();this.lv=new Vector3();this.lw=new Vector3();this.le=new Euler(0,0,0,'YXZ');
  this.camera=null;this.cameraRest=null;this.cameraActive=false;this.distance=0;
  this.q=new Quaternion();this.r=new Quaternion();this.v=new Vector3();this.anchored=new Vector3();this.previous=new Vector3();this.a=new Vector3();this.b=new Vector3();this.m=new Matrix4();this.bounds=new Box3();
 }
 /** The displayed swing angle (radians, counter-clockwise as the viewer sees it; positive swings her legs to screen right). */
 get angle(){return this.displayed;}
 /** bone: the grabbed humanoid bone; point: the press point in screen px (the ragdoll distance origin); pointer: the press point in
  * canvas NDC, which the scruff slide brings S to (default: the grabbed bone's projection). */
 start(bone,point,camera,{style='scruff',pointer=null}={}){
  const humanoid=this.vrm.humanoid,grabbed=humanoid.getRawBoneNode?.(bone)||null;
  const scruff=['upperChest','chest','spine','hips'].map(name=>humanoid.getRawBoneNode?.(name)).find(Boolean)||grabbed;
  this.style=style==='bone'&&grabbed?'bone':'scruff';this.bone=this.style==='bone'?grabbed:scruff;if(!this.bone)return;
  // The hips still carry last frame's offset here (the renderer handles pointer samples before restore()), so a grab during the
  // settle continues from where she is.
  this.vrm.scene.updateMatrixWorld(true);this.anchorPoint=this.bone.getWorldPosition(new Vector3());
  if(!this.active){const low=shoeTips(this.vrm.humanoid).reduce((a,b)=>!a||b.y<a.y?b:a,null);this.restToe=low?low.y:null;this.restFront=low?low.z:0;this.restAside=this.aside();this.theta=this.omega=this.displayed=0;for(const spring of [this.legs,this.arms,this.head,this.lift,this.lookYaw,this.lookPitch])spring.reset();this.offset.set(0,0,0);this.offsetVelocity.set(0,0,0);}
  this.slideTarget={x:0,y:0};this.slideTime=.25;this.anchorNDC=null;
  if(camera){
   this.camera=camera;if(!this.cameraActive){this.cameraRest={position:camera.position.clone(),quaternion:camera.quaternion.clone()};this.distance=camera.position.z-this.anchorPoint.z;}
   this.anchorNDC=this.anchorPoint.clone().project(camera);
   const target=this.style==='scruff'?pointer||(grabbed&&grabbed!==this.bone?grabbed.getWorldPosition(new Vector3()).project(camera):null):null;
   if(target){
    const dx=target.x-this.anchorNDC.x,dy=target.y-this.anchorNDC.y,tan=Math.tan(camera.fov*D/2),depth=camera.position.distanceTo(this.anchorPoint);
    this.slideTarget={x:dx,y:dy};this.slideTime=Math.min(.6,.25+.25*Math.hypot(dx*tan*camera.aspect*depth,dy*tan*depth)/.5);  // 0.25 s plus 0.25 s per 0.5 m
   }
  }
  this.origin={...point};this.last={...point};this.slow=0;this.active=true;this.held=true;this.flying=this.thrown=false;this.ragdoll=false;this.age=0;this.releaseAge=0;this.groundAge=0;
 }
 /** Pointer samples while held: only the opt-in ragdoll thresholds. */
 move(point,dt,settings){if(!this.active||!this.held)return;
  const speed=Math.hypot(point.x-this.last.x,point.y-this.last.y)/Math.max(.008,dt);this.last={...point};
  if(settings.ragdollEnabled&&!this.ragdoll&&(Math.hypot(point.x-this.origin.x,point.y-this.origin.y)>settings.distanceThreshold||speed>settings.speedThreshold)){this.ragdoll=true;this.slow=0;this.onTrigger('ragdoll');}
 }
 /** Let go: set down in place, or in the air until land() (flight: a throw, or a set-down gliding onto the floor). seconds: the expected
  * time to the contact; the hold on S hands over to the feet within it (0.1-0.35 s) and the swing relaxes by then (FLIGHT). */
 release({flight=false,seconds=Infinity}={}){if(!this.active||!this.held)return;this.held=false;this.flying=this.thrown=!!flight;this.releaseAge=0;this.groundAge=0;
  const t=Number.isFinite(seconds)&&seconds>0?seconds:null;this.handover=flight&&t?clamp(t,.1,LETGO):LETGO;this.airFrequency=flight&&t?clamp(FLIGHT.reach/t,FLIGHT.frequency,FLIGHT.max):FLIGHT.frequency;
  if(this.ragdoll){this.ragdoll=false;this.onTrigger('recover');}}
 /** In the air, the expected time to the contact changed (a wall or the top of the screen bounced her): the swing relaxes by then. */
 airborne(seconds){if(this.active&&this.flying&&Number.isFinite(seconds)&&seconds>0)this.airFrequency=clamp(FLIGHT.reach/seconds,FLIGHT.frequency,FLIGHT.max);}
 /** A thrown body's feet touched the floor: the set-down damping and the foot plant start now. */
 land(){if(!this.active||!this.flying)return;this.flying=false;this.groundAge=0;}
 restore(){if(!this.saved)return;this.saved.node.position.copy(this.saved.position);this.saved=null;}
 /** The canvas offset (NDC) that brings S under the cursor; the renderer adds it to the drag geometry every frame. */
 slide(){if(!this.active)return {x:0,y:0};const u=smooth(this.age/this.slideTime);return {x:this.slideTarget.x*u,y:this.slideTarget.y*u};}
 /** How far (m) her feet are to the side of S: about 0 standing, up to L sin(angle) while her body or legs swing. */
 aside(){
  const humanoid=this.vrm.humanoid;let x=0,n=0;
  for(const side of ['left','right']){const node=humanoid.getRawBoneNode?.(side+'Toes')||humanoid.getRawBoneNode?.(side+'Foot');if(node){x+=node.getWorldPosition(this.v).x;n++;}}
  return this.bone&&n?Math.abs(x/n-this.bone.getWorldPosition(this.v).x):0;
 }
 /** The lowest shoe tip (world y; shoeTips), or null without feet. */
 toeY(){const tips=shoeTips(this.vrm.humanoid);return tips.length?Math.min(...tips.map(t=>t.y)):null;}
 /** The plant: the world y offset that puts the shoe tip lowest on screen onto the line she stood on (restToe, at the depth her tips
  * stood at), seen through the camera. In perspective a tip nearer the camera shows lower, so feet bent or pointed toward the camera
  * (falling, landing) planted on the world line dipped up to 7 px into the taskbar; standing it is restToe - toeY(). Without a camera,
  * the world line. */
 plant(){
  const tips=shoeTips(this.vrm.humanoid),camera=this.camera;if(!tips.length||this.restToe===null)return 0;
  if(!camera)return this.restToe-Math.min(...tips.map(t=>t.y));
  let low=tips[0],lowY=Infinity;for(const t of tips){const y=this.pv.copy(t).project(camera).y;if(y<lowY){lowY=y;low=t;}}
  const tip=this.pv.copy(low).project(camera),line=this.pw.set(low.x,this.restToe,this.restFront).project(camera).y;
  return this.pw.set(tip.x,line,tip.z).unproject(camera).y-low.y;
 }
 /** Rolls a normalized bone about the world view axis through its joint: its parent's frame decides the local axis. */
 roll(node,angle){
  if(!node||!(Math.abs(angle)>1e-7))return;
  if(node.parent)node.parent.getWorldQuaternion(this.q);else this.q.identity();
  node.quaternion.premultiply(this.r.setFromAxisAngle(this.v.copy(VIEW).applyQuaternion(this.q.invert()),angle));
 }
 /** Adds a canonical x rotation (degrees) in a normalized bone's own frame: knee bend, toe point, hip flexion (x-). */
 bend(node,degrees){if(!node||!degrees)return;const q=this.r.setFromAxisAngle(this.v.set(1,0,0),degrees*D);node.quaternion.multiply(this.vrm0?flipVRM0(q):q);}
 /** Adds a canonical head turn (yaw about y, then pitch about x, radians) in a normalized bone's own frame. */
 turn(node,yaw,pitch){if(!node||!(Math.abs(yaw)+Math.abs(pitch)>1e-7))return;const q=this.r.setFromEuler(this.le.set(pitch,yaw,0,'YXZ'));node.quaternion.multiply(this.vrm0?flipVRM0(q):q);}
 /** The canonical yaw and pitch (radians; pitch positive looks down) from a normalized bone's facing to a world point, seen from where
  * the bone shows: the normalized rig never gets the anchor's translation (last frame's offset; held by the scruff it is 14 cm, which
  * made the holder look 20 degrees steeper than it is). */
 aim(node,point){
  node.getWorldQuaternion(this.lq);const d=this.lv.copy(point).sub(node.getWorldPosition(this.lw).add(this.offset)).applyQuaternion(this.lq.invert());
  if(this.vrm0){d.x=-d.x;d.z=-d.z;}  // a VRM 0.x rig faces -z in its own frame
  return {yaw:Math.atan2(d.x,d.z),pitch:Math.atan2(-d.y,Math.hypot(d.x,d.z))};
 }
 /** After MotionEngine, before humanoid.update. motion: window acceleration and velocity (px/s^2, px/s) from AvatarInertia, the
  * world size of a pixel, the hold (HoldBehaviour's step): dizzy or limp makes the body limp and the head unsteady; look: a world point
  * for the head look (null: none), eyeReach {yaw, pitch} (degrees the eyes cover before the head joins). */
 pose(settings,dt,{ax=0,ay=0,vx=0,vy=0,worldPerPixel=0,hold=null,look=null,eyeReach=LOOK.eyes}={}){
  if(!this.active)return;if(!settings.enabled){this.stop();return;}
  const h=clamp(dt,0,.1);this.h=h;this.age+=h;if(!this.held)this.releaseAge+=h;if(!this.held&&!this.flying)this.groundAge+=h;
  // The opt-in ragdoll recovers once the window has stayed under the speed threshold for 1 s (or on release).
  if(this.ragdoll){this.slow=Math.hypot(vx,vy)<settings.speedThreshold?this.slow+h:0;if(this.slow>=1||!settings.ragdollEnabled){this.ragdoll=false;this.origin={...this.last};this.onTrigger('recover');}}
  const limp=this.ragdoll?'ragdoll':hold?.state==='dizzy'||hold?.pose?.limp>.5?'limp':'';  // the first 0.08 s after the grab: a hit-stop
  const node=name=>this.vrm.humanoid.getNormalizedBoneNode?.(name)||null,neck=node('neck');
  // Set down (released), the dynamics turn critically damped toward upright over the first 0.1 s (r 0 to 1), and the limbs stop chasing
  // the body and settle on their own, so the swing, limb lag and lift settle from where they are with no jolt and no rocking on planted
  // feet: upright inside the 0.35 s handover. In the air nothing drives the swing (free fall) and it relaxes; on landing it is a set-down.
  // After a flight's contact nothing drives her either: the window's stop at the contact is the floor's, not a hand's (it kicked the
  // swing from -3.5 to 16 degrees and skated her feet 64 px).
  const r=this.held||this.flying?0:smoother(this.groundAge/SET_RAMP),mix=(held,down)=>held+(down-held)*r,fDown=SET_DOWN/TAU,air=this.flying,fAir=this.airFrequency,undriven=air||this.thrown&&!this.held;
  const w=TAU*mix(air?fAir:settings.swingFrequency,fDown),zeta=mix(air?FLIGHT.damping:settings.swingDamping,1),limit=settings.swingLimit*D,carry=undriven?0:(1-r)*(this.age>HIT_STOP?1:0);
  const a=carry*clamp(ax*worldPerPixel,-3*G,3*G)*settings.carryGain,lean=-carry*clamp(vx/1000,-1,1)*settings.airLean*D,lift=undriven?0:(1-r)*clamp(-ay*worldPerPixel/G,-1,1);
  const limb=settings.limbFrequency*(limp==='ragdoll'?.4:limp?.6:1),frequency=mix(air?Math.max(limb,fAir):limb,fDown),damping=air?1:mix(limp==='ragdoll'?.15:settings.limbDamping,1),steady=limp?0:settings.headStabilize;
  const shown=theta=>limit>0?limit*Math.tanh(theta/limit):0,step=h/SUBSTEPS;
  for(let i=0;i<SUBSTEPS;i++){
   this.omega+=(-w*w*Math.sin(this.theta)-2*zeta*w*this.omega-(a*w*w/G)*Math.cos(this.theta)+w*w*lean)*step;
   this.theta+=this.omega*step;if(Math.abs(this.theta)>1.4){this.theta=Math.sign(this.theta)*1.4;this.omega=0;}  // never over the top
   const angle=(1-r)*shown(this.theta),chase=air?0:1;  // in the air the limbs no longer chase world-down: they relax onto the body with it
   this.legs.step(-.85*chase*angle,frequency,damping,step);this.arms.step(-.9*chase*angle,mix(frequency*.83,fDown),air?1:mix(damping*.85,1),step);this.head.step(-steady*angle,mix(3,fDown),mix(.7,1),step);this.lift.step(lift,mix(2.5,fDown),mix(.5,1),step);
  }
  this.displayed=shown(this.theta);
  const legs=soft(this.legs.x,ROLL.legs),head=soft(this.head.x,ROLL.head);
  // Per arm: the roll toward her body (negative for the arm on screen right) is limited by how clear of her torso the arm is.
  const hips=node('hips')?.getWorldPosition(this.a).x??0,arm=side=>{
   const upper=node(side+'UpperArm'),lower=node(side+'LowerArm'),x=this.arms.x;if(!upper||!lower)return soft(x,ROLL.arms);
   const at=upper.getWorldPosition(this.b),d=lower.getWorldPosition(this.v).sub(at).normalize(),inward=(at.x>hips?-1:1)*x>0;
   return soft(x,inward?ROLL.armIn+(ROLL.arms-ROLL.armIn)*smooth(Math.max(d.z,d.y)/.45):ROLL.arms);
  },arms={left:arm('left'),right:arm('right')};
  this.roll(node('hips'),this.displayed);
  for(const side of ['left','right']){this.roll(node(side+'UpperLeg'),legs);this.roll(node(side+'UpperArm'),arms[side]);}
  this.roll(neck,.4*head);this.roll(node('head'),(neck?.6:1)*head);
  // The head look, measured from the head as posed and steadied so far (MotionEngine rewrites it every frame: nothing accumulates).
  const skull=node('head');let lookYaw=0,lookPitch=0;
  if(look&&skull){const e=this.aim(skull,look),ry=clamp(eyeReach?.yaw??LOOK.eyes.yaw,0,90)*D,rp=clamp(eyeReach?.pitch??LOOK.eyes.pitch,0,90)*D;
   lookYaw=clamp(Math.sign(e.yaw)*Math.max(0,Math.abs(e.yaw)-ry),-LOOK.yaw,LOOK.yaw);lookPitch=clamp(Math.sign(e.pitch)*Math.max(0,Math.abs(e.pitch)-rp),-LOOK.pitch,LOOK.pitch);}
  for(let i=0;i<SUBSTEPS;i++){this.lookYaw.step(lookYaw,LOOK.frequency,LOOK.damping,step);this.lookPitch.step(lookPitch,LOOK.frequency,LOOK.damping,step);}
  const yaw=clamp(this.lookYaw.x,-LOOK.yaw,LOOK.yaw),pitch=clamp(this.lookPitch.x,-LOOK.pitch,LOOK.pitch);
  this.turn(neck,.4*yaw,.4*pitch);this.turn(skull,(neck?.6:1)*yaw,(neck?.6:1)*pitch);
  const up=this.lift.x;
  if(Math.abs(up)>1e-3)for(const side of ['left','right']){
   const knee=node(side+'LowerLeg');
   if(up>0){const bent=knee?2*Math.acos(Math.min(1,Math.abs(knee.quaternion.w)))/D:0;this.bend(knee,-Math.min(10*up,bent));this.bend(node(side+'Foot'),10*up);}  // straighten, never past straight
   else{this.bend(node(side+'UpperLeg'),10*up);this.bend(knee,-10*up);}
  }
 }
 /** After humanoid.update, before the springs: holds S on its anchor while held; once released, eases from that hold to planting the
  * feet on the line they stood on (smootherstep over 0.35 s, so the translation keeps its speed and acceleration: no pop), plants them
  * until settings.recoverySeconds, then settles to 0. Then the camera safety net. */
 anchor(settings){
  if(!this.active)return;
  const root=this.vrm.scene,hips=this.vrm.humanoid.getRawBoneNode?.('hips'),moved=hips?.parent?hips:root;  // without raw hips (a bare fixture), the root
  this.saved={node:moved,position:moved.position.clone()};
  root.updateMatrixWorld(true);this.previous.copy(this.offset);
  const anchored=this.anchored.copy(this.anchorPoint).sub(this.bone.getWorldPosition(this.v));
  if(this.held)this.offset.copy(anchored);
  else if(this.flying||this.groundAge<settings.recoverySeconds){
   // Set down, feet still swinging under her stay clear of the floor until they are below her; thrown, the lowest toe meets it, and the
   // handover completes within 0.08 s of the contact (a short drop lands before the hand's hold has let go).
   const clear=this.thrown?0:Math.min(.15,.5*Math.max(0,this.aside()-(this.restAside||0))),u=Math.max(smoother(this.releaseAge/this.handover),this.thrown&&!this.flying?smoother(this.groundAge/CONTACT):0);
   this.offset.set(0,this.plant()+clear,0).lerp(anchored,1-u);
  }
  else for(const axis of ['x','y','z'])[this.offset[axis],this.offsetVelocity[axis]]=settle(this.offset[axis],this.offsetVelocity[axis],0,SETTLE,this.h);
  if(this.held||this.flying||this.groundAge<settings.recoverySeconds)this.offsetVelocity.copy(this.offset).sub(this.previous).divideScalar(Math.max(1e-4,this.h)).clampLength(0,3);
  // The world offset in the moved node's parent space: S moves rigidly with the hips, so this places it exactly.
  const inverse=this.m.copy(moved.parent?.matrixWorld||new Matrix4()).invert(),at=moved.getWorldPosition(this.a);
  moved.position.add(this.b.copy(at).add(this.offset).applyMatrix4(inverse).sub(at.applyMatrix4(inverse)));root.updateMatrixWorld(true);
  this.frame();
  if(!this.held&&!this.flying&&this.groundAge>=settings.recoverySeconds&&Math.abs(this.theta)<.2*D&&Math.abs(this.omega)<1*D&&[this.legs,this.arms,this.head,this.lift,this.lookYaw,this.lookPitch].every(s=>s.still)&&this.offset.length()<5e-4&&this.offsetVelocity.length()<1e-3&&!this.cameraActive)this.finish();
 }
 /** Safety net: zoom out about S (its NDC unchanged) when the swinging skeleton would leave the canvas; ease back once it fits. */
 frame(){
  const camera=this.camera;if(!camera||!this.cameraRest||!this.anchorNDC)return;
  const rest=this.cameraRest.position.z-this.anchorPoint.z,tan=Math.tan(camera.fov*D/2),tx=tan*camera.aspect,x=clamp(this.anchorNDC.x,-.95,.95),y=clamp(this.anchorNDC.y,-.95,.95),p=this.anchorPoint;
  let desired=rest;
  if(this.held){
   const bounds=this.bounds.makeEmpty();for(const e of this.catalog.entries)bounds.expandByPoint(e.node.getWorldPosition(this.v));bounds.expandByScalar(MARGIN);
   desired=Math.max(rest,(bounds.max.x-p.x)/(tx*(1-x)),(p.x-bounds.min.x)/(tx*(1+x)),(bounds.max.y-p.y)/(tan*(1-y)),(p.y-bounds.min.y)/(tan*(1+y)));
  }
  if(!this.cameraActive&&desired<=rest+1e-6)return;
  if(desired>this.distance){this.distance=desired;this.distanceVelocity=0;}else[this.distance,this.distanceVelocity]=settle(this.distance,this.distanceVelocity||0,desired,1.678/.3,this.h);  // out at once, back critically damped (0.3 s half-life)
  if(this.distance-rest<1e-4){this.distance=rest;this.distanceVelocity=0;this.cameraActive=false;camera.position.copy(this.cameraRest.position);camera.quaternion.copy(this.cameraRest.quaternion);camera.updateMatrixWorld(true);return;}
  this.cameraActive=true;const d=Math.max(.1,this.distance);
  camera.position.set(p.x-x*d*tx,p.y-y*d*tan,p.z+d);camera.quaternion.copy(this.cameraRest.quaternion);camera.updateMatrixWorld(true);
 }
 finish(){this.active=false;this.held=false;this.flying=false;this.ragdoll=false;this.offset.set(0,0,0);this.offsetVelocity.set(0,0,0);this.theta=this.omega=this.displayed=0;for(const spring of [this.legs,this.arms,this.head,this.lift,this.lookYaw,this.lookPitch])spring.reset();}
 /** Pickup switched off mid-hold: everything back at once. */
 stop(){if(this.ragdoll)this.onTrigger('recover');this.restore();this.finish();if(this.camera&&this.cameraRest&&this.cameraActive){this.camera.position.copy(this.cameraRest.position);this.camera.quaternion.copy(this.cameraRest.quaternion);this.camera.updateMatrixWorld(true);}this.cameraActive=false;}
}
