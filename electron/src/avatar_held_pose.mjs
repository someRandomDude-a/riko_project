import {Quaternion,Euler} from 'three';

/** Authored held poses in canonical VRM 1.0 axes (MotionEngine flips them once for VRM 0.x), as Euler degrees in 'XYZ' order.
 * Canonical axes: hips and spine x+ bend forward; upperLeg x- flexes the hip and x+ extends it; lowerLeg x+ bends the knee;
 * foot x+ points the toes; leftUpperArm z- and rightUpperArm z+ lower the arms; an arm x- swings forward; leftLowerArm y- and
 * rightLowerArm y+ bend the elbows; head x+ looks down; a left shoulder z+ (right z-) shrugs.
 * The scruff hang (Desktop Mate, Mate Engine's PET_DRAGGING): the hips pitch the whole torso forward as one piece, the thighs
 * counter-pitch so the legs hang along world-down with soft, uneven knees and pointed toes, the hands come in front of the chest
 * and the neck and head lift so she faces the viewer. The carry rig's pendulum and limb lag (avatar_pickup.mjs) ride on top. */
export const HELD_BONES=['hips','spine','chest','upperChest','neck','head','leftShoulder','rightShoulder','leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','leftHand','rightHand','leftUpperLeg','rightUpperLeg','leftLowerLeg','rightLowerLeg','leftFoot','rightFoot'];
/** Joint ranges every held pose stays inside (degrees): knee and elbow flexion, hip flexion and extension. */
export const JOINT_LIMITS={knee:[0,140],elbow:[0,150],hipFlex:[0,120],hipExtend:[0,30]};
const D=Math.PI/180,TAU=2*Math.PI;
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,Number.isFinite(v)?v:0));
const zero=()=>Object.fromEntries(HELD_BONES.map(b=>[b,[0,0,0]]));
const add=(pose,table,w=1)=>{if(w)for(const [b,v] of Object.entries(table)){const t=pose[b]||(pose[b]=[0,0,0]);t[0]+=v[0]*w;t[1]+=v[1]*w;t[2]+=v[2]*w;}return pose;};
const blend=(pose,table,w)=>{for(const b of new Set([...Object.keys(pose),...Object.keys(table)])){const t=pose[b]||(pose[b]=[0,0,0]),v=table[b]||[0,0,0];for(let i=0;i<3;i++)t[i]+=(v[i]-t[i])*w;}return pose;};
const both=(name,[x,y,z])=>({['left'+name]:[x,y,z],['right'+name]:[x,-y,-z]});

/** Hands in front of the chest, knees uneven (the right one bends more; the mirror moves it to the trailing side), toes pointed.
 * The arms are measured on Miku against what idle, walking and her bind pose already show: 5 degrees of forward flexion past the
 * hang (more pulls the shirt's armpit, 2.2x at 15) and a 75-degree elbow (VRoid inner-elbow skin stretches with every degree of bend:
 * 2.4x at 75, 2.8x at 110; today's 'thinking' pose bends 72). */
export const tuckPose=P=>({hips:[P,0,0],neck:[-.4*P,0,0],head:[-.6*P,0,0],...both('Shoulder',[0,0,8]),...both('UpperArm',[-P-5,0,-72]),...both('LowerArm',[0,-75,0]),...both('Hand',[0,0,-15]),
 leftUpperLeg:[-P+1,0,1],rightUpperLeg:[-P-1,0,-1],leftLowerLeg:[12,0,0],rightLowerLeg:[20,0,0],...both('Foot',[20,0,0])});
/** The tuck, with her right hand raised to her shoulder toward whoever holds her (the upper arm about level: a 150-degree reach
 * stretched the shirt 4.4x past anything she shows standing). */
export const reachPose=P=>({...tuckPose(P),rightUpperArm:[-P-50,0,25],rightLowerArm:[0,50,0],rightHand:[0,0,0]});
/** Resigned or dizzy: arms hanging toward world-down, soft elbows and knees, the head half lifted. */
export const limpPose=P=>({hips:[P,0,0],neck:[-.3*P,0,0],head:[-.4*P,0,0],...both('UpperArm',[-P,0,-80]),...both('LowerArm',[0,-15,0]),...both('UpperLeg',[-P,0,1]),...both('LowerLeg',[8,0,0]),...both('Foot',[25,0,0])});
/** Added with weight pose.startle on the grab: the head ducks, shoulders rise, thighs tuck 10 (no extra elbow bend: see the tuck). */
export const STARTLE={head:[8,0,0],...both('Shoulder',[0,0,10]),...both('UpperLeg',[-10,0,0])};
/** Carried fast: arms 20 degrees further out for balance. */
export const ALARM={...both('UpperArm',[0,0,20])};
/** A sulking kick, led by the knees: thighs in antiphase (0.45 A back, 0.18 A forward), each knee kicking up to 2.5 A and lagging its
 * thigh by 70 degrees (never hyperextending), arms pushing 7.5 degrees (at A 22) 90 degrees off. Knee-led because a thigh split
 * stretches the crotch of VRoid bottoms (on Miku a 22-degree thigh kick took them from 3x to 5.3x edge stretch; this keeps them at the
 * hanging pose's level); a 15-degree arm push stretched the shirt's armpit 2.75x. */
export function kickPose(phase,amplitude=22){
 const s=Math.sin(phase),thigh=x=>-amplitude*(x>0?.18:.45)*x,knee=p=>2.5*amplitude*Math.max(0,Math.sin(p-70*D)),arm=7.5*amplitude/22*Math.sin(phase+90*D);
 return {leftUpperLeg:[thigh(s),0,0],rightUpperLeg:[thigh(-s),0,0],leftLowerLeg:[knee(phase),0,0],rightLowerLeg:[knee(phase+Math.PI),0,0],leftUpperArm:[arm,0,0],rightUpperArm:[-arm,0,0]};
}
/** Falling (thrown or dropped): arms flung out just below level for balance with bent elbows and wrists up, the back arched a little,
 * knees bent unevenly and toes pointed, flailing a little (arms at 2.2 Hz, legs in antiphase). The head-look (avatar_pickup.mjs) turns
 * her face to the floor. Not arms over the head: on Miku any arm raised past level stretches the shirt's armpit 3-4.4x past anything
 * idle or walking shows (38 degrees up: 4.1x); out 24-36 degrees below level it stays at 2.3-2.6x (the bind T-pose itself is 2.2x). */
export function fallPose(time=0){
 const a=Math.sin(TAU*2.2*time),b=Math.sin(TAU*2.2*time+1.3),l=Math.sin(TAU*1.7*time);
 return {hips:[-6,0,0],spine:[-4,0,0],chest:[-3,0,0],neck:[3,0,0],head:[6,0,0],...both('Shoulder',[0,0,12]),leftUpperArm:[-25+5*a,0,-30+6*b],rightUpperArm:[-25-5*b,0,30+6*a],
  ...both('LowerArm',[0,-45,0]),...both('Hand',[0,0,25]),leftUpperLeg:[-28+5*l,0,2],rightUpperLeg:[-20-5*l,0,-2],leftLowerLeg:[32+6*l,0,0],rightLowerLeg:[44-6*l,0,0],...both('Foot',[25,0,0])};
}
/** Landing or settling with weight s (knees, hips and ankles flex together so planted feet stay flat; below 0 a small stretch). */
export function landingPose(s){
 const k=Math.max(0,s),u=Math.max(0,-s);
 return {spine:[8*k,0,0],head:[6*k-20*u,0,0],...both('UpperLeg',[-22*k,0,0]),...both('LowerLeg',[45*k,0,0]),...both('Foot',[-23*k+33*u,0,0])};
}
/** Breathing: the chest rises and falls; the shoulders lift on the inhale. depth in degrees, phase in radians. */
export const breathPose=(phase,depth)=>({chest:[-depth*Math.sin(phase),0,0],upperChest:[-depth*.6*Math.sin(phase),0,0],...both('Shoulder',[0,0,depth*.8*Math.max(0,Math.sin(phase))])});

/** Keeps knees, elbows and hips inside JOINT_LIMITS whatever the settings (a big kick with no scruff tilt would extend the hip past 30). */
function limit(table){
 for(const s of ['left','right']){const sign=s==='left'?-1:1,knee=table[s+'LowerLeg'],elbow=table[s+'LowerArm'],hip=table[s+'UpperLeg'];
  if(knee)knee[0]=clamp(knee[0],...JOINT_LIMITS.knee);if(elbow)elbow[1]=sign*clamp(sign*elbow[1],...JOINT_LIMITS.elbow);if(hip)hip[0]=clamp(hip[0],-JOINT_LIMITS.hipFlex[1],JOINT_LIMITS.hipExtend[1]);}
 return table;
}
/** Left and right swapped and y, z negated: the same pose in a mirror. Its own inverse. */
export function mirrorPose(pose){
 const out={};for(const [b,v] of Object.entries(pose)){const m=b.startsWith('left')?'right'+b.slice(4):b.startsWith('right')?'left'+b.slice(5):b;out[m]=[v[0],-v[1]||0,-v[2]||0];}
 return out;
}
const LEGS=['UpperLeg','LowerLeg','Foot'],SIGNS=[1,-1,-1];
/** The legs blended toward their mirror image by u (0 as authored, 1 swapped). Only the legs mirror, so the more bent knee trails the
 * drag while the arms (a raised hand) stay put, and a continuous u crossfades a change of direction instead of snapping. */
function mirrorLegs(table,u){
 if(!(u>0))return table;
 for(const part of LEGS){const l=table['left'+part],r=table['right'+part];if(!l||!r)continue;
  for(let i=0;i<3;i++){const a=l[i],b=r[i];l[i]=a+(SIGNS[i]*b-a)*u;r[i]=b+(SIGNS[i]*a-b)*u;}}
 return table;
}
export const toQuaternions=pose=>new Map(Object.entries(pose).map(([b,[x,y,z]])=>[b,new Quaternion().setFromEuler(new Euler(x*D,y*D,z*D,'XYZ'))]));

/** The held pose as Euler degrees. hold is HoldBehaviour.step's result (state and pose: variant, startle, alarm, limp, kick, kickPhase,
 * sway, fall, mirror) or a stand-in; time drives the playful and idle leg sways and the falling flail. mirror runs continuously from 1
 * (as authored) to -1 (the legs swapped). */
export function heldTable(hold,time=0,{pitch=30,kickAmplitude=22}={}){
 const P=clamp(pitch,0,60),pose=hold?.pose||{},breath=hold?.breath;
 const table=add(zero(),pose.variant==='reach'?reachPose(P):tuckPose(P));
 if(pose.variant==='swing'){const s=Math.sin(TAU*.8*time);add(table,{leftUpperLeg:[-10*s,0,0],rightUpperLeg:[10*s,0,0],leftLowerLeg:[8+8*s,0,0],rightLowerLeg:[8-8*s,0,0]});}
 add(table,ALARM,clamp(pose.alarm));
 const limp=clamp(pose.limp),kick=clamp(pose.kick);
 if(limp)blend(table,limpPose(P),limp);
 const sway=clamp(pose.sway)*(1-limp)*(1-kick);if(sway){const s=Math.sin(TAU*.45*time);add(table,{...both('UpperLeg',[4*s,0,0]),...both('LowerLeg',[2-2*s,0,0])},sway);}
 add(table,STARTLE,clamp(pose.startle));
 if(kick)add(table,kickPose(pose.kickPhase||0,clamp(kickAmplitude,0,45)),kick);
 if(breath?.depth)add(table,breathPose(breath.phase||0,breath.depth));
 const fall=clamp(pose.fall);if(fall)blend(table,fallPose(time),fall);
 limit(table);
 return mirrorLegs(table,clamp((1-(pose.mirror??1))/2));
}
/** heldTable as canonical quaternions: Map(bone -> Quaternion). */
export const heldPose=(hold,time,options)=>toQuaternions(heldTable(hold,time,options));
/** Multiplies landingPose(s) onto a canonical pose map (used for the in-place settle after a release). */
export function settlePose(base,s){
 if(!s)return base;
 for(const [bone,q] of toQuaternions(landingPose(s))){const current=base.get(bone);base.set(bone,current?current.clone().multiply(q):q);}
 return base;
}
