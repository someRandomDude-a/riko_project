export const CATALOG_KEY='riko:avatar-catalog';
export const studioDefaults={avatarStudioProfiles:{}};
const object=value=>value&&typeof value==='object'&&!Array.isArray(value)?value:{};
const number=(value,fallback,min,max)=>typeof value==='number'&&Number.isFinite(value)?Math.max(min,Math.min(max,value)):fallback;
const color=(value,fallback)=>typeof value==='string'&&/^#[a-f\d]{6}$/i.test(value)?value:fallback;
const vector=(value,fallback,min,max)=>[0,1,2].map(i=>number(value?.[i],fallback[i],min,max));
const choice=(value,options)=>options.some(([key])=>key===value)?value:options[0][0];
const numbers=(raw,fields)=>Object.fromEntries(fields.map(([key,,min,max,,fallback])=>[key,number(raw?.[key],fallback,min,max)]));
/** Shared by the studio controls and settings search: [key, label, min, max, step, default]. The pickup keys are new names, so values
 * stored for the old per-bone gravity rig (gravity, damping, bodySpring, jointLimit, inertia, maxSwing: 180) never come back.
 * recoverySeconds keeps its key: it is now the landing and recovery time, counted from when her feet touch down. */
export const PICKUP_FIELDS=[
 ['scruffPitch','Scruff tilt (degrees)',0,60,1,30],['swingFrequency','Swing frequency (Hz)',.3,3,.05,1],['swingDamping','Swing damping ratio',.05,1.5,.01,.28],
 ['swingLimit','Swing limit (degrees)',0,90,1,35],['carryGain','Carry strength (0 = moving her never swings her)',0,3,.05,1],['airLean','Air lean while moving (degrees)',0,30,1,10],
 ['limbFrequency','Limb follow frequency (Hz)',.5,5,.05,1.8],['limbDamping','Limb follow damping ratio',.05,1.5,.01,.35],['headStabilize','Head steadiness (0 = swings with the body)',0,1,.05,.65],
 ['throwSpeed','Throw speed: a flick at least this fast throws her (px/s)',200,5000,50,900],['fallGravity','Fall gravity (1 = real gravity)',.2,3,.05,.7],
 ['landingSquash','Landing squash (0 = none)',0,2,.05,1],['recoverySeconds','Landing and recovery time (seconds)',.5,10,.1,2.5],['distanceThreshold','Ragdoll distance threshold (px)',10,3000,10,350],['speedThreshold','Ragdoll speed threshold (px/s)',50,10000,50,1400],
 ['bounce','Hair & cloth bounce when moved (0 = off)',0,2,.05,1]];
export const PICKUP_TOGGLES=[['enabled','Enable pickup'],['ragdollEnabled','Go limp past the ragdoll thresholds']];
export const GRAB_STYLES=[['scruff','By the scruff (she slides so you hold her upper back)'],['bone','Where you grab her']];
export const RELEASE_STYLES=[['throw','A fast flick throws her; a gentle release sets her down'],['place','Always set her down where you let go'],['fall','Always drop her to the floor']];
export const GAZE_FIELDS=[['strength','Convergence strength',0,3,.01,1],['smoothing','Gaze smoothing response (without eye jumps)',1,30,.01,10],['minDistance','Minimum eye target distance (m)',.05,1,.01,.15],['maxYaw','Maximum gaze yaw (degrees)',0,90,.01,45],['maxPitch','Maximum gaze pitch (degrees)',0,60,.01,30],['maxConvergence','Maximum convergence per eye (degrees)',0,30,.01,15]];
export const GAZE_TOGGLES=[['enabled','Enable mouse eye tracking'],['convergence','Enable distance-based pupil convergence'],['saccades','Quick eye jumps (saccades) instead of gliding']];
export const REACTION_FIELDS=[['protestSeconds','Seconds held still before she protests',2,60,1,6],['resignSeconds','Seconds held still before she gives up',5,120,1,18],['kickAmplitude','Kick size (degrees)',0,45,1,22]];
export const REACTION_TOGGLES=[['enabled','Enable held reactions']];
export const FACE_FIELDS=[['restingSmile','Resting smile (no emotion, neutral or calm)',0,.6,.01,.15],['intensity','Reaction expression strength (0 = off)',0,1.5,.01,1]];
export function normalizeBone(value){
 const v=object(value);
 return {enabled:v.enabled===true,amount:number(v.amount,.5,0,1),frequency:number(v.frequency,3,.2,20),damping:number(v.damping,.7,0,2),maxAngle:number(v.maxAngle,20,0,90),offset:vector(v.offset,[0,0,0],-.5,.5),springOverride:v.springOverride===true,stiffness:number(v.stiffness,1,0,10),dragForce:number(v.dragForce,.4,0,1),gravityPower:number(v.gravityPower,0,0,5),gravityDir:vector(v.gravityDir,[0,-1,0],-1,1),hitRadius:number(v.hitRadius,.02,0,.5)};
}
export function normalizeProfile(value){
  const v=object(value),bones={};
 for(const [key,rule] of Object.entries(object(v.bones)).slice(0,2048))if(key.length<=512)bones[key]=normalizeBone(rule);
  return {bones,debug:v.debug===true,debugBones:Array.isArray(v.debugBones)?v.debugBones.filter(x=>typeof x==='string'&&x.length<=512).slice(0,2048):[],tPose:v.tPose===true,springPickRadius:number(v.springPickRadius,.025,.005,.2),
  mouseSphere:{enabled:v.mouseSphere?.enabled!==false,debug:v.mouseSphere?.debug===true,radius:number(v.mouseSphere?.radius,.15,.01,1),depth:number(v.mouseSphere?.depth,.08,-.5,.5),strength:number(v.mouseSphere?.strength,5,0,50),dragSpring:number(v.mouseSphere?.dragSpring,25,0,100),frequency:number(v.mouseSphere?.frequency,5,.5,20),damping:number(v.mouseSphere?.damping,.8,.1,2),bounce:number(v.mouseSphere?.bounce,.18,0,.5),bounceSeconds:number(v.mouseSphere?.bounceSeconds,.35,.1,2),maxAngle:number(v.mouseSphere?.maxAngle,35,0,90)},
  input:{dragThreshold:number(v.input?.dragThreshold,6,2,50),hoverDelay:number(v.input?.hoverDelay,600,0,5000)},
  gaze:{...Object.fromEntries(GAZE_TOGGLES.map(([key])=>[key,v.gaze?.[key]!==false])),...numbers(v.gaze,GAZE_FIELDS)},
  face:numbers(v.face,FACE_FIELDS),
  pickup:{enabled:v.pickup?.enabled!==false,grabStyle:choice(v.pickup?.grabStyle,GRAB_STYLES),release:choice(v.pickup?.release,RELEASE_STYLES),ragdollEnabled:v.pickup?.ragdollEnabled===true,...numbers(v.pickup,PICKUP_FIELDS)},
  reactions:{enabled:v.reactions?.enabled!==false,...numbers(v.reactions,REACTION_FIELDS)},
  lighting:{ambient:number(v.lighting?.ambient,2,0,5),sky:color(v.lighting?.sky,'#ffffff'),ground:color(v.lighting?.ground,'#443355'),lights:(Array.isArray(v.lighting?.lights)?v.lighting.lights:[]).slice(0,8).map(light=>({type:light?.type==='directional'?'directional':'point',color:color(light?.color,'#ffffff'),intensity:number(light?.intensity,1,0,20),position:vector(light?.position,[1,2,2],-10,10)}))},
  effect:{preset:['none','warm','cool','monochrome'].includes(v.effect?.preset)?v.effect.preset:'none',exposure:number(v.effect?.exposure,1,0,3),saturation:number(v.effect?.saturation,1,0,2),glslEnabled:v.effect?.glslEnabled===true,glsl:typeof v.effect?.glsl==='string'&&v.effect.glsl.length<=16384?v.effect.glsl:''}};
}
export function normalizeStudio(raw){
 const profiles={};for(const [key,value] of Object.entries(object(raw.avatarStudioProfiles)).slice(0,32))if(key.length<=2048)profiles[key]=normalizeProfile(value);
 return {avatarStudioProfiles:profiles};
}
export function importStudioPreset(text){
 if(text.length>262144)throw new Error('Preset exceeds 256 KiB');
 const value=JSON.parse(text);
 if(value?.format!=='riko-avatar-effects'||value.version!==1||!value.lighting||!value.effect)throw new Error('Expected a riko-avatar-effects version 1 preset');
 const profile=normalizeProfile(value);
 // A preset never silently enables imported GPU code.
 profile.effect.glslEnabled=false;
 return {lighting:profile.lighting,effect:profile.effect};
}
export function importGLSL(text){
 if(text.length>16384)throw new Error('GLSL exceeds 16 KiB');
 if(!/vec4\s+avatarEffect\s*\(\s*vec4\s+\w+\s*,\s*vec2\s+\w+\s*,\s*float\s+\w+\s*\)/.test(text))throw new Error('Define vec4 avatarEffect(vec4 color, vec2 uv, float time)');
 if(/#\s*(?:include|extension|version)|\b(?:while|for|do|discard)\b/.test(text))throw new Error('Includes, extensions, loops and discard are not supported');
 return text;
}
export function readCatalog(){try{const v=JSON.parse(localStorage.getItem(CATALOG_KEY)||'null');return typeof v?.key==='string'&&typeof v.source==='string'&&Array.isArray(v.bones)&&v.bones.every(e=>typeof e?.id==='string'&&typeof e.name==='string')?v:null;}catch{return null;}}
