export const CATALOG_KEY='riko:avatar-catalog';
export const studioDefaults={avatarStudioProfiles:{}};
const object=value=>value&&typeof value==='object'&&!Array.isArray(value)?value:{};
const number=(value,fallback,min,max)=>typeof value==='number'&&Number.isFinite(value)?Math.max(min,Math.min(max,value)):fallback;
const color=(value,fallback)=>typeof value==='string'&&/^#[a-f\d]{6}$/i.test(value)?value:fallback;
const vector=(value,fallback,min,max)=>[0,1,2].map(i=>number(value?.[i],fallback[i],min,max));
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
  gaze:{enabled:v.gaze?.enabled!==false,convergence:v.gaze?.convergence!==false,strength:number(v.gaze?.strength,1,0,3),smoothing:number(v.gaze?.smoothing,10,1,30),minDistance:number(v.gaze?.minDistance,.15,.05,1),maxYaw:number(v.gaze?.maxYaw,45,0,90),maxPitch:number(v.gaze?.maxPitch,30,0,60),maxConvergence:number(v.gaze?.maxConvergence,15,0,30)},
  pickup:{enabled:v.pickup?.enabled!==false,gravity:number(v.pickup?.gravity,9.8,0,30),damping:number(v.pickup?.damping,2,0,10),bodySpring:number(v.pickup?.bodySpring,12,1,80),inertia:number(v.pickup?.inertia,.5,0,3),maxSwing:number(v.pickup?.maxSwing,180,0,180),jointLimit:number(v.pickup?.jointLimit,60,0,120),ragdollEnabled:v.pickup?.ragdollEnabled!==false,distanceThreshold:number(v.pickup?.distanceThreshold,350,10,3000),speedThreshold:number(v.pickup?.speedThreshold,1400,50,10000),recoverySeconds:number(v.pickup?.recoverySeconds,2,.2,10)},
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
