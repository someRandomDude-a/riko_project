import {Quaternion,Vector3,SkeletonHelper} from 'three';
import {CATALOG_KEY} from './avatar_studio_settings.mjs';

/** Stable hierarchy IDs, including non-humanoid skin/spring bones and duplicate names. */
export function boneCatalog(vrm,source){
 const spring=new Map([...vrm.springBoneManager?.joints||[]].map(joint=>[joint.bone,joint]));
 const humanoid=new Map(Object.keys(vrm.humanoid?.normalizedHumanBones||{}).map(name=>[vrm.humanoid.getRawBoneNode(name),name]));
 const entries=[];
 function visit(node,path){
  if(node.isBone||spring.has(node)||humanoid.has(node))entries.push({id:path,node,name:node.name||'(unnamed)',human:humanoid.get(node)||'',joint:spring.get(node),restPosition:node.position.clone(),restQuaternion:node.quaternion.clone()});
  node.children.forEach((child,i)=>visit(child,path+'/'+i));
 }
 visit(vrm.scene,'root');
 const signature=JSON.stringify(entries.map(e=>[e.id,e.name,e.human,e.restPosition.toArray(),e.restQuaternion.toArray()]));
 let hash=2166136261;for(let i=0;i<signature.length;i++)hash=Math.imul(hash^signature.charCodeAt(i),16777619);
 return {key:source+'#'+(hash>>>0).toString(16),source,entries};
}
export function publishCatalog(catalog){
 const data={key:catalog.key,source:catalog.source,bones:catalog.entries.map(e=>({id:e.id,name:e.name,human:e.human,spring:!!e.joint,settings:e.joint?{...e.joint.settings,gravityDir:e.joint.settings.gravityDir.toArray()}:null}))};
 try{localStorage.setItem(CATALOG_KEY,JSON.stringify(data));window.dispatchEvent(new Event('avatar-catalog'));}catch{}
}
const axis=new Vector3(),error=new Vector3(),step=new Vector3(),rotation=new Quaternion();
function rotationVector(q,out){
 const sign=q.w<0?-1:1,w=Math.max(-1,Math.min(1,q.w*sign)),angle=2*Math.acos(w),s=Math.sqrt(1-w*w);
 return s<1e-7?out.set(0,0,0):out.set(q.x*sign,q.y*sign,q.z*sign).multiplyScalar(angle/s);
}
function vectorRotation(v,out){const length=v.length();return length<1e-8?out.identity():out.setFromAxisAngle(axis.copy(v).multiplyScalar(1/length),length);}

export class AvatarSecondaryMotion{
 constructor(vrm,catalog){
  this.vrm=vrm;this.entries=catalog.entries;this.states=new Map();this.applied=[];this.springDefaults=new Map();
  for(const e of this.entries)if(e.joint)this.springDefaults.set(e.id,{...e.joint.settings,gravityDir:e.joint.settings.gravityDir.clone()});
  this.normalizedRest=new Map(Object.keys(vrm.humanoid?.normalizedHumanBones||{}).map(name=>{const node=vrm.humanoid.getNormalizedBoneNode(name);return [node,{position:node.position.clone(),quaternion:node.quaternion.clone()}];}));
 }
 restore(){for(const {node,position,quaternion} of this.applied){node.position.copy(position);node.quaternion.copy(quaternion);}this.applied=[];}
 preview(){
  for(const entry of this.entries){entry.node.position.copy(entry.restPosition);entry.node.quaternion.copy(entry.restQuaternion);}
  for(const [node,rest] of this.normalizedRest){node.position.copy(rest.position);node.quaternion.copy(rest.quaternion);}
 }
 calibrate(profile){
  for(const entry of this.entries){
   const rule=profile.bones[entry.id];
   if(rule?.offset.some(value=>value!==0)){this.applied.push({node:entry.node,position:entry.node.position.clone(),quaternion:entry.node.quaternion.clone()});entry.node.position.add(new Vector3(...rule.offset));}
   if(entry.joint){const settings=entry.joint.settings,defaults=this.springDefaults.get(entry.id);
    for(const key of ['stiffness','dragForce','gravityPower','hitRadius'])settings[key]=rule?.springOverride?rule[key]:defaults[key];
    settings.gravityDir.copy(rule?.springOverride?new Vector3(...rule.gravityDir).normalize():defaults.gravityDir);
   }
  }
 }
 update(profile,dt){
  const active=new Set();
  for(const entry of this.entries){
   const rule=profile.bones[entry.id];if(!rule?.enabled||profile.tPose)continue;
   active.add(entry.id);const target=entry.node.quaternion.clone();
   let state=this.states.get(entry.id);if(!state){state={q:target.clone(),velocity:new Vector3()};this.states.set(entry.id,state);}
   // Fixed substeps keep high-frequency springs stable during slow frames.
   const duration=Math.max(0,Math.min(.1,dt)),count=Math.max(1,Math.ceil(duration*240)),h=duration/count,omega=2*Math.PI*rule.frequency;
   for(let i=0;i<count;i++){
    rotation.copy(state.q).invert().multiply(target);rotationVector(rotation,error);
    state.velocity.addScaledVector(error,omega*omega*h).multiplyScalar(Math.exp(-2*rule.damping*omega*h));
    step.copy(state.velocity).multiplyScalar(h);vectorRotation(step,rotation);state.q.multiply(rotation).normalize();
   }
   rotation.copy(target).invert().multiply(state.q);rotationVector(rotation,error);
   const limit=rule.maxAngle*Math.PI/180;if(error.length()>limit){error.setLength(limit);vectorRotation(error,rotation);state.q.copy(target).multiply(rotation);state.velocity.set(0,0,0);}
   if(!this.applied.some(item=>item.node===entry.node))this.applied.push({node:entry.node,position:entry.node.position.clone(),quaternion:target});
   entry.node.quaternion.copy(target).slerp(state.q,rule.amount);
  }
  for(const key of this.states.keys())if(!active.has(key))this.states.delete(key);
 }
}

export class AvatarSkeletonDebug{
 constructor(vrm,catalog,scene){
  this.helper=new SkeletonHelper(vrm.scene);this.helper.material.depthTest=false;this.helper.material.transparent=true;this.helper.material.opacity=.8;this.helper.visible=false;scene.add(this.helper);
  this.helper.frustumCulled=false;this.selected=new Set();
  this.edges=this.helper.bones.filter(b=>b.parent?.isBone).map(b=>catalog.entries.find(e=>e.node===b)?.id);
  const original=this.helper.updateMatrixWorld.bind(this.helper);
   this.helper.updateMatrixWorld=force=>{if(!this.helper.visible)return;original(force);const positions=this.helper.geometry.attributes.position;
   this.edges.forEach((id,i)=>{if(this.selected.size&&!this.selected.has(id)){positions.setXYZ(i*2,0,0,0);positions.setXYZ(i*2+1,0,0,0);}});positions.needsUpdate=true;
  };
 }
 update(profile){
  this.helper.visible=profile.debug;if(!profile.debug)return;
  this.selected=new Set(profile.debugBones);this.helper.updateMatrixWorld(true);
 }
}
