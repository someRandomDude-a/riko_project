import {Vector3,Quaternion,Box3} from 'three';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
/** Anchored pendulum and bounded articulated angular dynamics. No mesh collisions. */
export class AvatarPickup{
 constructor(vrm,catalog,onTrigger=()=>{}){this.vrm=vrm;this.catalog=catalog;this.onTrigger=onTrigger;this.active=false;this.saved=null;this.joints=new Map();this.angle=0;this.velocity=0;this.force=0;}
 start(bone,point,camera){
  this.restore();this.bone=this.vrm.humanoid.getRawBoneNode(bone);if(!this.bone)return;
  this.vrm.scene.updateMatrixWorld(true);this.anchor=this.bone.getWorldPosition(new Vector3());
  this.camera=camera;this.cameraRest=camera?{position:camera.position.clone(),quaternion:camera.quaternion.clone()}:null;this.anchorNDC=camera?this.anchor.clone().project(camera):null;
  const bounds=new Box3().setFromObject(this.vrm.scene);if(bounds.isEmpty())for(const e of this.catalog.entries)bounds.expandByPoint(e.node.getWorldPosition(new Vector3()));
  const center=bounds.getCenter(new Vector3()),offset=center.sub(this.anchor);
  this.massOffset=offset.clone();this.origin={...point};this.last={...point};this.active=true;this.held=true;this.ragdoll=false;this.age=0;this.angle=0;this.velocity=offset.y>0?.03:0;this.force=0;this.joints.clear();
 }
 move(point,dt,settings){if(!this.active||!this.held)return;
  const speed=Math.hypot(point.x-this.last.x,point.y-this.last.y)/Math.max(.008,dt);
  this.force=clamp((point.x-this.last.x)/Math.max(.008,dt)/500,-5,5)*settings.inertia;this.last={...point};
  if(settings.ragdollEnabled&&!this.ragdoll&&(Math.hypot(point.x-this.origin.x,point.y-this.origin.y)>settings.distanceThreshold||speed>settings.speedThreshold)){this.ragdoll=true;this.onTrigger('ragdoll');}
 }
 release(){if(this.active&&this.held){this.held=false;this.age=0;if(this.ragdoll)this.onTrigger('recover');}}
 restore(){if(!this.saved)return;const root=this.vrm.scene;root.position.copy(this.saved.position);root.quaternion.copy(this.saved.quaternion);for(const [node,q] of this.saved.bones)node.quaternion.copy(q);this.saved=null;}
 step(settings,dt){
  if(!this.active)return;if(!settings.enabled){this.active=false;return;}
  const root=this.vrm.scene;this.saved={position:root.position.clone(),quaternion:root.quaternion.clone(),bones:[]};
  if(!this.held)this.age+=dt;const blend=this.held?1:Math.max(0,1-this.age/settings.recoverySeconds);
   const limit=settings.maxSwing*Math.PI/180;
  const n=Math.max(1,Math.ceil(Math.min(.1,dt)*120)),h=Math.min(.1,dt)/n;
  for(let i=0;i<n;i++){const r=this.massOffset,length=Math.max(.1,r.length()),torque=this.held?-settings.gravity*(r.x*Math.cos(this.angle)-r.y*Math.sin(this.angle))/(length*length):-settings.bodySpring*this.angle;this.velocity+=(torque-settings.damping*this.velocity-this.force)*h;this.angle=clamp(this.angle+this.velocity*h,-limit,limit);}
  this.force*=Math.exp(-dt*6);root.quaternion.premultiply(new Quaternion().setFromAxisAngle(new Vector3(0,0,1),this.angle*blend));root.updateMatrixWorld(true);
  for(const entry of this.catalog.entries){
   const child=entry.node.children.find(node=>node.isBone);if(!child)continue;
   let state=this.joints.get(entry.id);if(!state){state={angle:new Vector3(),velocity:new Vector3()};this.joints.set(entry.id,state);}
   const direction=child.getWorldPosition(new Vector3()).sub(entry.node.getWorldPosition(new Vector3())).normalize();
   const torque=new Vector3().crossVectors(direction,new Vector3(-this.force,-settings.gravity,0));
   const parent=entry.node.parent?.getWorldQuaternion(new Quaternion())||new Quaternion();torque.applyQuaternion(parent.invert());
   const stiffness=this.ragdoll&&this.held ? .4 : settings.bodySpring;
   for(let i=0;i<n;i++){state.velocity.addScaledVector(torque,h).addScaledVector(state.angle,-stiffness*h).multiplyScalar(Math.exp(-settings.damping*h));state.angle.addScaledVector(state.velocity,h);const cap=settings.jointLimit*Math.PI/180;if(state.angle.length()>cap){state.angle.setLength(cap);state.velocity.multiplyScalar(.5);}}
   this.saved.bones.push([entry.node,entry.node.quaternion.clone()]);const angle=state.angle.length();if(angle>1e-7)entry.node.quaternion.multiply(new Quaternion().setFromAxisAngle(state.angle.clone().normalize(),angle*blend));
  }
  root.updateMatrixWorld(true);const current=this.bone.getWorldPosition(new Vector3());root.position.add(this.anchor.clone().sub(current).multiplyScalar(blend));root.updateMatrixWorld(true);
  if(this.camera){
   // Reframe the swinging skeleton without moving the held bone's screen anchor.
   const bounds=new Box3();for(const e of this.catalog.entries)bounds.expandByPoint(e.node.getWorldPosition(new Vector3()));bounds.expandByScalar(.2);
   const x=clamp(this.anchorNDC.x,-.95,.95),y=clamp(this.anchorNDC.y,-.95,.95),tan=Math.tan(this.camera.fov*Math.PI/360),tx=tan*this.camera.aspect;
   const desired=Math.max(this.cameraRest.position.z-this.anchor.z,(bounds.max.x-this.anchor.x)/(tx*(1-x)),(this.anchor.x-bounds.min.x)/(tx*(1+x)),(bounds.max.y-this.anchor.y)/(tan*(1-y)),(this.anchor.y-bounds.min.y)/(tan*(1+y)));
   const distance=Math.max(.1,desired),cx=this.anchor.x-x*distance*tx,cy=this.anchor.y-y*distance*tan;
   this.camera.position.set(cx,cy,this.anchor.z+distance);this.camera.lookAt(cx,cy,this.anchor.z);this.camera.updateMatrixWorld(true);
  }
  if(blend===0){this.restore();this.active=false;this.ragdoll=false;this.joints.clear();if(this.cameraRest){this.camera.position.copy(this.cameraRest.position);this.camera.quaternion.copy(this.cameraRest.quaternion);this.camera.updateMatrixWorld(true);}}
 }
}
