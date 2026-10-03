import {Vector3,Quaternion,Raycaster,Vector2,Plane} from 'three';
const clamp=(value,limit)=>Math.max(-limit,Math.min(limit,value));

/** Binocular convergence uses rotations only: anatomical eye positions never change. */
export class AvatarEyeGaze{
 constructor(vrm){this.vrm=vrm;this.lookAt=vrm.lookAt;this.originalAuto=this.lookAt?.autoUpdate;this.controlled=false;this.saved=[];this.angles={left:{yaw:0,pitch:0},right:{yaw:0,pitch:0}};this.ray=new Raycaster();this.plane=new Plane();this.target=new Vector3();}
 restore(){for(const [node,q] of this.saved)node.quaternion.copy(q);this.saved=[];}
 update(settings,dt,point,rect,camera,sphere,active=true){
  const look=this.lookAt;if(!look)return;
  if(!settings.enabled){if(this.controlled){look.autoUpdate=this.originalAuto;look.reset();this.controlled=false;}look.update(dt);return;}
  this.controlled=true;look.autoUpdate=false;
  const head=look.getLookAtWorldPosition(new Vector3());let tracking=false;
  if(active&&point&&rect.width&&rect.height){
   if(sphere?.active&&sphere.initialized)this.target.copy(sphere.position);
   else {this.ray.setFromCamera(new Vector2((point.x-rect.left)/rect.width*2-1,1-(point.y-rect.top)/rect.height*2),camera);this.plane.set(new Vector3(0,0,1),-(head.z+settings.minDistance));if(!this.ray.ray.intersectPlane(this.plane,this.target))return;}
   // Keep the synthetic target in front of the face, including during pickup.
   const forward=new Vector3(0,0,1).applyQuaternion(look.getFaceFrontQuaternion(new Quaternion())).applyQuaternion(look.getLookAtWorldQuaternion(new Quaternion()));
   const depth=this.target.clone().sub(head).dot(forward);if(depth<settings.minDistance)this.target.addScaledVector(forward,settings.minDistance-depth);
   tracking=true;
  }
  const desired={left:{yaw:0,pitch:0},right:{yaw:0,pitch:0}};
  if(tracking){look.lookAt(this.target);const center={yaw:clamp(look.yaw,settings.maxYaw),pitch:clamp(look.pitch,settings.maxPitch)};
   for(const side of ['left','right']){
    const eye=this.vrm.humanoid.getRawBoneNode(side+'Eye');
    if(settings.convergence&&eye){const shifted=this.target.clone().add(head).sub(eye.getWorldPosition(new Vector3()));look.lookAt(shifted);desired[side]={yaw:clamp(center.yaw+clamp(look.yaw-center.yaw,settings.maxConvergence)*settings.strength,settings.maxYaw),pitch:clamp(look.pitch,settings.maxPitch)};}
    else desired[side]={...center};
   }
  }
  const weight=1-Math.exp(-settings.smoothing*Math.max(0,Math.min(.1,dt)));
  for(const side of ['left','right'])for(const key of ['yaw','pitch'])this.angles[side][key]+=(desired[side][key]-this.angles[side][key])*weight;
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
 dispose(){this.restore();if(this.lookAt){this.lookAt.autoUpdate=this.originalAuto;this.lookAt.reset();}}
}
