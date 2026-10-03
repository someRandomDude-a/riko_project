import {Vector3,Quaternion,Raycaster,Vector2,Plane,Mesh,SphereGeometry,MeshBasicMaterial} from 'three';

export function endpointPosition(endpoint,out=new Vector3()){
 return endpoint.localTail?out.copy(endpoint.localTail).applyMatrix4(endpoint.node.matrixWorld):endpoint.node.getWorldPosition(out);
}
function endpoints(entry){
 const child=entry.joint?.child||entry.node.children.find(n=>n.isBone),localTail=!child?entry.joint?.initialLocalChildPosition?.clone():null;
 return [{node:entry.node,kind:'head'},child?{node:child,kind:'tail'}:localTail?{node:entry.node,localTail,kind:'tail'}:{node:entry.node,kind:'head'}];
}
export function nearestEndpoint(entry,point){
 const [first,last]=endpoints(entry);
 return endpointPosition(first).distanceToSquared(point)<=endpointPosition(last).distanceToSquared(point)?first:last;
}
/** A fixed rig attachment, with torque from one endpoint per contact/drag. */
export class AvatarMouseSphere{
 constructor(catalog,scene){this.catalog=catalog;this.position=new Vector3();this.target=new Vector3();this.velocity=new Vector3();this.active=false;this.initialized=false;this.drag=null;this.states=new Map();this.saved=[];this.clickAge=Infinity;this.ray=new Raycaster();this.plane=new Plane();
  this.debug=new Mesh(new SphereGeometry(1,16,12),new MeshBasicMaterial({color:0x77ddff,wireframe:true,transparent:true,opacity:.3,depthWrite:false}));this.debug.visible=false;scene.add(this.debug);
 }
  aim(point,rect,camera,z){if(!rect.width||!rect.height)return;this.active=!!this.drag||point.x>=rect.left&&point.x<=rect.left+rect.width&&point.y>=rect.top&&point.y<=rect.top+rect.height;if(!this.active)return;this.ray.setFromCamera(new Vector2((point.x-rect.left)/rect.width*2-1,1-(point.y-rect.top)/rect.height*2),camera);if(this.drag){const projected=this.ray.ray.intersectPlane(this.drag.plane,new Vector3());if(projected)this.drag.target.copy(projected).add(this.drag.offset);}this.plane.set(new Vector3(0,0,1),-z);if(this.ray.ray.intersectPlane(this.plane,this.target)&&!this.initialized){this.position.copy(this.target);this.initialized=true;}}
 click(){this.clickAge=0;}
  prepare(entry,point,rect,camera){
   if(!rect.width||!rect.height)return null;
   this.ray.setFromCamera(new Vector2((point.x-rect.left)/rect.width*2-1,1-(point.y-rect.top)/rect.height*2),camera);
   const [first,last]=endpoints(entry),segmentPoint=new Vector3();
   this.ray.ray.distanceSqToSegment(endpointPosition(first),endpointPosition(last),new Vector3(),segmentPoint);
   const endpoint=nearestEndpoint(entry,segmentPoint),position=endpointPosition(endpoint),plane=new Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new Vector3()),position);
   const projected=this.ray.ray.intersectPlane(plane,new Vector3());if(!projected)return null;
   return {entry,endpoint,plane,offset:position.clone().sub(projected),target:position.clone(),position:position.clone()};
  }
  start(entry,prepared=null){
   const endpoint=prepared?.endpoint||nearestEndpoint(entry,this.position),position=endpointPosition(endpoint);
   this.drag=prepared||{entry,endpoint,plane:new Plane(new Vector3(0,0,1),-position.z),offset:new Vector3(),target:this.position.clone(),position:position.clone(),legacy:true};return endpoint;
  }
 release(){this.drag=null;}
 restore(){for(const [node,q] of this.saved)node.quaternion.copy(q);this.saved=[];}
 update(profile,dt){
  const s=profile.mouseSphere;this.clickAge+=dt;this.debug.visible=s.enabled&&s.debug&&this.active;
  const desired=this.target.clone();if(this.clickAge<s.bounceSeconds)desired.z-=Math.sin(Math.PI*this.clickAge/s.bounceSeconds)*s.bounce;
  const n=Math.max(1,Math.ceil(Math.min(.1,dt)*240)),h=Math.min(.1,dt)/n,w=2*Math.PI*s.frequency;
  for(let i=0;i<n;i++){this.velocity.addScaledVector(desired.clone().sub(this.position),w*w*h).multiplyScalar(Math.exp(-2*s.damping*w*h));this.position.addScaledVector(this.velocity,h);}
  this.debug.position.copy(this.position);this.debug.scale.setScalar(s.radius);
  const torque=new Map(),eligible=this.catalog.entries.filter(e=>e.joint||profile.bones[e.id]?.enabled);
  if(s.enabled&&this.active){
    for(const entry of eligible){const endpoint=nearestEndpoint(entry,this.position),p=endpointPosition(endpoint),distance=p.distanceTo(this.position);if(distance>=s.radius)continue;
     const force=this.velocity.clone().multiplyScalar(s.strength*(1-distance/s.radius));this.addTorque(torque,entry,endpoint,force,eligible);
    }
   }
   // Local grabbing is independent of ambient sphere pushing. Angular targets
   // normalize the lever length: short hair joints are not effectively immovable.
   const targets=new Map(),dragGain=this.drag? s.dragSpring/10:0;
   if(this.drag&&dragGain>0){
    const drag=this.drag;if(drag.legacy)drag.target.copy(this.position);
    drag.position.lerp(drag.target,1-Math.exp(-2*Math.PI*s.frequency*Math.max(0,dt)));
    const p=endpointPosition(drag.endpoint);let owner=this.owner(drag.entry,drag.endpoint,eligible);
    while(owner){
     const origin=owner.node.getWorldPosition(new Vector3()),from=p.clone().sub(origin),to=drag.position.clone().sub(origin);
     if(from.lengthSq()>1e-10&&to.lengthSq()>1e-10){
      const q=new Quaternion().setFromUnitVectors(from.normalize(),to.normalize()),angle=2*Math.atan2(Math.hypot(q.x,q.y,q.z),q.w);
      const value=new Vector3(q.x,q.y,q.z);if(value.lengthSq()>1e-12)value.setLength(angle);
      value.applyQuaternion((owner.node.parent?.getWorldQuaternion(new Quaternion())||new Quaternion()).invert());targets.set(owner.id,value);
     }
     owner=eligible.find(e=>e.node===owner.node.parent);
    }
   }
   for(const entry of eligible){let state=this.states.get(entry.id);if(!state){state={angle:new Vector3(),velocity:new Vector3()};this.states.set(entry.id,state);}const force=torque.get(entry.id)||new Vector3(),rule=profile.bones[entry.id],frequency=rule?.frequency||3,damping=rule?.damping??.7,omega=2*Math.PI*frequency;
    const target=targets.get(entry.id),gain=target?dragGain:0;
    for(let i=0;i<n;i++){state.velocity.addScaledVector(force,h).addScaledVector(state.angle,-omega*omega*(1+gain)*h);if(target)state.velocity.addScaledVector(target,omega*omega*gain*h);state.velocity.multiplyScalar(Math.exp(-2*damping*omega*Math.sqrt(1+gain)*h));state.angle.addScaledVector(state.velocity,h);const cap=(rule?.maxAngle??s.maxAngle)*Math.PI/180;if(state.angle.length()>cap){state.angle.setLength(cap);state.velocity.multiplyScalar(.5);}}
    const length=state.angle.length();if(length>1e-7){this.saved.push([entry.node,entry.node.quaternion.clone()]);entry.node.quaternion.premultiply(new Quaternion().setFromAxisAngle(state.angle.clone().normalize(),length));entry.node.updateMatrixWorld(true);}
   }
  }
  owner(entry,endpoint,eligible){return endpoint.kind==='head'?eligible.find(e=>e.node===entry.node.parent):entry;}
  addTorque(map,entry,endpoint,force,eligible,chain=false){
   if(!endpoint.kind)endpoint={node:endpoint,kind:endpoint===entry.node?'head':'tail'};
  // A head belongs to its upstream spring. Never translate the rig or pull both ends.
   let owner=this.owner(entry,endpoint,eligible);
   while(owner){const lever=endpointPosition(endpoint).sub(owner.node.getWorldPosition(new Vector3())),value=new Vector3().crossVectors(lever,force);value.applyQuaternion((owner.node.parent?.getWorldQuaternion(new Quaternion())||new Quaternion()).invert());value.clampLength(0,30);if(!map.has(owner.id))map.set(owner.id,new Vector3());map.get(owner.id).add(value);if(!chain)break;owner=eligible.find(e=>e.node===owner.node.parent);}
 }
}
