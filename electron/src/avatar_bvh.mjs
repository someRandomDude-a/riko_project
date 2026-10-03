import {Box3,Vector3} from 'three';
import {boneCatalog} from './avatar_secondary.mjs';

const SEGMENTS=[
 ['hips','spine',.09],['spine','chest',.09],['chest','upperChest',.09],['upperChest','neck',.08],['neck','head',.045],['head',null,.095],
 ...['left','right'].flatMap(side=>[
  [side+'Shoulder',side+'UpperArm',.045],[side+'UpperArm',side+'LowerArm',.035],
  [side+'LowerArm',side+'Hand',.03],[side+'Hand',null,.035],
  [side+'UpperLeg',side+'LowerLeg',.045],[side+'LowerLeg',side+'Foot',.035],
  [side+'Foot',side+'Toes',.045]])
];
export const HIT_BONES=SEGMENTS.map(([bone])=>bone);
export function boneRegion(bone){return ['head','neck'].includes(bone)?'head':bone.includes('Arm')||bone.includes('Hand')||bone.includes('Shoulder')?'arms':bone.includes('Leg')||bone.includes('Foot')||bone.includes('Toes')?'legs':'torso';}

export class AvatarBoneBVH{
 constructor(volumes){
  this.volumes=volumes;
  this.segmentPoint=new Vector3();this.rayPoint=new Vector3();
  this.refitVolumes();
  this.root=this.build([...volumes]);
 }
   static fromVRM(vrm,height,catalog=null){
   const bone=name=>name?(vrm.humanoid?.getRawBoneNode?.(name)||vrm.humanoid?.getNormalizedBoneNode?.(name)):null;
  const volumes=SEGMENTS.flatMap(([start,end,width])=>{
   const first=bone(start);if(!first)return [];
   let last=bone(end);
   // Optional chest bones must not leave a gap between spine and neck.
   if(!last&&['spine','chest','upperChest'].includes(start))last=bone('neck')||bone('head');
   return [{bone:start,region:boneRegion(start),first,last:last||first,radius:Math.max(.001,height*width),a:new Vector3(),b:new Vector3(),bounds:new Box3()}];
  });
    const entries=catalog?.entries||(vrm.scene?boneCatalog(vrm,'').entries:[]);
    for(const entry of entries)if(entry.joint){
     // Humanoid springs keep their existing interaction IDs and body coverage.
     // Their spring capsule is additional, not a replacement for the body volume.
     const child=entry.joint.child||entry.node.children.find(n=>n.isBone);
     const tail=!child?entry.joint.initialLocalChildPosition?.clone():null;
     volumes.push({bone:entry.human||entry.id,region:entry.human?boneRegion(entry.human):'spring',entry,spring:true,first:entry.node,last:child||entry.node,localTail:tail,radius:.025,a:new Vector3(),b:new Vector3(),bounds:new Box3()});
    }
   return new AvatarBoneBVH(volumes);
 }
  refitVolumes(springRadius){
   for(const volume of this.volumes){
    if(volume.spring&&Number.isFinite(springRadius))volume.radius=Math.max(.005,Math.min(.2,springRadius));
    volume.a.setFromMatrixPosition(volume.first.matrixWorld);
    if(volume.localTail)volume.b.copy(volume.localTail).applyMatrix4(volume.first.matrixWorld);
    else volume.b.setFromMatrixPosition(volume.last.matrixWorld);
   volume.bounds.makeEmpty().expandByPoint(volume.a).expandByPoint(volume.b).expandByScalar(volume.radius);
  }
 }
 build(volumes){
  if(!volumes.length)return null;
  const bounds=new Box3();for(const volume of volumes)bounds.union(volume.bounds);
  if(volumes.length<=2)return {bounds,volumes};
  const size=bounds.getSize(new Vector3()),axis=size.x>=size.y&&size.x>=size.z?'x':size.y>=size.z?'y':'z';
  volumes.sort((a,b)=>(a.a[axis]+a.b[axis])-(b.a[axis]+b.b[axis]));
  const middle=Math.floor(volumes.length/2);
  return {bounds,left:this.build(volumes.slice(0,middle)),right:this.build(volumes.slice(middle))};
 }
  refit(springRadius){
   this.refitVolumes(springRadius);
  const update=node=>{
   if(!node)return;
   node.bounds.makeEmpty();
   if(node.volumes){for(const volume of node.volumes)node.bounds.union(volume.bounds);}
   else{update(node.left);update(node.right);node.bounds.union(node.left.bounds).union(node.right.bounds);}
  };
  update(this.root);
 }
 hit(ray){
  let best=null,distance=Infinity;
  const visit=node=>{
   if(!node||!ray.intersectsBox(node.bounds))return;
   if(node.volumes){for(const volume of node.volumes){
    const squared=ray.distanceSqToSegment(volume.a,volume.b,this.rayPoint,this.segmentPoint);
    if(squared>volume.radius**2)continue;
    const depth=Math.max(0,ray.origin.distanceTo(this.rayPoint)-Math.sqrt(volume.radius**2-squared));
    if(depth<distance){distance=depth;best={bone:volume.bone,region:volume.region};}
   }}else{visit(node.left);visit(node.right);}
  };
  visit(this.root);return best;
 }
 intersects(ray){return this.hit(ray)!==null;}
}
