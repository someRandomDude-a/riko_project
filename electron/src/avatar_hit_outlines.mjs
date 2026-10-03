import {Group,Mesh,CylinderGeometry,SphereGeometry,MeshBasicMaterial,Vector3} from 'three';
export class AvatarHitOutlines{
 constructor(bvh,scene){
  this.group=new Group();scene.add(this.group);this.group.visible=false;
  const cylinder=new CylinderGeometry(1,1,1,8),sphere=new SphereGeometry(1,8,6);
  this.normal=new MeshBasicMaterial({color:0x55ddff,wireframe:true,transparent:true,opacity:.45,depthTest:false,depthWrite:false});
  this.selected=this.normal.clone();this.selected.color.setHex(0xffdd55);
  this.up=new Vector3(0,1,0);this.direction=new Vector3();
  this.parts=bvh.volumes.map(volume=>{
   const body=new Mesh(cylinder,this.normal),first=new Mesh(sphere,this.normal),last=new Mesh(sphere,this.normal);
   this.group.add(body,first,last);return {volume,body,first,last};
  });
 }
 update(preferences,target){
  this.group.visible=preferences.avatarHitOutlines===true;if(!this.group.visible)return;
  const filter=preferences.avatarHitBones||[];
  for(const {volume,body,first,last} of this.parts){
   const visible=!filter.length||filter.includes(volume.bone),material=target?.bone===volume.bone?this.selected:this.normal;
   for(const mesh of [body,first,last]){mesh.visible=visible;mesh.material=material;}
   if(!visible)continue;
   first.position.copy(volume.a);last.position.copy(volume.b);first.scale.setScalar(volume.radius);last.scale.setScalar(volume.radius);
   this.direction.subVectors(volume.b,volume.a);const length=this.direction.length();
   body.visible=length>1e-6;body.position.copy(volume.a).add(volume.b).multiplyScalar(.5);body.scale.set(volume.radius,length,volume.radius);
   if(length>1e-6)body.quaternion.setFromUnitVectors(this.up,this.direction.divideScalar(length));
  }
 }
}
