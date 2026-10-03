import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMUtils} from '@pixiv/three-vrm';
import {VRMAnimationLoaderPlugin,createVRMAnimationClip} from '@pixiv/three-vrm-animation';
import {API} from './api.mjs';

/** Import adapters are isolated from policy and pose interpolation. */
export class MotionAssetSampler {
  constructor(vrm,{load}={}) {
    this.vrm=vrm;this.cache=new Map();this.closed=false;
    const loader=new GLTFLoader();loader.register(parser=>new VRMAnimationLoaderPlugin(parser));
    this.load=load|| (asset=>loader.loadAsync(asset.path?API+'/api/avatar/animation?path='+encodeURIComponent(asset.path):API+'/api/animation/assets/'+encodeURIComponent(asset.id)+'/file'));
  }
  sample(asset,age,actionID,report) {
    if(asset.kind==='pose'){
      const mask=new Set(asset.mask?.length?asset.mask:Object.keys(asset.pose.bones));
      reportOnce(this,asset.id,actionID,report);
      return {expressions:new Map(Object.entries(asset.pose.expressions||{})),bones:new Map(Object.entries(asset.pose.bones).filter(([bone])=>mask.has(bone)).map(([bone,euler])=>[bone,new THREE.Quaternion().setFromEuler(new THREE.Euler(...euler,'XYZ'))]))};
    }
    let entry=this.cache.get(asset.id);
    const signature=JSON.stringify([asset.mask,asset.speed,asset.loop]);
    if(entry&&entry.signature!==signature){entry.mixer?.stopAllAction();entry.mixer?.uncacheRoot(this.vrm.scene);this.cache.delete(asset.id);entry=null;}
    if(!entry){
      entry={asset,signature,pending:true,reported:new Set(),report,actionID};this.cache.set(asset.id,entry);
      Promise.resolve().then(()=>this.load(asset)).then(gltf=>{
        try{
          if(this.closed||this.cache.get(asset.id)!==entry)return;
          const animation=gltf.userData?.vrmAnimations?.[0];if(!animation)throw new Error('Asset has no VRM animation');
          const mask=new Set(asset.mask?.length?asset.mask:[...animation.humanoidTracks.rotation.keys()]);
          const missing=[...mask].filter(bone=>!this.vrm.humanoid.getNormalizedBoneNode(bone));
          if(missing.length)throw new Error('Missing avatar bones: '+missing.join(', '));
          const clip=createVRMAnimationClip(animation,this.vrm);
          const allowed=new Set([...mask].map(bone=>this.vrm.humanoid.getNormalizedBoneNode(bone).name+'.quaternion'));
          entry.expressions=[...animation.expressionTracks.preset.keys(),...animation.expressionTracks.custom.keys()].filter(name=>this.vrm.expressionManager?.getExpression(name));
          for(const name of entry.expressions)allowed.add(this.vrm.expressionManager.getExpressionTrackName(name));
          // Desktop controller owns translation; root/hips position tracks cannot move the surface twice.
          clip.tracks=clip.tracks.filter(track=>allowed.has(track.name));
          if(!clip.tracks.length)throw new Error('Animation has no compatible masked skeletal tracks');
          entry.bones=[...mask];entry.mixer=new THREE.AnimationMixer(this.vrm.scene);
          entry.playback=entry.mixer.clipAction(clip);entry.duration=Math.max(.001,clip.duration);entry.pending=false;
        }finally{VRMUtils.deepDispose(gltf.scene);}
      }).catch(error=>{if(!this.closed&&this.cache.get(asset.id)===entry){entry.pending=false;entry.error=error.message;}});
      return null;
    }
    if(entry.pending)return null;
    if(entry.error){if(!entry.reported.has(actionID)){entry.reported.add(actionID);report('error',entry.error);}return null;}
    const first=!entry.reported.has(actionID);
    if(first){entry.reported.add(actionID);report('started');}
    const seconds=age*(asset.speed||1);
    const time=asset.loop?seconds%entry.duration:Math.min(seconds,entry.duration);
    const saved=new Map(entry.bones.map(bone=>[bone,this.vrm.humanoid.getNormalizedBoneNode(bone).quaternion.clone()]));
    const savedExpressions=new Map(entry.expressions.map(name=>[name,this.vrm.expressionManager.getValue(name)]));
    entry.playback.setLoop(THREE.LoopOnce,1);entry.playback.clampWhenFinished=true;entry.playback.reset().play();entry.mixer.setTime(time);
    const bones=new Map(entry.bones.map(bone=>[bone,this.vrm.humanoid.getNormalizedBoneNode(bone).quaternion.clone()]));
    const expressions=new Map(entry.expressions.map(name=>[name,this.vrm.expressionManager.getValue(name)]));
    entry.mixer.stopAllAction();for(const [bone,value]of saved)this.vrm.humanoid.getNormalizedBoneNode(bone).quaternion.copy(value);
    for(const [name,value]of savedExpressions)this.vrm.expressionManager.setValue(name,value);
    if(!asset.loop&&seconds>=entry.duration&&!entry.completed?.has(actionID)){
      (entry.completed||=new Set()).add(actionID);report('completed');
    }
    return {bones,expressions};
  }
  retain(ids){
    for(const [id,entry]of this.cache)if(!ids.has(id)){entry.mixer?.stopAllAction();entry.mixer?.uncacheRoot(this.vrm.scene);this.cache.delete(id);}
    if(this.poseReports)for(const id of this.poseReports.keys())if(!ids.has(id))this.poseReports.delete(id);
  }
  dispose(){this.closed=true;this.retain(new Set());}
}

function reportOnce(sampler,assetID,actionID,report){sampler.poseReports||=new Map();if(sampler.poseReports.get(assetID)!==actionID){sampler.poseReports.set(assetID,actionID);report('started');}}
