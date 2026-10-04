import * as THREE from 'three';
import {importGLSL} from './avatar_studio_settings.mjs';
const vertex=`varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}`;
const fragment=custom=>`uniform sampler2D image;uniform float time;uniform float exposure;uniform float saturation;uniform vec3 tint;varying vec2 vUv;
${custom||'vec4 avatarEffect(vec4 color,vec2 uv,float time){return color;}'}
void main(){vec4 source=texture2D(image,vUv);vec3 rgb=source.rgb*exposure*tint;float luminance=dot(rgb,vec3(.2126,.7152,.0722));vec4 result=avatarEffect(vec4(mix(vec3(luminance),rgb,saturation),source.a),vUv,time);gl_FragColor=vec4(clamp(result.rgb,0.0,1.0),source.a);
#include <colorspace_fragment>
}`;

/** Model-space lights plus alpha-preserving screen-space effects; VRM shaders stay intact. */
export class AvatarEffects{
  constructor(renderer,scene,report=()=>{},samples=4){
  this.renderer=renderer;this.scene=scene;this.report=report;this.lights=new THREE.Group();scene.add(this.lights);
  this.ambient=new THREE.HemisphereLight(0xffffff,0x443355,2);this.lights.add(this.ambient);
  this.lightKey='';this.shaderKey=null;this.failed=false;
  this.target=new THREE.WebGLRenderTarget(1,1,{depthBuffer:true});
  this.target.samples=Math.min(samples,renderer.capabilities?.maxSamples??samples);
  this.screen=new THREE.Scene();this.camera=new THREE.Camera();
  this.uniforms={image:{value:this.target.texture},time:{value:0},exposure:{value:1},saturation:{value:1},tint:{value:new THREE.Vector3(1,1,1)}};
  this.material=new THREE.ShaderMaterial({uniforms:this.uniforms,vertexShader:vertex,fragmentShader:fragment(''),depthTest:false,depthWrite:false});
  this.quad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.material);this.screen.add(this.quad);
  this.previousError=renderer.debug.onShaderError;
  renderer.debug.onShaderError=(gl,program,vs,fs)=>{
   this.failed=true;this.report('Shader compilation failed. Custom effect disabled for this session. '+(gl.getShaderInfoLog(fs)||'').slice(0,500));
   this.previousError?.(gl,program,vs,fs);
  };
 }
 update(profile,time){
   if(profile.lighting!==this.lightingSource){
    this.lightingSource=profile.lighting;
   const key=JSON.stringify(profile.lighting);
   if(key!==this.lightKey){
   this.lightKey=key;const lighting=profile.lighting;
   this.ambient.color.set(lighting.sky);this.ambient.groundColor.set(lighting.ground);this.ambient.intensity=lighting.ambient;
   for(const light of [...this.lights.children])if(light!==this.ambient){this.lights.remove(light);light.dispose?.();}
   for(const spec of lighting.lights){const light=spec.type==='directional'?new THREE.DirectionalLight(spec.color,spec.intensity):new THREE.PointLight(spec.color,spec.intensity);light.position.fromArray(spec.position);this.lights.add(light);}
   }
   }
  const effect=profile.effect,shader=effect.glslEnabled?effect.glsl:'';
  if(shader!==this.shaderKey){
   this.shaderKey=shader;this.failed=false;
   try{if(shader)importGLSL(shader);this.material.fragmentShader=fragment(shader);this.material.needsUpdate=true;this.report('');}
   catch(error){this.failed=true;this.report(error.message);}
  }
  this.uniforms.time.value=time;this.uniforms.exposure.value=effect.exposure;
  this.uniforms.saturation.value=effect.preset==='monochrome'?0:effect.saturation;
  this.uniforms.tint.value.set(...(effect.preset==='warm'?[1.1,1,.85]:effect.preset==='cool'?[.85,1,1.1]:[1,1,1]));
 }
 render(scene,camera,profile){
  const effect=profile.effect;
  if(this.failed||(effect.preset==='none'&&effect.exposure===1&&effect.saturation===1&&!effect.glslEnabled)){this.renderer.render(scene,camera);return;}
  const size=this.renderer.getDrawingBufferSize(new THREE.Vector2());
  if(this.target.width!==size.x||this.target.height!==size.y)this.target.setSize(size.x,size.y);
  this.renderer.setRenderTarget(this.target);this.renderer.render(scene,camera);this.renderer.setRenderTarget(null);
  this.renderer.render(this.screen,this.camera);
  if(this.failed)this.renderer.render(scene,camera);
 }
 dispose(){this.renderer.debug.onShaderError=this.previousError;this.target.dispose();this.quad.geometry.dispose();this.material.dispose();this.scene.remove(this.lights);for(const light of this.lights.children)light.dispose?.();}
}
