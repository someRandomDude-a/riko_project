import test from 'node:test';
import assert from 'node:assert/strict';
import {Bone,Group,Quaternion,Vector3} from 'three';
import {normalizeBone,normalizeProfile,normalizeStudio,importStudioPreset,importGLSL} from './avatar_studio_settings.mjs';
import {boneCatalog,AvatarSecondaryMotion,AvatarSkeletonDebug} from './avatar_secondary.mjs';
import {AvatarEffects} from './avatar_effects.mjs';

function fixture(){
 const scene=new Group(),hips=new Bone(),hair=new Bone(),extra=new Bone();hips.name='hips';hair.name=extra.name='hair';hair.position.y=1;extra.position.x=.2;scene.add(hips);hips.add(hair,extra);
 const normalized=new Bone(),joint={bone:hair,settings:{stiffness:2,dragForce:.6,gravityPower:.2,hitRadius:.03,gravityDir:new Vector3(0,-1,0)}};
 const vrm={scene,humanoid:{normalizedHumanBones:{hips:{}},getRawBoneNode:()=>hips,getNormalizedBoneNode:()=>normalized},springBoneManager:{joints:new Set([joint])}};
 const catalog=boneCatalog(vrm,'test.vrm');return {scene,hips,hair,extra,normalized,joint,vrm,catalog,secondary:new AvatarSecondaryMotion(vrm,catalog)};
}
test('catalog includes humanoid, spring and other bones with stable duplicate-name IDs',()=>{
 const {vrm,catalog}=fixture();assert.equal(catalog.entries.length,3);assert.equal(new Set(catalog.entries.map(e=>e.id)).size,3);
 assert.equal(catalog.key,boneCatalog(vrm,'test.vrm').key);assert.notEqual(catalog.key,boneCatalog(vrm,'other.vrm').key);
 vrm.scene.children[0].children[0].position.y=2;assert.notEqual(catalog.key,boneCatalog(vrm,'test.vrm').key);
});
test('studio normalization bounds bone physics, offsets, lights and shader imports',()=>{
 const p=normalizeProfile({bones:{a:{enabled:true,frequency:Infinity,offset:[100,NaN,-10],maxAngle:100}},lighting:{lights:Array(12).fill({intensity:999,position:[999,0,0]})},effect:{exposure:-1,glsl:'x'.repeat(20000)}});
 assert.equal(p.bones.a.frequency,3);assert.deepEqual(p.bones.a.offset,[.5,0,-.5]);assert.equal(p.bones.a.maxAngle,90);
 assert.equal(p.lighting.lights.length,8);assert.equal(p.lighting.lights[0].intensity,20);assert.equal(p.effect.exposure,0);assert.equal(p.effect.glsl,'');
 assert.deepEqual(normalizeStudio({avatarStudioProfiles:null}),{avatarStudioProfiles:{}});
 assert.equal(normalizeBone({}).enabled,false);
});
test('position calibration is non-cumulative and removing overrides restores imported springs',()=>{
 const {hair,joint,catalog,secondary}=fixture(),id=catalog.entries.find(e=>e.node===hair).id;
 const profile=normalizeProfile({bones:{[id]:{offset:[.1,0,0],springOverride:true,stiffness:4,gravityDir:[1,0,0]}}});
 for(let i=0;i<20;i++){secondary.restore();secondary.calibrate(profile);assert.equal(hair.position.x,.1);assert.equal(joint.settings.stiffness,4);}
 secondary.restore();secondary.calibrate(normalizeProfile({}));assert.equal(hair.position.x,0);assert.equal(joint.settings.stiffness,2);assert.deepEqual(joint.settings.gravityDir.toArray(),[0,-1,0]);
});
test('secondary motion follows changing animation, stays finite and caps angular lag',()=>{
 const {hips,catalog,secondary}=fixture(),id=catalog.entries[0].id,profile=normalizeProfile({bones:{[id]:{enabled:true,amount:1,frequency:20,damping:0,maxAngle:15}}});
 secondary.update(profile,1/60);
 const target=new Quaternion().setFromAxisAngle(new Vector3(0,1,0),1);
 for(let i=0;i<120;i++){secondary.restore();hips.quaternion.copy(target);secondary.update(profile,.1);assert.ok(hips.quaternion.toArray().every(Number.isFinite));assert.ok(hips.quaternion.angleTo(target)<=15*Math.PI/180+1e-7);}
 secondary.restore();assert.ok(hips.quaternion.angleTo(target)<1e-7);secondary.update(normalizeProfile({}),.1);assert.equal(secondary.states.size,0);
});
test('damped secondary motion lags a new target then converges; preview resets the complete skeleton',()=>{
 const {hips,hair,normalized,catalog,secondary}=fixture(),id=catalog.entries[0].id,profile=normalizeProfile({bones:{[id]:{enabled:true,amount:1,frequency:2,damping:.7,maxAngle:90}}});
 secondary.update(profile,1/60);const target=new Quaternion().setFromAxisAngle(new Vector3(1,0,0),.7);
 secondary.restore();hips.quaternion.copy(target);secondary.update(profile,1/60);assert.ok(hips.quaternion.angleTo(target)>.1);
 for(let i=0;i<240;i++){secondary.restore();hips.quaternion.copy(target);secondary.update(profile,1/60);}
 assert.ok(hips.quaternion.angleTo(target)<.001);secondary.restore();hair.position.x=1;normalized.quaternion.copy(target);secondary.preview();assert.equal(hair.position.x,0);assert.ok(normalized.quaternion.angleTo(new Quaternion())<1e-7);
});
test('skeleton debug filters survive subsequent renderer matrix updates',()=>{
 const {scene,vrm,catalog,hair}=fixture(),debug=new AvatarSkeletonDebug(vrm,catalog,scene),id=catalog.entries.find(e=>e.node===hair).id;
 scene.updateMatrixWorld(true);debug.update(normalizeProfile({debug:true,debugBones:[id]}));scene.updateMatrixWorld(true);
 const p=debug.helper.geometry.attributes.position;assert.notEqual(p.getY(0),0);assert.equal(p.getX(2),0);assert.equal(p.getY(2),0);
 debug.update(normalizeProfile({debug:true}));scene.updateMatrixWorld(true);assert.notEqual(p.getX(2),0);
 debug.update(normalizeProfile({}));assert.equal(debug.helper.visible,false);
});
test('JSON and GLSL imports enforce format and never silently enable GPU code',()=>{
 const code='vec4 avatarEffect(vec4 color, vec2 uv, float time){return vec4(color.rgb*.9,color.a);}';assert.equal(importGLSL(code),code);
 assert.throws(()=>importGLSL('void main(){}'),/Define/);assert.throws(()=>importGLSL(code+'\nvoid bad(){while(true){}}'),/loops/);
 assert.throws(()=>importGLSL('x'.repeat(16385)),/16 KiB/);assert.throws(()=>importStudioPreset('{}'),/version 1/);
 const preset=importStudioPreset(JSON.stringify({format:'riko-avatar-effects',version:1,lighting:{ambient:1},effect:{glsl:code,glslEnabled:true}}));assert.equal(preset.effect.glslEnabled,false);assert.equal(preset.effect.glsl,code);
});
test('lights update without duplication and post effects preserve source alpha',()=>{
 const renderer={debug:{},getDrawingBufferSize:v=>v.set(200,300),setRenderTarget:()=>{},render:()=>{}},scene=new Group(),effects=new AvatarEffects(renderer,scene);
 const profile=normalizeProfile({lighting:{ambient:.5,lights:[{type:'point',intensity:3}]},effect:{preset:'warm'}});
 effects.update(profile,10);effects.update(profile,11);assert.equal(effects.lights.children.length,2);assert.equal(effects.ambient.intensity,.5);
 assert.match(effects.material.fragmentShader,/source\.a/);effects.render(scene,{},profile);assert.equal(effects.target.width,200);
 let message='';effects.report=text=>{message=text;};renderer.debug.onShaderError({getShaderInfoLog:()=> 'bad shader'},null,null,null);assert.equal(effects.failed,true);assert.match(message,/disabled/);
 effects.dispose();assert.equal(scene.children.length,0);
});
