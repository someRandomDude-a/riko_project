import React, {useEffect, useRef, useState} from 'react';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin, VRMUtils} from '@pixiv/three-vrm';
import {MotionAssetSampler} from './motion_assets.mjs';
import {MotionEngine, BODY_BONES} from './motion_engine.mjs';
import {DesktopWalker, nearbyPointer} from './motion_geometry.mjs';
import {cameraDistance, resizeHeldAvatar} from './avatar_geometry.mjs';
import {API, request} from './api.mjs';
import {avatarModelURL,validateAvatarFormat} from './avatar_model.mjs';
import {pointerSampler} from './avatar_pointer.mjs';
import {AvatarBoneBVH} from './avatar_bvh.mjs';
import {AvatarHitOutlines} from './avatar_hit_outlines.mjs';
import {boneCatalog,publishCatalog,AvatarSecondaryMotion,AvatarSkeletonDebug} from './avatar_secondary.mjs';
import {normalizeProfile} from './avatar_studio_settings.mjs';
import {AvatarEffects} from './avatar_effects.mjs';
import {AvatarPickup} from './avatar_pickup.mjs';
import {useAvatarEditorOpen} from './avatar_editor_channel.mjs';
import {AvatarGesture,AvatarHover} from './avatar_input.mjs';
import {AvatarMouseSphere} from './avatar_mouse_sphere.mjs';
import {normalizeGraphics,applyTextureFiltering} from './avatar_graphics.mjs';
import {AvatarEyeGaze} from './avatar_gaze.mjs';

const EXPRESSION_NAMES = ['happy', 'sad', 'angry', 'surprised', 'neutral'];
const EMOTION_MAP = {joy: 'happy', love: 'happy', affection: 'happy', sadness: 'sad', anger: 'angry', surprise: 'surprised', calm: 'neutral', neutral: 'neutral'};

function clamp(value, min = 0, max = 1) { return Math.max(min, Math.min(max, Number(value) || 0)); }

/** Native VRM presentation layer. The runtime supplies intent; this loop owns animation frames. */
export default function AvatarRenderer({emotion, modelPath, modelFormat='auto', fov=30, actions = [], animation={}, preferences={}, geometry = {x:0,y:0,width:480,height:720}, onPosition,onError,onDragging}) {
  const canvasRef = useRef(null);
  const [pointerState,setPointerState]=useState('idle');
  const [shaderError,setShaderError]=useState('');
  const editorOpen=useAvatarEditorOpen(false);
  const graphics=normalizeGraphics(preferences).avatarGraphics;
  const draggingRef=useRef(onDragging);draggingRef.current=onDragging;
  const preferencesRef=useRef(preferences);preferencesRef.current=preferences;
  const emotionRef = useRef(emotion);
  emotionRef.current = emotion;
  const actionsRef = useRef(actions);
  actionsRef.current = actions;
  const geometryRef = useRef(geometry), positionRef = useRef(onPosition);
  geometryRef.current = geometry; positionRef.current = onPosition;
  const animationRef=useRef(animation);animationRef.current=animation;
  const errorRef=useRef(onError);errorRef.current=onError;

  useEffect(() => {
    const canvas = canvasRef.current;
    const renderer = new THREE.WebGLRenderer({canvas, alpha: true, antialias:graphics.antialias, powerPreference:'high-performance'});
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, graphics.pixelRatio));
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(fov, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
    const effects=new AvatarEffects(renderer,scene,setShaderError,graphics.samples);
    setShaderError('');
    let catalog=null,secondary=null,skeletonDebug=null,pickup=null,mouseSphere=null,eyeGaze=null,previousTPose=false,cursorPending=false,lastDragPoint=performance.now(),lastFilter=null,lastMousePoint=null;
    const gesture=new AvatarGesture(),hover=new AvatarHover();
    let profile=normalizeProfile({}),profileSource;

    const loader = new GLTFLoader();
    loader.register(parser => new VRMLoaderPlugin(parser));
    const report=(id,status,error='')=>{request('/api/avatar/animation/result',{method:'POST',body:{action_id:id,status,error}}).catch(()=>{});};
    const walker=new DesktopWalker(report);
    let motion=null;
    const interaction={held:false,pointer:null,reaction:'',until:0,target:null,rule:null};
    let lastPointerPost=0,lastHoldPost=0;
    function interact(kind,pointer=null,target=interaction.target){
      if(['hold','click','release','hover','leave','hoverDelayed'].includes(kind)){
       interaction.rule=preferencesRef.current.avatarHitRules?.[target?.bone]?.[kind]||null;
       if(['hover','leave','hoverDelayed'].includes(kind)){interaction.reaction=interaction.rule?'idle':'';interaction.until=performance.now()+(kind==='leave'?800:60000);}
       if(interaction.rule?.assetId)request('/api/animation/assets/'+encodeURIComponent(interaction.rule.assetId)+'/preview',{method:'POST'}).catch(()=>{});
      }
      const serverKind=interaction.localDrag&&['hold','release'].includes(kind)?'pointer':kind;
      request('/api/animation/interaction',{method:'POST',body:{kind:serverKind,pointer,target}}).catch(()=>{});
    }
    let vrm = null;
    let pickBVH=null,hitOutlines=null;
    let disposed = false;
    let frame = 0;
    const targetExpressions = Object.fromEntries(EXPRESSION_NAMES.map(name => [name, 0]));
    let blink = 0;
    let blinkCooldown = 2.5;
    const clock = new THREE.Clock();
    let modelSize, modelCenter;
    function fitModel() {
      if (!modelSize) return;
      const distance = cameraDistance(modelSize, camera.aspect, camera.fov);
      camera.near = Math.max(.001, distance / 1000);
      camera.far = Math.max(100, distance + modelSize.length() * 4);
      camera.position.set(modelCenter.x, modelCenter.y, modelCenter.z + distance);
      camera.lookAt(modelCenter);
      camera.updateProjectionMatrix();
    }

    const source=modelPath?avatarModelURL(API,modelPath,modelFormat):new URL('models/Mita.vrm',document.baseURI).href;
    loader.load(source, gltf => {
      if (disposed) { VRMUtils.deepDispose(gltf.scene); return; }
      vrm = gltf.userData.vrm;
      if (!vrm) { VRMUtils.deepDispose(gltf.scene); errorRef.current?.('Selected model is not a supported VRM');return; }
      try{validateAvatarFormat(vrm.meta?.metaVersion,modelFormat);}catch(error){VRMUtils.deepDispose(gltf.scene);vrm=null;errorRef.current?.(error.message);return;}
      errorRef.current?.('');
      VRMUtils.rotateVRM0(vrm);
      scene.add(vrm.scene);
      catalog=boneCatalog(vrm,modelPath||'models/Mita.vrm');publishCatalog(catalog);
      secondary=new AvatarSecondaryMotion(vrm,catalog);skeletonDebug=new AvatarSkeletonDebug(vrm,catalog,scene);
      mouseSphere=new AvatarMouseSphere(catalog,scene);
      eyeGaze=new AvatarEyeGaze(vrm);
      pickup=new AvatarPickup(vrm,catalog,kind=>{interaction.rule=preferencesRef.current.avatarHitRules?.[interaction.grabTarget?.bone]?.[kind]||null;if(interaction.rule?.assetId)request('/api/animation/assets/'+encodeURIComponent(interaction.rule.assetId)+'/preview',{method:'POST'}).catch(()=>{});if(kind==='recover'){interaction.reaction='recovering';interaction.until=performance.now()+profile.pickup.recoverySeconds*1000;}request('/api/animation/interaction',{method:'POST',body:{kind,pointer:interaction.pointer,target:interaction.grabTarget}}).catch(()=>{});});
      const bounds = new THREE.Box3().setFromObject(vrm.scene);
      const size = bounds.getSize(new THREE.Vector3());
      vrm.scene.position.y -= bounds.min.y;
      vrm.scene.updateMatrixWorld(true);
      modelSize = size;
      modelCenter = new THREE.Box3().setFromObject(vrm.scene).getCenter(new THREE.Vector3());
      pickBVH=AvatarBoneBVH.fromVRM(vrm,size.y,catalog);
      hitOutlines=new AvatarHitOutlines(pickBVH,scene);
      fitModel();
      motion=new MotionEngine(vrm,new MotionAssetSampler(vrm),report);
      const bones=Object.keys(vrm.humanoid.normalizedHumanBones||{}).filter(name=>vrm.humanoid.getNormalizedBoneNode(name));
      request('/api/animation/capabilities',{method:'POST',body:{bones:bones.length?bones:BODY_BONES.filter(name=>vrm.humanoid.getNormalizedBoneNode(name)),expressions:Object.keys(vrm.expressionManager?.expressionMap||{})}}).catch(()=>{});
    }, undefined, error => {if(!disposed){console.warn(`VRM model unavailable (${modelPath || source}):`, error);errorRef.current?.('Avatar model could not be loaded. Check the selected VRM file.');}});

    const resize = () => {
      camera.aspect = canvas.clientWidth / canvas.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
      fitModel();
    };
    addEventListener('resize', resize);
    const observer = new ResizeObserver(resize); observer.observe(canvas);
    const raycaster = new THREE.Raycaster(), ndc=new THREE.Vector2(), samples=pointerSampler(); let drag = null, hit = false, interactive = false;
    function feedback(){setPointerState(drag?(drag.kind==='spring'?'spring:':'held:')+interaction.target?.bone:hit?'hover:'+interaction.target?.bone:'idle');canvas.style.cursor=drag?'grabbing':hit?'grab':'default';}
    function setInteractive(value) {
      if (value === interactive) return;
      interactive = value;
      window.riko?.avatarInteractive(value);
    }
    function pick(x,y,fresh=false){
      const rect=canvas.getBoundingClientRect();
      if(!vrm||!rect.width||!rect.height||x<rect.left||x>rect.right||y<rect.top||y>rect.bottom){interaction.target=null;return false;}
      ndc.set((x-rect.left)/rect.width*2-1,1-(y-rect.top)/rect.height*2);
      raycaster.setFromCamera(ndc,camera);
       if(fresh){vrm.scene.updateMatrixWorld(true);pickBVH?.refit(profile.springPickRadius);}
      interaction.target=pickBVH?.hit(raycaster.ray)||null;
      return !!interaction.target;
    }
    const mousemove = event => {samples.queue(event);};
    function processPointer(point){
      lastMousePoint=point;mouseSphere?.aim(point,canvas.getBoundingClientRect(),camera,(modelCenter?.z||0)+profile.mouseSphere.depth);
      interaction.pointer=nearbyPointer(geometryRef.current,point,interaction.pointer?.near);
      if(performance.now()-lastPointerPost>150){lastPointerPost=performance.now();interact('pointer',interaction.pointer);}
      if (drag) {
        if(gesture.move(point,profile.input.dragThreshold)){
         drag.started=true;interaction.held=drag.kind==='body';interaction.reaction='';walker.stop();interact('hold',interaction.pointer);
          if(drag.kind==='spring'){mouseSphere.start(drag.entry,drag.springGrab);mouseSphere.aim(point,canvas.getBoundingClientRect(),camera,(modelCenter?.z||0)+profile.mouseSphere.depth);interaction.reaction='idle';interaction.until=performance.now()+60000;}else if(profile.pickup.enabled)pickup?.start(interaction.target.bone,gesture.press.origin,camera);
        }
        if(!drag.started)return;
        if(drag.kind==='spring')return;
        pickup?.move(point,(performance.now()-lastDragPoint)/1000,profile.pickup);lastDragPoint=performance.now();
        drag.geometry = {...drag.geometry, x: drag.left + point.x-drag.x, y: drag.top + point.y-drag.y};
        geometryRef.current=drag.geometry;
        positionRef.current?.(drag.geometry.x, drag.geometry.y, false, drag.geometry);
        return;
      }
      hit=pick(point.x,point.y);
      for(const event of hover.update(interaction.target,performance.now(),profile.input.hoverDelay))interact(event.kind,interaction.pointer,event.target);
      setInteractive(hit);feedback();
    }
    const pointerdown = event => {
      if (event.button !== 0||drag) return;
      // Do not rely on the last throttled hover result for pickup.
      hit=pick(event.clientX,event.clientY,true);
      if(!hit)return;
      samples.clear();
      interaction.pointer=nearbyPointer(geometryRef.current,{x:event.clientX,y:event.clientY},interaction.pointer?.near);
      const entry=catalog.entries.find(e=>e.id===interaction.target.bone||e.human===interaction.target.bone);
       drag = {x: event.clientX, y: event.clientY, left: geometryRef.current.x, top: geometryRef.current.y, geometry: {...geometryRef.current},started:false,kind:entry&&(entry.joint||!entry.human&&profile.bones[entry.id]?.enabled)?'spring':'body',entry};
       window.overlayInputBridge?.drag('avatar',true);
       if(drag.kind==='spring')drag.springGrab=mouseSphere.prepare(entry,{x:event.clientX,y:event.clientY},canvas.getBoundingClientRect(),camera);
      interaction.localDrag=drag.kind==='spring';
      hover.update(interaction.target,performance.now(),profile.input.hoverDelay);
      gesture.down({x:event.clientX,y:event.clientY},{...interaction.target});mouseSphere?.aim({x:event.clientX,y:event.clientY},canvas.getBoundingClientRect(),camera,(modelCenter?.z||0)+profile.mouseSphere.depth);
       lastMousePoint={x:event.clientX,y:event.clientY};draggingRef.current?.(true);interaction.grabTarget={...interaction.target};lastDragPoint=performance.now();
      if(event.pointerId!==undefined)try{canvas.setPointerCapture(event.pointerId);}catch{}
      setInteractive(true);
      feedback();
      event.preventDefault();
    };
    const pointerup = event => {
      if (drag) {
        processPointer({x:event.clientX,y:event.clientY});
         const origin = drag; drag = null;
         window.overlayInputBridge?.drag('avatar',false);
        const clicked=!origin.started;gesture.up();
        interaction.held=false;interaction.reaction=clicked?'clicked':origin.kind==='spring'?'idle':'settling';interaction.until=performance.now()+800;
        interact(clicked?'click':'release',interaction.pointer);
        const final=clicked||origin.kind==='spring'?origin.geometry:{...origin.geometry,x:origin.left+event.clientX-origin.x,y:origin.top+event.clientY-origin.y};
        if(!clicked&&origin.kind==='body')positionRef.current?.(final.x,final.y,true,final);
        if(clicked)mouseSphere?.click();mouseSphere?.release();
        geometryRef.current=final;draggingRef.current?.(false);pickup?.release();
        if(event.pointerId!==undefined&&canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);
      }
      samples.clear();hit=pick(event.clientX,event.clientY);setInteractive(hit);feedback();
    };
     const pointercancel = () => {
       window.overlayInputBridge?.drag('avatar',false);
      if (drag?.started){if(drag.kind==='body')positionRef.current?.(drag.geometry.x, drag.geometry.y, true, drag.geometry);interact('release');interaction.held=false;interaction.reaction='settling';interaction.until=performance.now()+800;}
      gesture.up();mouseSphere?.release();
      drag = null;pickup?.release();draggingRef.current?.(false); hit = false;samples.clear(); setInteractive(false);feedback();
    };
    const leave=()=>{if(!drag){for(const event of hover.update(null,performance.now(),profile.input.hoverDelay))interact(event.kind,null,event.target);interaction.pointer=null;interaction.target=null;hit=false;samples.clear();if(mouseSphere)mouseSphere.active=false;setInteractive(false);feedback();}};
    addEventListener('mouseleave',leave);
    addEventListener('pointerup',pointerup);addEventListener('mouseup',pointerup);addEventListener('pointermove',mousemove);
    const mouseFallback=event=>pointerdown(event);canvas.addEventListener('mousedown',mouseFallback);
    // Electron's click-through forwarding emits mousemove, not pointermove.
    // Use it for hover; captured pointermove continues dragging off the mesh.
    addEventListener('mousemove', mousemove);
    canvas.addEventListener('pointerdown', pointerdown);
    canvas.addEventListener('pointermove', mousemove);
    canvas.addEventListener('pointerup', pointerup);
    canvas.addEventListener('pointercancel', pointercancel);
    const lostCapture=()=>{if(drag&&!window.overlayInputBridge?.native)pointercancel();};
    canvas.addEventListener('lostpointercapture', lostCapture);
    const nativeOff=window.overlayInputBridge?.subscribe(point=>{if(drag&&point.phase==='up')pointerup({clientX:point.x,clientY:point.y});});
    const wheel = event => {
      if (!drag?.started||drag.kind!=='body') return;
      event.preventDefault();
      samples.clear();
      const geometry = resizeHeldAvatar(drag.geometry, event.deltaY, {x:event.clientX,y:event.clientY}, {width:innerWidth,height:innerHeight});
      drag = {...drag,x:event.clientX,y:event.clientY,left:geometry.x,top:geometry.y,geometry};
      positionRef.current?.(geometry.x, geometry.y, false, geometry);
    };
    canvas.addEventListener('wheel', wheel, {passive:false});

    const animate = () => {
      frame = requestAnimationFrame(animate);
      const delta = Math.min(clock.getDelta(), 0.1);
      const elapsed = clock.elapsedTime;
      const point=samples.take(performance.now(),!!drag);if(point)processPointer(point);
      if(drag&&!cursorPending&&window.avatarCursorBridge){cursorPending=true;const currentDrag=drag;window.avatarCursorBridge.position().then(point=>{if(!disposed&&drag===currentDrag){if(point.buttons===0)pointerup({clientX:point.x,clientY:point.y});else samples.queue({clientX:point.x,clientY:point.y});}}).catch(()=>{}).finally(()=>{cursorPending=false;});}
      const current = emotionRef.current || {};
      const nextProfile=preferencesRef.current.avatarStudioProfiles?.[catalog?.key];
      if(nextProfile!==profileSource){profileSource=nextProfile;profile=normalizeProfile(nextProfile);}
      effects.update(profile,elapsed);
      if (vrm) {
        eyeGaze?.restore();mouseSphere?.restore();pickup?.restore();secondary.restore();
        if(!drag&&(!interaction.reaction||interaction.reaction==='idle'))for(const event of hover.update(interaction.target,performance.now(),profile.input.hoverDelay))interact(event.kind,interaction.pointer,event.target);
        const filter=preferencesRef.current.avatarGraphics?.anisotropy??8;if(filter!==lastFilter){lastFilter=filter;applyTextureFiltering(vrm.scene,renderer,{anisotropy:filter});}
        const expression = vrm.expressionManager;
        const custom=(interaction.held||performance.now()<interaction.until)?interaction.rule:null;
        const active = custom?.expression&&custom.expression!=='default'?custom.expression:EMOTION_MAP[current.primary] || 'neutral';
        for (const name of EXPRESSION_NAMES) {
          targetExpressions[name] = name === active ? clamp(custom?.expression&&custom.expression!=='default'?custom.intensity:current.intensity ?? 0.65) : 0;
          const existing = expression?.getValue(name) || 0;
          expression?.setValue(name, THREE.MathUtils.damp(existing, targetExpressions[name], 8, delta));
        }
        blinkCooldown -= delta;
        if (blinkCooldown <= 0) { blink = 1; blinkCooldown = 3 + Math.random() * 4; }
        blink = Math.max(0, blink - delta * 7);
        expression?.setValue('blink', blink);

        // Small procedural idle motion keeps the model alive without tracking.
        vrm.scene.rotation.y = (vrm.meta?.metaVersion === '0' ? Math.PI : 0) + Math.sin(elapsed * 0.7) * 0.025;
        vrm.scene.position.x = Math.sin(elapsed * 0.45) * 0.006;
        if(performance.now()>interaction.until)interaction.reaction='';
        if(interaction.held&&performance.now()-lastHoldPost>500){lastHoldPost=performance.now();interact('drag',interaction.pointer);}
        const walk=profile.tPose?null:walker.step(actionsRef.current.find(a=>a.kind==='motion.locomotion'&&a.status==='running'),geometryRef.current,{width:innerWidth,height:innerHeight},delta,interaction.held||!!drag?.started);
        if(walk){positionRef.current?.(walk.x,walk.y,walk.final,walk);vrm.scene.rotation.y+=(walk.facing<0?.35:-.35);}
        if(profile.tPose)secondary.preview();
        else motion?.update({actions:actionsRef.current,emotion:current,interaction,walking:!!walker.active,walkSpeed:walker.velocity/(animationRef.current.settings?.walk_speed||240),settings:{...animationRef.current.settings,enabled:animationRef.current.enabled}},delta);
        // Same order as VRM.update, with non-destructive calibration before springs
        // and rotational secondary motion after them. Never simulate springs twice.
        vrm.humanoid.update();secondary.calibrate(profile);
        if(custom?.expression&&custom.expression!=='default')for(const name of EXPRESSION_NAMES)expression?.setValue(name,name===active?custom.intensity:0);
        vrm.expressionManager?.update();
        vrm.nodeConstraintManager?.update();
        if(profile.tPose!==previousTPose){vrm.scene.updateMatrixWorld(true);vrm.springBoneManager?.reset();previousTPose=profile.tPose;}
        if(!profile.tPose)vrm.springBoneManager?.update(delta);
        for(const material of vrm.materials||[])material.update?.(delta);
        secondary.update(profile,delta);
        if(!profile.tPose)pickup?.step(profile.pickup,delta);
        if(!profile.tPose){vrm.scene.updateMatrixWorld(true);if(lastMousePoint)mouseSphere.aim(lastMousePoint,canvas.getBoundingClientRect(),camera,(modelCenter?.z||0)+profile.mouseSphere.depth);mouseSphere?.update(profile,delta);}
        vrm.scene.updateMatrixWorld(true);
        eyeGaze?.update({...profile.gaze,enabled:profile.gaze.enabled&&!profile.tPose&&animationRef.current.settings?.mouse_tracking!==false},delta,lastMousePoint,canvas.getBoundingClientRect(),camera,mouseSphere,!!interaction.pointer?.near);
         vrm.expressionManager?.update();vrm.scene.updateMatrixWorld(true);pickBVH?.refit(profile.springPickRadius);
        skeletonDebug.update(profile);hitOutlines?.update(preferencesRef.current,interaction.target);
      }
      const activeRule=(interaction.held||performance.now()<interaction.until)?interaction.rule:null;
      const visualProfile=activeRule?.effect?{...profile,effect:{...profile.effect,preset:activeRule.effect}}:profile;
      effects.update(visualProfile,elapsed);effects.render(scene,camera,visualProfile);
    };
    animate();
    return () => {
       disposed = true;
       nativeOff?.();window.overlayInputBridge?.drag('avatar',false);
      cancelAnimationFrame(frame);
      removeEventListener('resize', resize);
      observer.disconnect(); removeEventListener('mousemove', mousemove);
      removeEventListener('mouseleave',leave);removeEventListener('pointerup',pointerup);removeEventListener('mouseup',pointerup);removeEventListener('pointermove',mousemove);canvas.removeEventListener('mousedown',mouseFallback);draggingRef.current?.(false);
      canvas.removeEventListener('pointerdown', pointerdown);
      canvas.removeEventListener('pointermove', mousemove);
      canvas.removeEventListener('pointerup', pointerup);
      canvas.removeEventListener('pointercancel', pointercancel);
      canvas.removeEventListener('lostpointercapture', lostCapture);
      canvas.removeEventListener('wheel', wheel);
      window.riko?.avatarInteractive(false);
      eyeGaze?.dispose();mouseSphere?.restore();pickup?.restore();secondary?.restore();effects.dispose();renderer.dispose();
      walker.stop();motion?.dispose();interact('leave');
      VRMUtils.deepDispose(scene);
    };
  }, [modelPath,modelFormat,fov,graphics.antialias,graphics.samples,graphics.pixelRatio]);

  return <><canvas key={String(graphics.antialias)} ref={canvasRef} className="avatar-canvas" style={{inset:'auto',left:geometry.x,top:geometry.y,width:geometry.width,height:geometry.height,pointerEvents:'auto'}} aria-label="Character avatar" />
    {shaderError&&<div className="avatar-grab-feedback" role="alert" style={{left:geometry.x+geometry.width/2,top:Math.max(8,geometry.y+40)}}>{shaderError}</div>}
    {editorOpen&&pointerState!=='idle'&&<div className="avatar-grab-feedback" role="status" style={{left:geometry.x+geometry.width/2,top:Math.max(8,geometry.y+8)}}>{pointerState.startsWith('spring:')?'Spring endpoint · drag to pull':pointerState.startsWith('held:')?'Holding · drag to move · scroll to resize':'Drag to pick up'} · {pointerState.split(':')[1]}</div>}</>;
}
