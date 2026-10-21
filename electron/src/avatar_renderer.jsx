import React, {useEffect, useRef, useState} from 'react';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin, VRMUtils} from '@pixiv/three-vrm';
import {MotionAssetSampler} from './motion_assets.mjs';
import {MotionEngine, BODY_BONES} from './motion_engine.mjs';
import {DesktopWalker, nearbyPointer} from './motion_geometry.mjs';
import {cameraDistance, slidePixels, dragPosition, resizeDrag} from './avatar_geometry.mjs';
import {API, request} from './api.mjs';
import {avatarModelURL,validateAvatarFormat,keepAvatarDrawn} from './avatar_model.mjs';
import {pointerSampler,pointerReporter} from './avatar_pointer.mjs';
import {AvatarBoneBVH} from './avatar_bvh.mjs';
import {AvatarHitOutlines} from './avatar_hit_outlines.mjs';
import {boneCatalog,publishCatalog,AvatarSecondaryMotion,AvatarSkeletonDebug} from './avatar_secondary.mjs';
import {normalizeProfile} from './avatar_studio_settings.mjs';
import {AvatarEffects} from './avatar_effects.mjs';
import {AvatarPickup,shoeTips} from './avatar_pickup.mjs';
import {useAvatarEditorOpen} from './avatar_editor_channel.mjs';
import {AvatarGesture,AvatarHover,hoverReported} from './avatar_input.mjs';
import {AvatarMouseSphere} from './avatar_mouse_sphere.mjs';
import {normalizeGraphics,applyTextureFiltering} from './avatar_graphics.mjs';
import {AvatarEyeGaze,gazeActive} from './avatar_gaze.mjs';
import {AvatarInertia,withSpringForce} from './avatar_inertia.mjs';
import {AvatarFace,registerFaceParts,reportedExpressions} from './avatar_face.mjs';
import {HoldBehaviour,HoldFace,HELD_STATES,DROP_STATES,passiveHold} from './avatar_hold.mjs';
import {Release,workArea} from './avatar_drop.mjs';
import {frameLoop,frameInterval,avatarActive,motionBusy} from './avatar_frame_rate.mjs';
import {connectEvents} from './event_connection.mjs';

/** Where the hold reactions look, from her (world metres): the holder's hand above and in front of the scruff, the floor ahead of her feet. */
const HOLDER=new THREE.Vector3(0,.3,.35),FLOOR=new THREE.Vector3(0,-1,.6);
/** The bones whose projection bounds her body for the walls a throw bounces off, re-measured every frame of the flight (her arms fling
 * out on the way): the skin reaches about 8 cm past them, the hair about 24 cm above the head bone (Miku tilted 30 degrees in the air). */
const BOX_BONES=['head','neck','hips','leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','leftHand','rightHand','leftFoot','rightFoot','leftToes','rightToes'];
/** The depth (world z) of her shoe tips standing (avatar_pickup.mjs shoeTips; Miku: 96 mm, the screen-lowest shoe vertex). */
function shoeDepth(vrm,fallback){const tips=shoeTips(vrm.humanoid);return tips.length?tips.reduce((sum,tip)=>sum+tip.z,0)/tips.length:fallback;}

/** Native VRM presentation layer. The runtime supplies intent; this loop owns animation frames. */
export default function AvatarRenderer({emotion, modelPath, modelFormat='auto', fov=30, actions = [], animation={}, preferences={}, geometry = {x:0,y:0,width:480,height:720}, asleep=false, onPosition,onError,onDragging}) {
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
  const asleepRef=useRef(asleep);asleepRef.current=asleep;
  // The render loop's wake (avatar_frame_rate.mjs): a change from the backend or the studio draws now, even while she idles or sleeps,
  // and asleep she then draws at the idle rate until she has settled again (a face fade or a pose change finishes before she pauses).
  const wakeRef=useRef(null);

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
    let profile=normalizeProfile({}),profileSource,holdSettings={...profile.reactions,recoverySeconds:profile.pickup.recoverySeconds,landingSquash:profile.pickup.landingSquash};

    const loader = new GLTFLoader();
    loader.register(parser => new VRMLoaderPlugin(parser));
    const report=(id,status,error='')=>{request('/api/avatar/animation/result',{method:'POST',body:{action_id:id,status,error}}).catch(()=>{});};
    const walker=new DesktopWalker(report);
    // The window's motion moves the springs and swings the carry rig; the cursor's (the window without the scruff slide) drives the hold reactions.
    const inertia=new AvatarInertia(),cursorMotion=new AvatarInertia(),springForce=new THREE.Vector3(),hold=new HoldBehaviour();
    let motion=null,face=null,holdFace=new HoldFace(null);
    const interaction={held:false,pointer:null,reaction:'',until:0,target:null,rule:null,hold:null};
    let lastHoldPost=0,lastWorldPerPixel=0,capabilities=null,pointerAt=-Infinity,movedAt=-Infinity,quietSince=performance.now();
    // The backend hears where the pointer is only when it comes near her or goes, and once a second while it moves near her.
    const pointerReports=pointerReporter();
    const lookPoint=new THREE.Vector3(),scratch=new THREE.Vector3(),ground=new THREE.Vector3(),restCamera=new THREE.PerspectiveCamera();
    // Letting go (avatar_drop.mjs): the work area of the overlay's display, where her soles are in the canvas with the current camera
    // (the safety net may still be easing back) or the one at rest, and her body's extent in the canvas for the walls.
    const area=()=>workArea(window.screen,{x:window.screenX,y:window.screenY},{width:innerWidth,height:innerHeight});
    // Her soles are at y 0, and the tips of her shoes are the lowest point she shows: in perspective, nearer the camera is lower on screen
    // (on Miku the shoe tip is 6.5 px below a point at the model's centre depth; modelFront puts it within 0.1 px).
    function footLine(rest=false){
     if(!modelCenter)return canvas.clientHeight;let view=camera;
     if(rest&&pickup?.cameraActive&&pickup.cameraRest){view=restCamera.copy(camera);view.position.copy(pickup.cameraRest.position);view.quaternion.copy(pickup.cameraRest.quaternion);view.updateMatrixWorld(true);}
     ground.set(modelCenter.x,0,modelFront).project(view);return (1-ground.y)/2*canvas.clientHeight;
    }
    function bodyBox(){
     const w=canvas.clientWidth,h=canvas.clientHeight,box={left:w,right:0,top:h};
     for(const name of BOX_BONES){const node=vrm?.humanoid.getRawBoneNode(name);if(!node)continue;const p=node.getWorldPosition(scratch).project(camera),x=(p.x+1)/2*w,y=(1-p.y)/2*h;box.left=Math.min(box.left,x);box.right=Math.max(box.right,x);box.top=Math.min(box.top,y);}
     const pad=lastWorldPerPixel>0?.08/lastWorldPerPixel:0;
     return box.left<=box.right?{left:box.left-pad,right:box.right+pad,top:box.top-3*pad}:{left:0,right:w,top:0};
    }
    function headPoint(){return vrm?.humanoid.getRawBoneNode('head')?.getWorldPosition(scratch)||scratch.copy(modelCenter||camera.position);}
    // The release lifecycle (throw, fall, set-down, catch and the one final PATCH); the carry rig joins it once the model has loaded.
    const release=new Release({place:(g,final)=>{geometryRef.current=g;positionRef.current?.(g.x,g.y,final,g);},dragging:value=>draggingRef.current?.(value),interact:kind=>interact(kind),hold,
     measure:{area,foot:footLine,box:bodyBox,worldPerPixel:()=>lastWorldPerPixel,height:()=>canvas.clientHeight}});
    function interact(kind,pointer=null,target=interaction.target,post=true){
      if(['hold','click','release','hover','leave','hoverDelayed'].includes(kind)){
       interaction.rule=preferencesRef.current.avatarHitRules?.[target?.bone]?.[kind]||null;
       if(['hover','leave','hoverDelayed'].includes(kind)){interaction.reaction=interaction.rule?'idle':'';interaction.until=performance.now()+(kind==='leave'?800:60000);}
       if(interaction.rule?.assetId)request('/api/animation/assets/'+encodeURIComponent(interaction.rule.assetId)+'/preview',{method:'POST'}).catch(()=>{});
      }
      if(!post)return;
      if(kind==='leave')pointerReports.reset();
      const serverKind=interaction.localDrag&&['hold','release'].includes(kind)?'pointer':kind;
      request('/api/animation/interaction',{method:'POST',body:{kind:serverKind,pointer,target}}).catch(()=>{});
    }
    // Hover events run their local hit rules; the backend hears them as hoverReported says (one 'hover' for a move between bones).
    function hovered(events,pointer){const reported=hoverReported(events);events.forEach((event,i)=>interact(event.kind,pointer,event.target,reported[i]));}
    let vrm = null;
    let pickBVH=null,hitOutlines=null;
    let disposed = false,headNode = null;
    const clock = new THREE.Clock();
    let modelSize, modelCenter, modelFront=0;
    function fitModel() {
      if (!modelSize) return;
      const distance = cameraDistance(modelSize, camera.aspect, camera.fov);
      camera.near = Math.max(.001, distance / 1000);
      camera.far = Math.max(100, distance + modelSize.length() * 4);
      camera.position.set(modelCenter.x, modelCenter.y, modelCenter.z + distance);
      camera.lookAt(modelCenter);
      camera.updateProjectionMatrix();
    }

    // The backend serves the configured VRM (its default when none is set); there is no bundled fallback model.
    const source=avatarModelURL(API,modelPath||'',modelFormat);
    loader.load(source, gltf => {
      if (disposed) { VRMUtils.deepDispose(gltf.scene); return; }
      vrm = gltf.userData.vrm;
      if (!vrm) { VRMUtils.deepDispose(gltf.scene); errorRef.current?.('Selected model is not a supported VRM');return; }
      try{validateAvatarFormat(vrm.meta?.metaVersion,modelFormat);}catch(error){VRMUtils.deepDispose(gltf.scene);vrm=null;errorRef.current?.(error.message);return;}
      errorRef.current?.('');
      VRMUtils.rotateVRM0(vrm);
      keepAvatarDrawn(vrm.scene);
      scene.add(vrm.scene);
      catalog=boneCatalog(vrm,modelPath||'models/Mita.vrm');publishCatalog(catalog);  // the old fallback's name keeps that model's studio profile
      secondary=new AvatarSecondaryMotion(vrm,catalog);skeletonDebug=new AvatarSkeletonDebug(vrm,catalog,scene);
      mouseSphere=new AvatarMouseSphere(catalog,scene);
      eyeGaze=new AvatarEyeGaze(vrm);headNode=vrm.humanoid.getRawBoneNode('head');
      pickup=release.pickup=new AvatarPickup(vrm,catalog,kind=>{interaction.rule=preferencesRef.current.avatarHitRules?.[interaction.grabTarget?.bone]?.[kind]||null;if(interaction.rule?.assetId)request('/api/animation/assets/'+encodeURIComponent(interaction.rule.assetId)+'/preview',{method:'POST'}).catch(()=>{});if(kind==='recover'&&interaction.held)lastHoldPost=0;else if(kind==='recover'&&!profile.reactions.enabled){interaction.reaction='settling';interaction.until=performance.now()+800;}request('/api/animation/interaction',{method:'POST',body:{kind,pointer:interaction.pointer,target:interaction.grabTarget}}).catch(()=>{});});
      const bounds = new THREE.Box3().setFromObject(vrm.scene);
      const size = bounds.getSize(new THREE.Vector3());
      vrm.scene.position.y -= bounds.min.y;
      vrm.scene.updateMatrixWorld(true);
      modelSize = size;
      modelCenter=new THREE.Box3().setFromObject(vrm.scene).getCenter(new THREE.Vector3());modelFront=shoeDepth(vrm,modelCenter.z);
      pickBVH=AvatarBoneBVH.fromVRM(vrm,size.y,catalog);
      hitOutlines=new AvatarHitOutlines(pickBVH,scene);
      fitModel();
      motion=new MotionEngine(vrm,new MotionAssetSampler(vrm),report);
      // Face parts register after the bone catalog (its ids never move) and before the capabilities report (64 names of 1-64 characters at most).
      registerFaceParts(vrm);face=vrm.expressionManager?new AvatarFace(vrm.expressionManager):null;holdFace=new HoldFace(face);
      const bones=Object.keys(vrm.humanoid.normalizedHumanBones||{}).filter(name=>vrm.humanoid.getNormalizedBoneNode(name));
      capabilities={bones:bones.length?bones:BODY_BONES.filter(name=>vrm.humanoid.getNormalizedBoneNode(name)),expressions:reportedExpressions(vrm.expressionManager)};
      reportCapabilities();rouse();
    }, undefined, error => {if(!disposed){console.warn(`VRM model unavailable (${modelPath || source}):`, error);errorRef.current?.('Avatar model could not be loaded. Check the selected VRM file.');}});

    const resize = () => {
      camera.aspect = canvas.clientWidth / canvas.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
      fitModel();rouse();  // a resize clears the canvas: draw again even when paused
    };
    addEventListener('resize', resize);
    const observer = new ResizeObserver(resize); observer.observe(canvas);
    const raycaster = new THREE.Raycaster(), ndc=new THREE.Vector2(), samples=pointerSampler(); let drag = null, hit = false, interactive = false;
    function feedback(){setPointerState(drag?(drag.kind==='spring'?'spring:':'held:')+interaction.target?.bone:hit?'hover:'+interaction.target?.bone:'idle');canvas.style.cursor=drag?'grabbing':hit?'grab':'default';}
    // The scruff slide in px for a drag (its NDC offset times the current canvas size), and the drag's geometry with it.
    function slideOf(current){return current?.kind==='body'&&pickup?.held?pickup.slide():null;}
    function slidePx(current){return slidePixels(slideOf(current),current.geometry);}
    function placeDrag(){const s=slidePx(drag);drag.slid=s;drag.geometry={...drag.geometry,...dragPosition(drag,drag.point,s)};geometryRef.current=drag.geometry;positionRef.current?.(drag.geometry.x,drag.geometry.y,false,drag.geometry);}
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
       if(fresh)vrm.scene.updateMatrixWorld(true);
       pickBVH?.refit(profile.springPickRadius);
      interaction.target=pickBVH?.hit(raycaster.ray)||null;
      return !!interaction.target;
    }
    const mousemove = event => {
      samples.queue(event);
      // Over or near her (where her eyes and head follow it: nearbyPointer) the loop runs at the display's rate for a moment; elsewhere
      // it stays idle, except that a move off her while the overlay takes the pointer is handled at once (it must turn click-through again).
      const g=geometryRef.current,x=event.clientX,y=event.clientY;
      if(x>=g.x&&x<=g.x+g.width&&y>=g.y&&y<=g.y+g.height||nearbyPointer(g,{x,y},interaction.pointer?.near).near){pointerAt=performance.now();loop.wake();}else if(interactive)loop.wake();
    };
    /** The drag threshold was crossed (or she was caught mid-fall): she is held. */
    function startGrab(point){
     drag.started=true;interaction.held=drag.kind==='body';interaction.reaction='';walker.stop();interact('hold',interaction.pointer);hold.grab(performance.now()/1000,{region:interaction.grabTarget?.region,kind:drag.kind,emotion:emotionRef.current,caught:!!drag.caught});
     if(drag.kind==='spring'){mouseSphere.start(drag.entry,drag.springGrab);mouseSphere.aim(point,canvas.getBoundingClientRect(),camera,(modelCenter?.z||0)+profile.mouseSphere.depth);interaction.reaction='idle';interaction.until=performance.now()+60000;}else if(profile.pickup.enabled)pickup?.start(interaction.target.bone,gesture.press.origin,camera,{style:profile.pickup.grabStyle,pointer:drag.ndc});
    }
    /** A body drag ended at canvas position `final` with the cursor's velocity (px/s): thrown (she flies to the floor) or set down in
     * place (onto the floor when her feet would end below it or just above it). Release sends the one PATCH (final=true) once she
     * stands and the camera's safety net is back at rest. */
    function letGo(origin,final,velocity){
     cursorMotion.shift(origin.slid?.x||0,origin.slid?.y||0);  // it tracked the canvas without the scruff slide; from now on with it
     release.letGo(performance.now()/1000,final,velocity,{slid:origin.slid,settings:profile.pickup});
    }
    function processPointer(point){
      lastMousePoint=point;mouseSphere?.aim(point,canvas.getBoundingClientRect(),camera,(modelCenter?.z||0)+profile.mouseSphere.depth);
      interaction.pointer=nearbyPointer(geometryRef.current,point,interaction.pointer?.near);
      if(pointerReports.due(interaction.pointer,performance.now()))interact('pointer',interaction.pointer);
      if (drag) {
        if(gesture.move(point,profile.input.dragThreshold))startGrab(point);
        if(!drag.started)return;
        if(drag.kind==='spring')return;
        pickup?.move(point,(performance.now()-lastDragPoint)/1000,profile.pickup);lastDragPoint=performance.now();
        drag.point=point;placeDrag();
        return;
      }
      hit=pick(point.x,point.y);
      hovered(hover.update(interaction.target,performance.now(),profile.input.hoverDelay),interaction.pointer);
      setInteractive(hit);feedback();
    }
    const pointerdown = event => {
      if (event.button !== 0||drag) return;
      pointerAt=performance.now();loop.wake();
      // Do not rely on the last throttled hover result for pickup.
      hit=pick(event.clientX,event.clientY,true);
      if(!hit)return;
      // A press on her mid-fall catches her (a bigger surprise): the flight, or a glide onto the floor, stops where she is and she is held
      // at once, with no drag threshold. Standing, she is only waiting for the camera to settle: her place is sent now.
      const {caught,grabbed}=release.grab();
      if(grabbed)cursorMotion.reset();  // it tracked the flight; from now on only the hand moves her
      samples.clear();
      interaction.pointer=nearbyPointer(geometryRef.current,{x:event.clientX,y:event.clientY},interaction.pointer?.near);
      const entry=catalog.entries.find(e=>e.id===interaction.target.bone||e.human===interaction.target.bone);
       drag = {x: event.clientX, y: event.clientY, point:{x:event.clientX,y:event.clientY}, ndc:{x:ndc.x,y:ndc.y}, left: geometryRef.current.x, top: geometryRef.current.y, geometry: {...geometryRef.current},started:false,kind:!grabbed&&entry&&(entry.joint||!entry.human&&profile.bones[entry.id]?.enabled)?'spring':'body',entry,caught};
       window.overlayInputBridge?.drag('avatar',true);
       if(drag.kind==='spring')drag.springGrab=mouseSphere.prepare(entry,{x:event.clientX,y:event.clientY},canvas.getBoundingClientRect(),camera);
      interaction.localDrag=drag.kind==='spring';hold.press(performance.now()/1000);
      hover.update(interaction.target,performance.now(),profile.input.hoverDelay);
      gesture.down({x:event.clientX,y:event.clientY},{...interaction.target});mouseSphere?.aim({x:event.clientX,y:event.clientY},canvas.getBoundingClientRect(),camera,(modelCenter?.z||0)+profile.mouseSphere.depth);
       lastMousePoint={x:event.clientX,y:event.clientY};draggingRef.current?.(true);interaction.grabTarget={...interaction.target};lastDragPoint=performance.now();
      if(event.pointerId!==undefined)try{canvas.setPointerCapture(event.pointerId);}catch{}
      if(grabbed){gesture.press.dragging=true;startGrab({x:event.clientX,y:event.clientY});}
      setInteractive(true);
      feedback();
      event.preventDefault();
    };
    const pointerup = event => {
      if (drag) {
        processPointer({x:event.clientX,y:event.clientY});
         const origin = drag; drag = null;
         window.overlayInputBridge?.drag('avatar',false);
        const clicked=!origin.started,now=performance.now(),slid=slidePx(origin);gesture.up();
        interaction.held=false;interaction.reaction=clicked?'clicked':origin.kind==='spring'?'idle':'settling';interaction.until=now+800;
        interact(clicked?'click':'release',interaction.pointer);
        if(clicked)mouseSphere?.click();mouseSphere?.release();
        if(!clicked&&origin.kind==='body')letGo(origin,{...origin.geometry,...dragPosition(origin,{x:event.clientX,y:event.clientY},slid)},{x:cursorMotion.velocity.x,y:cursorMotion.velocity.y});
        else{if(clicked)hold.click(now/1000);else hold.release(now/1000);geometryRef.current=origin.geometry;draggingRef.current?.(false);pickup?.release();}
        if(event.pointerId!==undefined&&canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);
      }
      samples.clear();hit=pick(event.clientX,event.clientY);setInteractive(hit);feedback();
    };
     const pointercancel = () => {
       window.overlayInputBridge?.drag('avatar',false);
      const origin=drag;drag=null;gesture.up();mouseSphere?.release();
      if(origin?.started){interact('release');interaction.held=false;interaction.reaction='settling';interaction.until=performance.now()+800;}
      // A cancelled body drag is set down where it was (onto the floor if her feet are under it); anything else just ends.
      if(origin?.started&&origin.kind==='body')letGo(origin,origin.geometry,{x:0,y:0});
      else{if(origin)hold.release(performance.now()/1000);pickup?.release();draggingRef.current?.(false);}
      hit = false;samples.clear(); setInteractive(false);feedback();
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
      // Resize about the cursor; the scruff slide is kept in NDC, so the drag's base excludes it at the new size. A resize is not motion:
      // both trackers shift with it, so it never swings her, kicks her springs or reads as a fast carry.
      const before=geometryRef.current,slidBefore=drag.slid||{x:0,y:0};
      drag = resizeDrag(drag,event.deltaY,{x:event.clientX,y:event.clientY},{width:innerWidth,height:innerHeight},slideOf(drag));
      const geometry=drag.geometry;inertia.shift(geometry.x-before.x,geometry.y-before.y);cursorMotion.shift(geometry.x-drag.slid.x-(before.x-slidBefore.x),geometry.y-drag.slid.y-(before.y-slidBefore.y));
      geometryRef.current=geometry;positionRef.current?.(geometry.x, geometry.y, false, geometry);
    };
    canvas.addEventListener('wheel', wheel, {passive:false});

    // One frame; returns how long until the next (avatar_frame_rate.mjs): every display frame while she moves, about 30 fps standing
    // still, nothing while hidden or settled asleep, and nothing before the model has loaded.
    const animate = () => {
      const delta = Math.min(clock.getDelta(), 0.1);
      const elapsed = clock.elapsedTime,now=performance.now();
      const point=samples.take(performance.now(),!!drag);if(point)processPointer(point);
      if(drag&&!cursorPending&&window.avatarCursorBridge){cursorPending=true;const currentDrag=drag;window.avatarCursorBridge.position().then(point=>{if(!disposed&&drag===currentDrag){if(point.buttons===0)pointerup({clientX:point.x,clientY:point.y});else samples.queue({clientX:point.x,clientY:point.y});}}).catch(()=>{}).finally(()=>{cursorPending=false;});}
      const current = emotionRef.current || {};
      const nextProfile=preferencesRef.current.avatarStudioProfiles?.[catalog?.key];
      if(nextProfile!==profileSource){profileSource=nextProfile;profile=normalizeProfile(nextProfile);holdSettings={...profile.reactions,recoverySeconds:profile.pickup.recoverySeconds,landingSquash:profile.pickup.landingSquash};}
      if (vrm) {
        eyeGaze?.restore();mouseSphere?.restore();pickup?.restore();secondary.restore();
        if(!drag&&(!interaction.reaction||interaction.reaction==='idle'))hovered(hover.update(interaction.target,performance.now(),profile.input.hoverDelay),interaction.pointer);
        const filter=preferencesRef.current.avatarGraphics?.anisotropy??8;if(filter!==lastFilter){lastFilter=filter;applyTextureFiltering(vrm.scene,renderer,{anisotropy:filter});}
        // A hit rule lasts its own window (0.8 s after a click or release, or while held), never the whole hold reaction that follows.
        if(!interaction.held&&performance.now()>=interaction.until)interaction.rule=null;
        const custom=(interaction.held||performance.now()<interaction.until)?interaction.rule:null;

        // Small procedural idle motion keeps the model alive without tracking.
        vrm.scene.rotation.y = (vrm.meta?.metaVersion === '0' ? Math.PI : 0) + Math.sin(elapsed * 0.7) * 0.025;
        vrm.scene.position.x = Math.sin(elapsed * 0.45) * 0.006;
        if(performance.now()>interaction.until)interaction.reaction='';
        if(interaction.held&&performance.now()-lastHoldPost>500){lastHoldPost=performance.now();interact('drag',interaction.pointer);}
        const walk=profile.tPose?null:walker.step(actionsRef.current.find(a=>a.kind==='motion.locomotion'&&a.status==='running'),geometryRef.current,{width:innerWidth,height:innerHeight},delta,interaction.held||!!drag?.started||release.active);
        if(walk){positionRef.current?.(walk.x,walk.y,walk.final,walk);vrm.scene.rotation.y+=(walk.facing<0?.35:-.35);}
        // Frame order: window motion (a throw's flight or a set-down's glide, the scruff slide, then both trackers), the hold, MotionEngine,
        // the carry rig on the normalized bones (with the head look), humanoid.update, the anchor translation, springs with the window's
        // inertial force, the face, then the eyes.
        if(release.active){
         release.step(now/1000,delta);
         // She moves under a cursor that may not: keep the hit test current so a press can catch her.
         if(release.flying&&lastMousePoint){hit=pick(lastMousePoint.x,lastMousePoint.y);setInteractive(hit);feedback();}
        }
        if(drag?.started&&drag.kind==='body'){const s=slidePx(drag);if(s.x!==drag.slid?.x||s.y!==drag.slid?.y)placeDrag();}
        inertia.track(geometryRef.current.x,geometryRef.current.y,delta);
        const slid=drag?.slid||{x:0,y:0};cursorMotion.track(geometryRef.current.x-slid.x,geometryRef.current.y-slid.y,delta);
        const worldPerPixel=2*Math.tan(camera.fov*Math.PI/360)*camera.position.distanceTo(modelCenter||camera.position)/Math.max(1,canvas.clientHeight);lastWorldPerPixel=worldPerPixel;
        const held=hold.step(now/1000,delta,{vx:cursorMotion.velocity.x,vy:cursorMotion.velocity.y,ax:cursorMotion.acceleration.x,ay:cursorMotion.acceleration.y,ragdoll:!!pickup?.ragdoll},holdSettings);
        interaction.hold=held?.pose?held:interaction.held&&!profile.reactions.enabled?passiveHold(!!pickup?.ragdoll):null;
        // While a hold reaction runs it is the reaction: hover events and backend base overlays stay suppressed until it ends.
        if(held?.pose)interaction.reaction=held.state;else if(HELD_STATES.has(interaction.reaction)||DROP_STATES.has(interaction.reaction))interaction.reaction='';
        holdFace.apply(held);
        // Where the reaction looks: a world point for the eyes and the carry rig's head (the viewer, the holder's hand, the floor), the
        // cursor even when it is not near her, 'ahead' (dizzy: the eyes rest), or nothing: normal tracking (gazeActive).
        const look=held?.state?held.look:null;
        const lookAt=look==='viewer'?lookPoint.copy(camera.position):look==='holder'?lookPoint.copy(pickup?.active&&pickup.anchorPoint?pickup.anchorPoint:headPoint()).add(HOLDER):look==='floor'?lookPoint.copy(headPoint()).add(FLOOR):null;
        const headLook=lookAt||(look==='cursor'&&eyeGaze?.tracking?eyeGaze.target:null);
        if(profile.tPose)secondary.preview();
        else motion?.update({actions:actionsRef.current,emotion:current,interaction,walking:!!walker.active,walkSpeed:walker.velocity/(animationRef.current.settings?.walk_speed||240),settings:{...animationRef.current.settings,enabled:animationRef.current.enabled},held:{pitch:profile.pickup.scruffPitch,kickAmplitude:profile.reactions.kickAmplitude}},delta);
        if(!profile.tPose)pickup?.pose(profile.pickup,delta,{ax:inertia.acceleration.x,ay:inertia.acceleration.y,vx:inertia.velocity.x,vy:inertia.velocity.y,worldPerPixel,hold:interaction.hold,look:headLook,eyeReach:eyeGaze?.reach});
        // Same order as VRM.update, with non-destructive calibration before springs
        // and rotational secondary motion after them. Never simulate springs twice.
        vrm.humanoid.update();secondary.calibrate(profile);
        vrm.nodeConstraintManager?.update();
        if(profile.tPose!==previousTPose){vrm.scene.updateMatrixWorld(true);vrm.springBoneManager?.reset();previousTPose=profile.tPose;}
        if(!profile.tPose)pickup?.anchor(profile.pickup);
        // Dragging or walking moves the window, not the skeleton: its motion reaches the springs as an inertial force.
        if(!profile.tPose&&vrm.springBoneManager)withSpringForce(vrm.springBoneManager.joints,inertia.force(worldPerPixel,profile.pickup.bounce,delta,springForce),()=>vrm.springBoneManager.update(delta));
        for(const material of vrm.materials||[])material.update?.(delta);
        secondary.update(profile,delta);
        // World matrices: the sphere reads every spring endpoint, so it gets a full pass while the pointer is over her or a spring is
        // dragged (aim needs only the camera); the eyes need only the head's chain; render() makes the frame's one full pass.
        if(!profile.tPose){if(lastMousePoint)mouseSphere.aim(lastMousePoint,canvas.getBoundingClientRect(),camera,(modelCenter?.z||0)+profile.mouseSphere.depth);if(mouseSphere.active||mouseSphere.drag)vrm.scene.updateMatrixWorld(true);mouseSphere?.update(profile,delta);}
        // The face writes last: base emotion (or a resting smile) crossfaded against the reaction or a hit rule's expression, then blinks; the mouth breathes with the body.
        if(face)for(const [name,value] of face.step(delta,{emotion:emotionRef.current,rule:custom?.expression&&custom.expression!=='default'?custom:null,assets:profile.tPose?null:motion?.expressionTargets,breathPhase:held?.breath.phase,settings:profile.face}))vrm.expressionManager.setValue(name,value);
        vrm.expressionManager?.update();
        headNode?.updateWorldMatrix(true,false);
        eyeGaze?.update({...profile.gaze,enabled:profile.gaze.enabled&&!profile.tPose&&animationRef.current.settings?.mouse_tracking!==false},delta,lastMousePoint,canvas.getBoundingClientRect(),camera,mouseSphere,gazeActive(look,interaction.pointer?.near),lookAt);
        if(eyeGaze?.bigShift)face?.gazeShift();  // a big gaze shift comes with a blink (never two within 2 s, never while her eyes are held wide)
        vrm.expressionManager?.update();
        if(preferencesRef.current.avatarHitOutlines){vrm.scene.updateMatrixWorld(true);pickBVH?.refit(profile.springPickRadius);}  // a debug view
        skeletonDebug.update(profile);hitOutlines?.update(preferencesRef.current,interaction.target);
      }
      const activeRule=(interaction.held||performance.now()<interaction.until)?interaction.rule:null;
      const visualProfile=activeRule?.effect?{...profile,effect:{...profile.effect,preset:activeRule.effect}}:profile;
      effects.update(visualProfile,elapsed);effects.render(scene,camera,visualProfile);
      if(!vrm)return Infinity;
      const moving=avatarActive({dragging:!!drag||!!mouseSphere?.drag,released:release.active||!!pickup?.cameraActive,walking:!!walker.active,
        animating:motionBusy(actionsRef.current,motion?.mode,motion?.blending)||!!profile.effect.glslEnabled,reacting:interaction.held||!!interaction.hold||!!interaction.reaction&&interaction.reaction!=='idle',now});
      if(moving)movedAt=now;
      const active=moving||avatarActive({pointerAt,movedAt,now});  // and for a moment after: the pointer's 1.5 s, her springs' 1 s
      if(active)quietSince=now;
      return frameInterval({mode:preferencesRef.current.avatarGraphics?.frameRate==='full'?'full':'adaptive',active,hidden:document.hidden,asleep:asleepRef.current,quietFor:now-quietSince,settled:!asleepRef.current||!face||face.settled});
    };
    const loop=frameLoop(animate);
    // A wake from outside the loop (a change in props, a resize, the window shown, the model loaded) draws now and starts a fresh
    // settle window, so asleep she draws at the idle rate until the change has played out, not one frame of it.
    function rouse(){quietSince=performance.now();loop.wake();}
    wakeRef.current=rouse;
    const visibility=()=>rouse();document.addEventListener('visibilitychange',visibility);
    // A new backend (a restart while Electron runs) starts without the model's bones and expressions: report them again on reconnect,
    // or library animations and previews stop until the model reloads.
    function reportCapabilities(){if(capabilities&&!disposed)request('/api/animation/capabilities',{method:'POST',body:capabilities}).catch(()=>{});}
    const disconnectEvents=connectEvents(()=>{},connected=>{if(connected)reportCapabilities();});
    loop.start();
    return () => {
       release.dispose();disposed = true;
       nativeOff?.();window.overlayInputBridge?.drag('avatar',false);
      loop.stop();wakeRef.current=null;document.removeEventListener('visibilitychange',visibility);disconnectEvents();
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
  useEffect(()=>{wakeRef.current?.();},[emotion,actions,animation,preferences,asleep]);

  return <><canvas key={String(graphics.antialias)} ref={canvasRef} className="avatar-canvas" style={{inset:'auto',left:geometry.x,top:geometry.y,width:geometry.width,height:geometry.height,pointerEvents:'auto'}} aria-label="Character avatar" />
    {shaderError&&<div className="avatar-grab-feedback" role="alert" style={{left:geometry.x+geometry.width/2,top:Math.max(8,geometry.y+40)}}>{shaderError}</div>}
    {editorOpen&&pointerState!=='idle'&&<div className="avatar-grab-feedback" role="status" style={{left:geometry.x+geometry.width/2,top:Math.max(8,geometry.y+8)}}>{pointerState.startsWith('spring:')?'Spring endpoint · drag to pull':pointerState.startsWith('held:')?'Holding · drag to move · scroll to resize':'Drag to pick up'} · {pointerState.split(':')[1]}</div>}</>;
}
