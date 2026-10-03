import React, {useEffect, useRef} from 'react';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin, VRMUtils} from '@pixiv/three-vrm';
import {MotionAssetSampler} from './motion_assets.mjs';
import {MotionEngine, BODY_BONES} from './motion_engine.mjs';
import {DesktopWalker, nearbyPointer} from './motion_geometry.mjs';
import {cameraDistance, resizeHeldAvatar} from './avatar_geometry.mjs';
import {API, request} from './api.mjs';
import {avatarModelURL,validateAvatarFormat} from './avatar_model.mjs';

const EXPRESSION_NAMES = ['happy', 'sad', 'angry', 'surprised', 'neutral'];
const EMOTION_MAP = {joy: 'happy', love: 'happy', affection: 'happy', sadness: 'sad', anger: 'angry', surprise: 'surprised', calm: 'neutral', neutral: 'neutral'};

function clamp(value, min = 0, max = 1) { return Math.max(min, Math.min(max, Number(value) || 0)); }

/** Native VRM presentation layer. The runtime supplies intent; this loop owns animation frames. */
export default function AvatarRenderer({emotion, modelPath, modelFormat='auto', fov=30, actions = [], animation={}, geometry = {x:0,y:0,width:480,height:720}, onPosition,onError}) {
  const canvasRef = useRef(null);
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
    const renderer = new THREE.WebGLRenderer({canvas, alpha: true, antialias: true, powerPreference:'high-performance'});
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(fov, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x443355, 2));

    const loader = new GLTFLoader();
    loader.register(parser => new VRMLoaderPlugin(parser));
    const report=(id,status,error='')=>{request('/api/avatar/animation/result',{method:'POST',body:{action_id:id,status,error}}).catch(()=>{});};
    const walker=new DesktopWalker(report);
    let motion=null;
    const interaction={held:false,pointer:null,reaction:'',until:0};
    let lastPointerPost=0,lastHoldPost=0;
    function interact(kind,pointer=null){request('/api/animation/interaction',{method:'POST',body:{kind,pointer}}).catch(()=>{});}
    let vrm = null;
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
      const bounds = new THREE.Box3().setFromObject(vrm.scene);
      const size = bounds.getSize(new THREE.Vector3());
      vrm.scene.position.y -= bounds.min.y;
      vrm.scene.updateMatrixWorld(true);
      modelSize = size;
      modelCenter = new THREE.Box3().setFromObject(vrm.scene).getCenter(new THREE.Vector3());
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
    const raycaster = new THREE.Raycaster(); let drag = null, hit = false, interactive = false;
    function setInteractive(value) {
      if (value === interactive) return;
      interactive = value;
      window.riko?.avatarInteractive(value);
    }
    const mousemove = event => {
      interaction.pointer=nearbyPointer(geometryRef.current,{x:event.clientX,y:event.clientY},interaction.pointer?.near);
      if(performance.now()-lastPointerPost>150){lastPointerPost=performance.now();interact('pointer',interaction.pointer);}
      if (drag) {
        drag.geometry = {...drag.geometry, x: drag.left + event.clientX-drag.x, y: drag.top + event.clientY-drag.y};
        positionRef.current?.(drag.geometry.x, drag.geometry.y, false, drag.geometry);
      }
      const rect = canvas.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1, 1-(event.clientY-rect.top)/rect.height*2), camera);
      hit = !!vrm && event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom && raycaster.intersectObject(vrm.scene,true).length > 0;
      setInteractive(hit || !!drag || !!event.target.closest?.('button,.bubble'));
    };
    const pointerdown = event => {
      if (event.button !== 0 || !hit) return;
      drag = {x: event.clientX, y: event.clientY, left: geometryRef.current.x, top: geometryRef.current.y, geometry: {...geometryRef.current}};
      interaction.held=true;interaction.reaction='';walker.stop();interact('hold',interaction.pointer);
      canvas.setPointerCapture(event.pointerId);
      setInteractive(true);
      event.preventDefault();
    };
    const pointerup = event => {
      if (drag) {
        const origin = drag; drag = null;
        const clicked=Math.hypot(event.clientX-origin.x,event.clientY-origin.y)<4;
        interaction.held=false;interaction.reaction=clicked?'clicked':'settling';interaction.until=performance.now()+800;
        interact(clicked?'click':'release',interaction.pointer);
        positionRef.current?.(origin.left + event.clientX - origin.x, origin.top + event.clientY - origin.y, true, origin.geometry);
      }
      mousemove(event);
    };
    const pointercancel = () => {
      if (drag){positionRef.current?.(drag.geometry.x, drag.geometry.y, true, drag.geometry);interact('release');interaction.held=false;interaction.reaction='settling';interaction.until=performance.now()+800;}
      drag = null; hit = false; setInteractive(false);
    };
    const leave=()=>{if(!drag){interaction.pointer=null;interact('leave');setInteractive(false);}};
    addEventListener('blur',pointercancel);addEventListener('mouseleave',leave);
    // Electron's click-through forwarding emits mousemove, not pointermove.
    // Use it for hover; captured pointermove continues dragging off the mesh.
    addEventListener('mousemove', mousemove);
    canvas.addEventListener('pointerdown', pointerdown);
    canvas.addEventListener('pointermove', mousemove);
    canvas.addEventListener('pointerup', pointerup);
    canvas.addEventListener('pointercancel', pointercancel);
    canvas.addEventListener('lostpointercapture', pointercancel);
    const wheel = event => {
      if (!drag) return;
      event.preventDefault();
      const geometry = resizeHeldAvatar(drag.geometry, event.deltaY, {x:event.clientX,y:event.clientY}, {width:innerWidth,height:innerHeight});
      drag = {x:event.clientX,y:event.clientY,left:geometry.x,top:geometry.y,geometry};
      positionRef.current?.(geometry.x, geometry.y, false, geometry);
    };
    canvas.addEventListener('wheel', wheel, {passive:false});

    const animate = () => {
      frame = requestAnimationFrame(animate);
      const delta = Math.min(clock.getDelta(), 0.1);
      const elapsed = clock.elapsedTime;
      const current = emotionRef.current || {};
      if (vrm) {
        const expression = vrm.expressionManager;
        const active = EMOTION_MAP[current.primary] || 'neutral';
        for (const name of EXPRESSION_NAMES) {
          targetExpressions[name] = name === active ? clamp(current.intensity ?? 0.65) : 0;
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
        const walk=walker.step(actionsRef.current.find(a=>a.kind==='motion.locomotion'&&a.status==='running'),geometryRef.current,{width:innerWidth,height:innerHeight},delta,interaction.held);
        if(walk){positionRef.current?.(walk.x,walk.y,walk.final,walk);vrm.scene.rotation.y+=(walk.facing<0?.35:-.35);}
        motion?.update({actions:actionsRef.current,emotion:current,interaction,walking:!!walker.active,walkSpeed:walker.velocity/(animationRef.current.settings?.walk_speed||240),settings:{...animationRef.current.settings,enabled:animationRef.current.enabled}},delta);
        vrm.update(delta);
      }
      renderer.render(scene, camera);
    };
    animate();
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      removeEventListener('resize', resize);
      observer.disconnect(); removeEventListener('mousemove', mousemove);
      removeEventListener('blur',pointercancel);removeEventListener('mouseleave',leave);
      canvas.removeEventListener('pointerdown', pointerdown);
      canvas.removeEventListener('pointermove', mousemove);
      canvas.removeEventListener('pointerup', pointerup);
      canvas.removeEventListener('pointercancel', pointercancel);
      canvas.removeEventListener('lostpointercapture', pointercancel);
      canvas.removeEventListener('wheel', wheel);
      window.riko?.avatarInteractive(false);
      renderer.dispose();
      walker.stop();motion?.dispose();interact('leave');
      VRMUtils.deepDispose(scene);
    };
  }, [modelPath,modelFormat,fov]);

  return <canvas ref={canvasRef} className="avatar-canvas" style={{inset:'auto',left:geometry.x,top:geometry.y,width:geometry.width,height:geometry.height,pointerEvents:'auto'}} aria-label="Character avatar" />;
}
