import {useEffect,useRef} from 'react';
export default function useWindowGesture(kind,onClick){
 const active=useRef(null),moved=useRef(false);
 function end(cancelled=false){
   const g=active.current;if(!g)return;
   active.current=null;
   delete g.element.dataset.dragging;
   window.gestureBridge?.send({phase:'end',id:g.id});
   if(g.element.hasPointerCapture(g.pointer))g.element.releasePointerCapture(g.pointer);
   if(cancelled)moved.current=true;
 }
 useEffect(()=>{
   const up=()=>end(),cancel=()=>end(true),move=e=>{if(active.current&&!(e.buttons&1))cancel();};
   window.addEventListener('pointerup',up,true);window.addEventListener('pointercancel',cancel,true);window.addEventListener('blur',cancel);window.addEventListener('pointermove',move,true);
   return()=>{cancel();window.removeEventListener('pointerup',up,true);window.removeEventListener('pointercancel',cancel,true);window.removeEventListener('blur',cancel);window.removeEventListener('pointermove',move,true);};
 },[]);
 return {
  onPointerDown:e=>{if(e.button!==0)return;end(true);const id=crypto.randomUUID();active.current={id,element:e.currentTarget,pointer:e.pointerId,x:e.screenX,y:e.screenY};moved.current=false;e.currentTarget.dataset.dragging='true';e.currentTarget.setPointerCapture(e.pointerId);window.gestureBridge?.send({phase:'begin',id,kind});},
  onPointerMove:e=>{const g=active.current;if(!g||!(e.buttons&1))return;if(Math.hypot(e.screenX-g.x,e.screenY-g.y)>4)moved.current=true;},
  onPointerUp:()=>end(),onPointerCancel:()=>end(true),onLostPointerCapture:()=>end(true),
  onClick:()=>{if(!moved.current)onClick?.();moved.current=false;},
 };
}
