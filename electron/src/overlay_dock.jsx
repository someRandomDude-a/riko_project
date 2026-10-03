import React,{useEffect,useRef,useState} from 'react';
import DockControls from './dock_controls.jsx';

export function dockPosition(x,y,width,height){return {x:Math.max(8,Math.min(width-112,x)),y:Math.max(8,Math.min(height-112,y))};}
export default function OverlayDock({preferences}){
  const drag=useRef(null);
  const [visible,setVisible]=useState(false);
  const [position,setPosition]=useState(()=>{try{const p=JSON.parse(localStorage.getItem('riko:dock')||'null');if(Number.isFinite(p?.x)&&Number.isFinite(p?.y))return dockPosition(p.x,p.y,innerWidth,innerHeight);}catch{}return dockPosition(innerWidth-224,innerHeight-104,innerWidth,innerHeight);});
  useEffect(()=>{
    let alive=true;window.controlVisibilityBridge?.get().then(value=>{if(alive)setVisible(!value);}).catch(()=>{});
    const off=window.controlVisibilityBridge?.subscribe(value=>setVisible(!value));
    const resize=()=>setPosition(p=>dockPosition(p.x,p.y,innerWidth,innerHeight));window.addEventListener('resize',resize);
    return()=>{alive=false;off?.();window.removeEventListener('resize',resize);window.dockBridge?.interactive(false);};
  },[]);
  useEffect(()=>{if(!visible){drag.current=null;window.dockBridge?.interactive(false);}},[visible]);
  function start(event){if(event.button!==0)return;event.currentTarget.setPointerCapture(event.pointerId);drag.current={x:event.clientX,y:event.clientY,start:position,moved:false};}
  function move(event){const d=drag.current;if(!d)return;const dx=event.clientX-d.x,dy=event.clientY-d.y;if(Math.hypot(dx,dy)>5)d.moved=true;if(d.moved)setPosition(dockPosition(d.start.x+dx,d.start.y+dy,innerWidth,innerHeight));}
  function finish(event){const d=drag.current;if(!d)return;drag.current=null;if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId);if(d.moved){try{localStorage.setItem('riko:dock',JSON.stringify(position));}catch{}}else window.dockBridge?.open(position);const r=event.currentTarget.parentElement.parentElement.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)window.dockBridge?.interactive(false);}
  if(!visible)return null;
  return <section className="overlay-dock" aria-label="Microphone dock" style={{left:position.x,top:position.y}} onPointerEnter={()=>window.dockBridge?.interactive(true)} onPointerLeave={()=>{if(!drag.current)window.dockBridge?.interactive(false);}}>
    <DockControls collapsed preferences={preferences} onExpand={()=>{}} handleProps={{onPointerDown:start,onPointerMove:move,onPointerUp:finish,onPointerCancel:()=>{drag.current=null;window.dockBridge?.interactive(false);},onClick:event=>{if(event.detail===0)window.dockBridge?.open(position);}}}/>
  </section>;
}
