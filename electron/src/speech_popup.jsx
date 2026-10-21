import React,{useEffect,useLayoutEffect,useRef,useState} from 'react';
import FormattedText from './formatted_text.jsx';
import {bubblePlacement,clampBubble,dockPopupPlacement} from './feedback_model.mjs';
import {beginPopupDrag,movePopupDrag} from './popup_drag.mjs';
import {popupInteractive} from './interactive_bridge.mjs';
export function usePopupVisibility(text,active,seconds){
 const [visible,setVisible]=useState(false);
 useEffect(()=>{setVisible(!!text);if(!text||active)return;const timer=setTimeout(()=>setVisible(false),seconds*1000);return()=>clearTimeout(timer);},[text,active,seconds]);
 return visible;
}
function SpeechPopup({prefix,text,visible,preferences:p,geometry,surface,update,inline=false,sending=false}){
 const root=useRef(null),content=useRef(null),drag=useRef(null),cached=useRef(''),poll=useRef({frame:0,pending:false});
 const [mounted,setMounted]=useState(visible),[size,setSize]=useState({width:0,height:0}),[position,setPosition]=useState(null);
 const [viewport,setViewport]=useState({width:window.innerWidth,height:window.innerHeight});
 if(text)cached.current=text;
 useEffect(()=>{if(visible){setMounted(true);return;}if(inline){setMounted(false);return;}const timer=setTimeout(()=>setMounted(false),260);return()=>clearTimeout(timer);},[visible,inline]);
 useLayoutEffect(()=>{if(!mounted||!content.current)return;const observer=new ResizeObserver(entries=>{const r=entries[0].contentRect,width=r.width+36,height=inline?r.height+28:Math.min(window.innerHeight*.4,r.height+28);setSize(old=>old.width===width&&old.height===height?old:{width,height});});observer.observe(content.current);const resize=()=>setViewport({width:window.innerWidth,height:window.innerHeight});window.addEventListener('resize',resize);return()=>{observer.disconnect();window.removeEventListener('resize',resize);};},[mounted,inline]);
  useEffect(()=>setPosition(null),[p[prefix+'X'],p[prefix+'Y'],p[prefix+'Anchor']]);
  const latest=useRef(null);latest.current={size,viewport,update,prefix};
  function end(){const d=drag.current;if(!d)return;drag.current=null;const node=root.current;if(node){delete node.dataset.dragging;if(d.pointerId!==undefined&&node.hasPointerCapture?.(d.pointerId))try{node.releasePointerCapture(d.pointerId);}catch{}}
   window.overlayInputBridge?.drag('popup',false);popupInteractive(false);
   const {viewport,update,prefix}=latest.current;if(d.position)update?.({[prefix+'Anchor']:'screen',[prefix+'X']:d.position.left/viewport.width*100,[prefix+'Y']:d.position.top/viewport.height*100});
  }
  function move(point){const d=drag.current;if(!d)return;const {size,viewport}=latest.current;setPosition({...movePopupDrag(d,point,size,viewport)});}
  // Windows: the cursor's position comes from main once per frame while a drag runs (overlayInputBridge.native); no frame loop runs
  // between drags.
  function follow(){
   const p=poll.current;cancelAnimationFrame(p.frame);
   const tick=()=>{if(!drag.current){p.frame=0;return;}p.frame=requestAnimationFrame(tick);if(p.pending)return;p.pending=true;window.overlayInputBridge.position().then(point=>{if(drag.current){move(point);if(point.buttons===0)end();}}).catch(()=>{}).finally(()=>{p.pending=false;});};
   p.frame=requestAnimationFrame(tick);
  }
  useEffect(()=>{
   const mousemove=e=>{if(drag.current)move({x:e.clientX,y:e.clientY});},up=()=>end();
   window.addEventListener('mousemove',mousemove);window.addEventListener('mouseup',up);
   const off=!inline&&window.overlayInputBridge?.subscribe(point=>{if(!drag.current)return;move(point);if(point.phase==='up')end();});
   return()=>{cancelAnimationFrame(poll.current.frame);poll.current.frame=0;off?.();window.removeEventListener('mousemove',mousemove);window.removeEventListener('mouseup',up);if(drag.current){drag.current=null;window.overlayInputBridge?.drag('popup',false);}};
  },[inline]);
 if(!mounted)return null;
 let placement=bubblePlacement(p,prefix,geometry||{x:0,y:0,width:viewport.width,height:viewport.height});
 if(p[prefix+'Anchor']==='dock'&&surface?.visible){
  const b=surface.bounds,s=surface.screen;
  placement=dockPopupPlacement(prefix,b,s,size,viewport);
 }else if(p[prefix+'Anchor']==='dock')placement={left:'50vw',top:prefix==='reply'?'20vh':'92vh'};
 const width=Math.min(p[prefix+'Width'],Math.max(180,180+Math.sqrt(cached.current.length)*20));
  function start(e){if(inline||e.button!==0||drag.current||e.target.closest?.('a,button,input,select,textarea'))return;const pos=clampBubble(position||placement,size,viewport);drag.current={...beginPopupDrag({x:e.clientX,y:e.clientY},pos),pointerId:e.pointerId};root.current.dataset.dragging='true';window.overlayInputBridge?.drag('popup',true);popupInteractive(true);if(window.overlayInputBridge?.native)follow();if(e.pointerId!==undefined)try{root.current.setPointerCapture(e.pointerId);}catch{}e.preventDefault();}
  return <div ref={root} role="status" aria-label={prefix==='reply'?'Assistant response':'Your transcript'} className={'desktop-speech-cloud speech-popup '+prefix+(visible?' popup-visible':' popup-leaving')+(inline?' popup-inline':'')+(sending?' popup-sending':'')}
 style={{...(inline?{}:clampBubble(position||placement,size,viewport)),width:inline?'100%':width,height:size.height||undefined,opacity:p[prefix+'Opacity'],'--popup-color':p.popupTheme?'light-dark(#f1f3f7,#182130)':p.popupColor,'--popup-blur':p.popupBlur+'px'}}
 onPointerEnter={()=>!inline&&popupInteractive(true)} onPointerLeave={()=>{if(!drag.current&&!inline)popupInteractive(false);}}
  onPointerDown={start} onMouseDown={start}
  onPointerMove={e=>{if(!drag.current)return;if(e.buttons===0&&!window.overlayInputBridge?.native){end();return;}move({x:e.clientX,y:e.clientY});}}
  onPointerUp={end} onPointerCancel={end} onLostPointerCapture={()=>{if(!window.overlayInputBridge?.native)end();}}><div ref={content}><FormattedText text={cached.current}/></div></div>;
}
// Memoized: its parent's updates (a streamed token, a snapshot) render it only when its own props changed.
export default React.memo(SpeechPopup);
