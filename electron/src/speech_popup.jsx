import React,{useEffect,useLayoutEffect,useRef,useState} from 'react';
import FormattedText from './formatted_text.jsx';
import {bubblePlacement,clampBubble,dockPopupPlacement} from './feedback_model.mjs';
export function usePopupVisibility(text,active,seconds){
 const [visible,setVisible]=useState(false);
 useEffect(()=>{setVisible(!!text);if(!text||active)return;const timer=setTimeout(()=>setVisible(false),seconds*1000);return()=>clearTimeout(timer);},[text,active,seconds]);
 return visible;
}
export default function SpeechPopup({prefix,text,visible,preferences:p,geometry,surface,update,inline=false,sending=false}){
 const root=useRef(null),content=useRef(null),drag=useRef(null),cached=useRef('');
 const [mounted,setMounted]=useState(visible),[size,setSize]=useState({width:0,height:0}),[position,setPosition]=useState(null);
 const [viewport,setViewport]=useState({width:window.innerWidth,height:window.innerHeight});
 if(text)cached.current=text;
 useEffect(()=>{if(visible){setMounted(true);return;}if(inline){setMounted(false);return;}const timer=setTimeout(()=>setMounted(false),260);return()=>clearTimeout(timer);},[visible,inline]);
 useLayoutEffect(()=>{if(!mounted||!content.current)return;const observer=new ResizeObserver(entries=>{const r=entries[0].contentRect;setSize({width:r.width+36,height:inline?r.height+28:Math.min(window.innerHeight*.4,r.height+28)});});observer.observe(content.current);const resize=()=>setViewport({width:window.innerWidth,height:window.innerHeight});window.addEventListener('resize',resize);return()=>{observer.disconnect();window.removeEventListener('resize',resize);};},[mounted,inline]);
 useEffect(()=>setPosition(null),[p[prefix+'X'],p[prefix+'Y'],p[prefix+'Anchor']]);
 if(!mounted)return null;
 let placement=bubblePlacement(p,prefix,geometry||{x:0,y:0,width:viewport.width,height:viewport.height});
 if(p[prefix+'Anchor']==='dock'&&surface?.visible){
  const b=surface.bounds,s=surface.screen;
  placement=dockPopupPlacement(prefix,b,s,size,viewport);
 }else if(p[prefix+'Anchor']==='dock')placement={left:'50vw',top:prefix==='reply'?'20vh':'92vh'};
 const width=Math.min(p[prefix+'Width'],Math.max(180,180+Math.sqrt(cached.current.length)*20));
 function end(e){const d=drag.current;if(!d)return;drag.current=null;delete e.currentTarget.dataset.dragging;if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);window.popupBridge?.interactive(false);if(d.position)update?.({[prefix+'Anchor']:'screen',[prefix+'X']:d.position.left/viewport.width*100,[prefix+'Y']:d.position.top/viewport.height*100});}
  return <div ref={root} role="status" aria-label={prefix==='reply'?'Assistant response':'Your transcript'} className={'desktop-speech-cloud speech-popup '+prefix+(visible?' popup-visible':' popup-leaving')+(inline?' popup-inline':'')+(sending?' popup-sending':'')}
 style={{...(inline?{}:clampBubble(position||placement,size,viewport)),width:inline?'100%':width,height:size.height||undefined,opacity:p[prefix+'Opacity'],'--popup-color':p.popupTheme?'light-dark(#f1f3f7,#182130)':p.popupColor,'--popup-blur':p.popupBlur+'px'}}
 onPointerEnter={()=>!inline&&window.popupBridge?.interactive(true)} onPointerLeave={()=>{if(!drag.current&&!inline)window.popupBridge?.interactive(false);}}
 onPointerDown={e=>{if(inline||e.button!==0)return;const pos=clampBubble(position||placement,size,viewport);drag.current={x:e.screenX,y:e.screenY,...pos};e.currentTarget.dataset.dragging='true';e.currentTarget.setPointerCapture(e.pointerId);}}
 onPointerMove={e=>{const d=drag.current;if(!d)return;if(!(e.buttons&1)){end(e);return;}d.position=clampBubble({left:d.left+e.screenX-d.x,top:d.top+e.screenY-d.y},size,viewport);setPosition(d.position);}}
 onPointerUp={end} onPointerCancel={end} onLostPointerCapture={end}><div ref={content}><FormattedText text={cached.current}/></div></div>;
}
