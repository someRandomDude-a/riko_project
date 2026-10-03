import React, {useEffect, useRef, useState} from 'react';
import FormattedText from './formatted_text.jsx';
import {API, reportSurface, mediaURL} from './api.mjs';
import {scheduleBoardCapture} from './whiteboard_capture.mjs';
import {readBoardView, saveBoardView} from './board_view.mjs';
import DashBar from './dash_bar.jsx';

function BoardObject({command, zoom, position, move, measured, acknowledge}) {
  const ref = useRef(null), drag = useRef(null), last = useRef('');
  const measuredRef = useRef(measured);
  measuredRef.current = measured;
  const p = command.payload, b = command.bounds || {x: p.x || 0, y: p.y || 0, width: p.width || 420, height: 120};
  const drawWidth = p.points ? Math.max(1, Math.max(...p.points.map(v=>v[0]))-Math.min(...p.points.map(v=>v[0])))+(p.size||6) : 1;
  const drawHeight = p.points ? Math.max(1, Math.max(...p.points.map(v=>v[1]))-Math.min(...p.points.map(v=>v[1])))+(p.size||6) : 1;
  useEffect(() => {
    const measure = () => {
      const images = [...ref.current.querySelectorAll('img')];
      if (images.some(image => image.complete && !image.naturalWidth)) {if(acknowledge)reportSurface('whiteboard',command.id,'error','Markdown/image asset unavailable');return;}
      if (images.some(image => !image.complete)) return;
      const bounds = {x: b.x, y: b.y, width: Math.max(ref.current.offsetWidth,ref.current.scrollWidth), height: Math.max(ref.current.offsetHeight,ref.current.scrollHeight)};
      const signature = JSON.stringify(bounds);
      if (bounds.width && bounds.height && signature !== last.current) {
        last.current = signature; measuredRef.current(command, bounds);
        if (acknowledge) reportSurface('whiteboard', command.id, 'rendered', '', bounds);
      }
    };
    const element = ref.current;
    // An image may finish without changing its broken/placeholder dimensions.
    // ResizeObserver alone cannot reliably acknowledge success or failure.
    element.addEventListener('load', measure, true);
    element.addEventListener('error', measure, true);
    const observer = new ResizeObserver(measure); observer.observe(element); measure();
    return () => {
      observer.disconnect();
      element.removeEventListener('load', measure, true);
      element.removeEventListener('error', measure, true);
    };
  }, [command.id, b.x, b.y, acknowledge]);
  const points = p.points?.map(([x, y]) => `${x-b.x},${y-b.y}`).join(' ');
  return <div ref={ref} className="board-object" style={{left: position?.x ?? b.x, top: position?.y ?? b.y, width: command.kind === 'draw' ? drawWidth : p.width || 420, padding:command.kind==='draw'?0:8, color: p.color, fontSize: Math.max(12, p.size || 18)}} onPointerDown={event => {
    if (event.target.closest('a')) return;
    event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {x: event.clientX, y: event.clientY, left: position?.x ?? b.x, top: position?.y ?? b.y};
  }} onPointerMove={event => {if (drag.current) move(command.id, {x: drag.current.left + (event.clientX-drag.current.x)/zoom, y: drag.current.top + (event.clientY-drag.current.y)/zoom});}} onPointerUp={() => {drag.current = null;}} onPointerCancel={() => {drag.current = null;}}>
    {command.kind === 'text' ? <FormattedText text={p.text}/> : command.kind === 'image' ? <img draggable="false" src={mediaURL(p.path)} style={{width:'100%'}}/> : <svg width={drawWidth} height={drawHeight} style={{overflow:'visible',display:'block'}}><polyline points={points} fill="none" stroke={p.color} strokeWidth={p.size} strokeLinecap="round"/>{p.points?.length === 1 && <circle cx={p.points[0][0]-b.x} cy={p.points[0][1]-b.y} r={p.size/2} fill={p.color}/>}</svg>}
  </div>;
}

export default function Whiteboard({commands = [], pages = ['page-1'], modelPage = 'page-1', clearCommand, acknowledge = false, loaded = true, persistenceError = '', revision,collapsed=false}) {
  const storageKey = 'riko:whiteboard:' + (acknowledge ? 'surface' : 'workspace');
  const saved = useRef(readBoardView(storageKey));
  const root = useRef(null), pan = useRef(null), dirty = useRef(!!saved.current?.dirty), seen = useRef(saved.current?.seen || null), viewRef = useRef(null), pendingBack = useRef(null);
  const [page, setPage] = useState(saved.current?.page || modelPage), [view, setView] = useState(saved.current?.view || {x:40,y:40,z:1}), [back, setBack] = useState(saved.current?.back || null), [positions, setPositions] = useState(saved.current?.positions || {});
  const hydrated = useRef(false);
  const [options,setOptions]=useState(false),[pagePicker,setPagePicker]=useState(false);
  useEffect(()=>{
    if(collapsed||!acknowledge||!loaded||!revision||!window.whiteboardBridge?.capture)return;
    return scheduleBoardCapture({revision,ready:document.fonts?.ready,
      rect:()=>{const rect=root.current.getBoundingClientRect();return{x:rect.x,y:rect.y,width:rect.width,height:rect.height};},
      capture:rect=>window.whiteboardBridge.capture(rect),
      send:(revision,png)=>fetch(API+'/api/whiteboard/image?revision='+encodeURIComponent(revision),{method:'POST',headers:{'Content-Type':'image/png'},body:new Uint8Array(png)})});
  },[acknowledge,loaded,revision,page,JSON.stringify(view),JSON.stringify(positions),collapsed]);
  const localState = useRef(null);
  localState.current = {page, view, back, positions, seen: seen.current, modelPage, dirty:dirty.current};
  useEffect(() => {
    if(!loaded)return;
    const timer = setTimeout(() => saveBoardView(storageKey, localState.current), 150);
    return () => {clearTimeout(timer); saveBoardView(storageKey, localState.current);};
  }, [page, view, back, positions, loaded]);
  useEffect(() => {
    if (loaded && !hydrated.current) {
      hydrated.current = true;
      if (saved.current?.seen === commands.at(-1)?.id) seen.current = saved.current.seen;
      if (saved.current?.page && pages.includes(saved.current.page)) setPage(saved.current.page);
      else if(!pages.includes(page))setPage(modelPage);
    }
  }, [loaded]);
  viewRef.current = {...view, page};
  const lastModelPage=useRef(saved.current?.modelPage || null);
  useEffect(() => {if(!loaded)return;if(lastModelPage.current!==modelPage){if(lastModelPage.current!==null){if(dirty.current && !pendingBack.current)pendingBack.current=viewRef.current;setPage(modelPage);}else if(!saved.current?.page || !pages.includes(saved.current.page))setPage(modelPage);lastModelPage.current=modelPage;}}, [modelPage,loaded]);
  useEffect(() => {if (clearCommand && acknowledge) reportSurface('whiteboard', clearCommand.id, 'rendered');}, [clearCommand?.id]);
  const newest = commands.at(-1);
  function measured(command, bounds) {
    if (command.id !== newest?.id || seen.current === command.id) return;
    seen.current = command.id;
    if (dirty.current) setBack(pendingBack.current || viewRef.current);
    pendingBack.current = null;
    dirty.current = false;
    setPage(command.page || modelPage);
    setView({x: root.current.clientWidth/2 - bounds.x - bounds.width/2, y: root.current.clientHeight/2 - bounds.y - bounds.height/2, z:1});
  }
  useEffect(() => {if (newest && newest.id !== seen.current) {if(dirty.current && !pendingBack.current)pendingBack.current=viewRef.current;setPage(newest.page || modelPage);}}, [newest?.id]);
  function userView(next) {dirty.current = true; setView(next);}
  return <div className="board-shell">
    {persistenceError&&<div role="alert">{persistenceError}</div>}
    <nav className="board-toolbar board-dock"><div className="board-dock-buttons"><button aria-expanded={options} onClick={()=>collapsed?window.windowBridge?.mode('full'):setOptions(!options)}>Pages</button>{options&&!collapsed&&<><button disabled={pages.indexOf(page)<=0} onClick={()=>{dirty.current=true;setPage(pages[pages.indexOf(page)-1]);}}>Previous</button><button aria-expanded={pagePicker} onClick={()=>setPagePicker(!pagePicker)}>{page} ▦</button><button disabled={pages.indexOf(page)>=pages.length-1} onClick={()=>{dirty.current=true;setPage(pages[pages.indexOf(page)+1]);}}>Next</button><button onClick={()=>{dirty.current=true;setPositions({});setView({x:40,y:40,z:1});setBack(null);pendingBack.current=null;}}>Reset all</button></>}{acknowledge&&!collapsed&&<><button aria-label="Minimize whiteboard" onClick={()=>window.windowBridge?.mode('collapsed')}>−</button><button aria-label="Expand whiteboard" onClick={()=>window.windowBridge?.action('maximize')}>↗</button><button aria-label="Close whiteboard" onClick={()=>window.windowBridge?.mode('collapsed')}>×</button></>}</div>{acknowledge&&<DashBar label={collapsed?'Drag to move; click to open whiteboard':'Drag to move; click to collapse whiteboard'} onClick={()=>window.windowBridge?.mode(collapsed?'full':'collapsed')}/>}</nav>
    {pagePicker&&options&&<section className="board-page-picker" aria-label="Choose a page">{pages.map(p=><button key={p} className={'page-preview '+(page===p?'active':'')} onClick={()=>{dirty.current=true;setPage(p);setPagePicker(false);}}><strong>{p}</strong><div>{commands.filter(c=>(c.page||'page-1')===p).slice(0,4).map(c=><div key={c.id}>{c.kind==='text'?<FormattedText text={c.payload.text.slice(0,350)}/>:c.kind==='image'?<img src={mediaURL(c.payload.path)} alt="Page image"/>:<small>Drawing</small>}</div>)}</div></button>)}</section>}
    <div ref={root} className="board-viewport" onWheel={e=>{const rect=root.current.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,z=Math.max(.2,Math.min(4,view.z*Math.exp(-e.deltaY*.001)));userView({x:x-(x-view.x)*z/view.z,y:y-(y-view.y)*z/view.z,z});}} onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);pan.current={x:e.clientX,y:e.clientY,view};}} onPointerMove={e=>{if(pan.current)userView({...pan.current.view,x:pan.current.view.x+e.clientX-pan.current.x,y:pan.current.view.y+e.clientY-pan.current.y})}} onPointerUp={()=>{pan.current=null}} onPointerCancel={()=>{pan.current=null}}>
      <div className="board-world" style={{transform:`translate(${view.x}px,${view.y}px) scale(${view.z})`}}>{commands.filter(c=>(c.page||'page-1')===page).map(command=><BoardObject key={command.id} command={command} zoom={view.z} position={positions[command.id]} move={(id,position)=>setPositions(old=>({...old,[id]:position}))} measured={measured} acknowledge={acknowledge}/>)}</div>
    </div>
  </div>;
}
