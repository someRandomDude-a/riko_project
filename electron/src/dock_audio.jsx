import React,{useEffect,useId,useRef,useState} from 'react';
import {Volume2,VolumeX} from './ui/icons.jsx';
import {request} from './api.mjs';
import {audioPopover,volumeWriter} from './dock_audio.mjs';

export function DockVolume({id,gain,disabled,onChange}){
 return <div id={id} className="dock-volume-panel" role="group" aria-label="Audio output volume"><input id={id+'-slider'} aria-label="Audio output volume" aria-orientation="horizontal" type="range" min="0" max="1" step=".01" value={gain} disabled={disabled} onChange={e=>onChange(Number(e.target.value))}/><output htmlFor={id+'-slider'}>{Math.round(gain*100)}%</output></div>;
}

export default function DockAudio({muted,volume=1,disabled,onToggle,onError}){
 const root=useRef(null),writer=useRef(null),id=useId();
 const [panel,setPanel]=useState({open:false,armed:false}),panelRef=useRef(panel);panelRef.current=panel;
 const [gain,setGain]=useState(volume),errorRef=useRef(onError);errorRef.current=onError;
 const confirmed=useRef(volume);
 useEffect(()=>{writer.current=volumeWriter(async value=>{const response=await request('/api/audio/volume',{method:'PATCH',body:{volume:value}});confirmed.current=response.volume;},error=>errorRef.current?.(error.message),()=>setGain(confirmed.current));return()=>{writer.current?.close();};},[]);
 useEffect(()=>{confirmed.current=volume;if(!writer.current?.pending)setGain(volume);},[volume]);
 function event(kind){const next=audioPopover(panelRef.current,kind),state={open:next.open,armed:next.armed};panelRef.current=state;setPanel(state);if(next.toggle)onToggle();}
 useEffect(()=>{const outside=e=>{if(!root.current?.contains(e.target))event('close');};const escape=e=>{if(e.key==='Escape'&&panelRef.current.open){root.current?.querySelector('button')?.focus();event('close');}};document.addEventListener('pointerdown',outside);document.addEventListener('keydown',escape);return()=>{document.removeEventListener('pointerdown',outside);document.removeEventListener('keydown',escape);};},[]);
 return <div className="dock-audio" ref={root} onPointerEnter={()=>event('reveal')} onPointerLeave={()=>{if(!panelRef.current.armed&&!root.current?.contains(document.activeElement))event('close');}} onFocus={()=>event('reveal')} onBlur={e=>{if(!e.currentTarget.contains(e.relatedTarget))event('close');}}>
   <button type="button" className="dock-side" aria-label={muted?'Unmute audio output':'Mute audio output'} aria-pressed={muted} aria-expanded={panel.open} aria-controls={id} title="Audio output" disabled={disabled} onClick={()=>event('click')}>{muted?<VolumeX size={19}/>:<Volume2 size={19}/>}</button>
   {panel.open&&<DockVolume id={id} gain={gain} disabled={disabled} onChange={value=>{setGain(value);writer.current?.set(value);}}/>}
 </div>;
}
