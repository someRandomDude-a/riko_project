import React,{useEffect,useState} from 'react';
import {Minus,Maximize2,Minimize2,X} from './ui/icons.jsx';
import {branding} from './branding.mjs';
export default function WindowChrome({title,children,blur=true}){
  const [maximized,setMaximized]=useState(false),[error,setError]=useState('');
  useEffect(()=>{window.materialBridge?.set(blur).catch(()=>{});},[blur]);
  useEffect(()=>{
    let alive=true;
    window.windowBridge?.state().then(value=>{if(alive)setMaximized(value.maximized);}).catch(()=>{});
    const off=window.windowBridge?.subscribe(value=>{if(typeof value.maximized==='boolean')setMaximized(value.maximized);if(typeof value.transitioning==='boolean')document.documentElement.classList.toggle('surface-transition',value.transitioning);});
    return()=>{alive=false;off?.();};
  },[]);
  async function act(action){try{await window.windowBridge?.action(action);setError('');}catch{setError('Could not change the window. Try again.');}}
  return <><header className="window-chrome" onDoubleClick={event=>{if(!event.target.closest('button,nav'))act('maximize');}}>
    <img className="window-brand" src={branding.logo} alt=""/><span className="window-title">{title}</span>{children}
    {window.windowBridge&&<div className="window-controls"><button aria-label="Minimize window" title="Minimize" onClick={()=>act('minimize')}><Minus size={16}/></button><button aria-label={maximized?'Restore window':'Maximize window'} title={maximized?'Restore':'Maximize'} onClick={()=>act('maximize')}>{maximized?<Minimize2 size={15}/>:<Maximize2 size={15}/>}</button><button className="window-close" aria-label="Close window" title="Close" onClick={()=>act('close')}><X size={18}/></button></div>}
  </header>{error&&<p role="alert">{error}</p>}</>;
}
