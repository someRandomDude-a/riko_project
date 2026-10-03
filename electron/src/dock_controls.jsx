import React,{useEffect,useState} from 'react';
import {Mic,MicOff,AudioLines,Volume2,VolumeX,ArrowUp,ArrowUpRight,ArrowDown} from './ui/icons.jsx';
import useVoiceState from './use_voice_state.jsx';
import useRuntime from './use_runtime.jsx';
import {microphoneAction,voiceLabel} from './voice_state.mjs';
import useBorderEffects from './use_border_effects.jsx';
import {request} from './api.mjs';
import {connectEvents} from './event_connection.mjs';
import DashBar from './dash_bar.jsx';

export default function DockControls({preferences,form,canSend=true,onSend,onExpand,handleProps,collapsed=false,micOnly=false,disabled=false,generating=false}){
  const voice=useVoiceState(),runtime=useRuntime(),mic=microphoneAction(voice);
  const [pending,setPending]=useState(false),[error,setError]=useState('');
  const [connected,setConnected]=useState(false);
  useEffect(()=>connectEvents(()=>{},setConnected),[]);
   const muted=runtime.audio===false;
   const effects=useBorderEffects(preferences,voice,generating||runtime.runtime?.generating);
  async function act(path){setPending(true);setError('');try{await request(path,{method:'POST'});}catch(e){setError(e.message);}finally{setPending(false);}}
  const status=!connected?'offline':generating||runtime.runtime?.generating?'generating':!voice.enabled?'off':voice.phase==='capturing'?'listening':voice.phase==='transcribing'?'processing':voice.wake?.active?'awake':'waiting';
   return <div className={'glass-dock configured-borders status-'+status+' '+(micOnly?'mic-only ':'')+(collapsed?'dock-collapsed ':'')+(effects.dock.active?effects.dock.className:'')} style={{'--mic-level':voice.level,...effects.dock.style}}>
    <div className="dock-buttons">
      {!collapsed&&<button type="button" className="dock-side" aria-label={muted?'Unmute audio output':'Mute audio output'} aria-pressed={muted} title={muted?'Unmute audio':'Mute audio'} disabled={pending} onClick={()=>act('/api/audio/toggle')}>{muted?<VolumeX size={19}/>:<Volume2 size={19}/>}</button>}
        <button type="button" className={'dock-mic '+(voice.enabled?'mic-running ':'')+(effects.mic.active?effects.mic.className:'')} style={effects.mic.style} aria-label={mic.label} title={voiceLabel(voice)+' · '+mic.label} disabled={disabled||pending||voice.status==='starting'||(mic.icon==='speak'&&(voice.wake?.calibrating||voice.wake?.testing))} onClick={()=>act(mic.path)}>{!voice.enabled?<MicOff size={27}/>:voice.phase==='capturing'?<AudioLines size={27}/>:<Mic size={27}/>}</button>
      {!collapsed&&<button type={form?'submit':'button'} form={form} className="dock-side" aria-label={form?'Send message':'Open chat to write a message'} title={form?'Send message':'Write a message'} disabled={!canSend} onClick={onSend}><ArrowUp size={20}/></button>}
    </div>
    <small role="status">{pending?'Please wait…':voiceLabel(voice)}</small>
    <DashBar className="dock-handle" label={collapsed&&!micOnly?'Drag to move; click to open mini chat':'Drag to move; click to collapse'} onClick={()=>window.windowBridge?.mode(micOnly||collapsed?'compact':'collapsed')}/>
    {error&&<p role="alert">{error}</p>}
  </div>;
}
