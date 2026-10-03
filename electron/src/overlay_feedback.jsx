import React,{useEffect,useState} from 'react';
import {connectEvents} from './event_connection.mjs';
import useVoiceState from './use_voice_state.jsx';
import useBorderEffects from './use_border_effects.jsx';
import {initialReply,reduceOverlayReply} from './overlay_reply.mjs';
import SpeechPopup,{usePopupVisibility} from './speech_popup.jsx';
export default function OverlayFeedback({preferences:p,geometry,update}){
 const voice=useVoiceState();
 const [chatVisible,setChatVisible]=useState(true),[output,setOutput]=useState(initialReply),[surface,setSurface]=useState(null);
 const effects=useBorderEffects(p,voice,output.generating);
  useEffect(()=>{const hit=e=>window.popupBridge?.interactive(!!e.target.closest?.('.speech-popup.popup-visible')||!!document.querySelector('.speech-popup[data-dragging=true]'));document.addEventListener('pointermove',hit,true);document.addEventListener('mousemove',hit,true);return()=>{document.removeEventListener('pointermove',hit,true);document.removeEventListener('mousemove',hit,true);};},[]);
 useEffect(()=>{
  let alive=true;
  window.controlVisibilityBridge?.get().then(v=>{if(alive)setChatVisible(v);}).catch(()=>{});
  const off=window.controlVisibilityBridge?.subscribe(setChatVisible),surfaceOff=window.chatSurfaceBridge?.subscribe(setSurface);
  const disconnect=connectEvents(event=>setOutput(old=>reduceOverlayReply(old,event)),connected=>{if(!connected)setOutput(old=>({...old,generating:false}));});
  return()=>{alive=false;off?.();surfaceOff?.();disconnect();window.popupBridge?.interactive(false);};
 },[]);
 const transcript=voice.transcript?.text||'';
 const transcriptVisible=usePopupVisibility(transcript,voice.enabled&&!voice.transcript?.final,p.transcriptSeconds);
 const replyVisible=usePopupVisibility(output.text,output.generating,p.replySeconds);
  const mini=surface?.mode==='compact';
 return <>
  {effects.screen.active&&<div className={'screen-border-effect '+effects.screen.className} aria-label="Screen voice feedback" style={effects.screen.style}/>}
  <SpeechPopup prefix="transcript" text={transcript} visible={p.overlayTranscript&&transcriptVisible&&!chatVisible&&!mini} preferences={p} geometry={geometry} surface={surface} update={update}/>
  <SpeechPopup prefix="reply" text={output.text} visible={!mini&&replyVisible&&p.replyBubble!=='off'&&(p.replyBubble==='always'||!chatVisible)} preferences={p} geometry={geometry} surface={surface} update={update}/>
 </>;
}
