import React,{useEffect,useState} from 'react';
import {MessageCircle} from './ui/icons.jsx';
import {request} from './api.mjs';
import useResource from './use_resource.jsx';
export default function DiscordLaunch({disabled=false,onError}){
 const [status,setStatus]=useResource('discord'),[busy,setBusy]=useState(false);
 useEffect(()=>{let alive=true;request('/api/discord/process').then(value=>{if(alive)setStatus(value);}).catch(()=>{});return()=>{alive=false;};},[]);
 async function start(){setBusy(true);onError?.('');try{setStatus(await request('/api/discord/start',{method:'POST'}));}catch(error){onError?.(error.message);}finally{setBusy(false);}}
  return <button type="button" className="quiet-button discord-launch" aria-label={status?.running?'Check Discord status':'Start Discord client'} title={status?.error||'Start or check the Discord transport'} disabled={disabled||busy} onClick={start}><MessageCircle size={15}/>{busy?'Starting Discord…':status?.ready?'Discord connected':status?.running?'Discord connecting…':'Start Discord'}</button>;
}
