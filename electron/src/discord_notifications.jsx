import React,{useState} from 'react';
import useRuntime from './use_runtime.jsx';
export default function DiscordNotifications(){
 const state=useRuntime(),[dismissed,setDismissed]=useState([]);
 const items=(state.notifications||[]).filter(n=>n.source==='discord'&&!dismissed.includes(n.id)).slice(0,3);
 return <div aria-live="polite">{items.map(n=><div className={'notice '+(n.level==='error'?'error':'')} key={n.id}><strong>Discord</strong><span>{n.text}</span><button aria-label="Dismiss Discord notification" onClick={()=>setDismissed(old=>[...old.slice(-19),n.id])}>Dismiss</button></div>)}</div>;
}
