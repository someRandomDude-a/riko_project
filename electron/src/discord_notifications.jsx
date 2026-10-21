import React,{useState} from 'react';
import useRuntime from './use_runtime.jsx';
const selectNotifications=state=>state.notifications;
function DiscordNotifications(){
 const notifications=useRuntime(selectNotifications),[dismissed,setDismissed]=useState([]);
 const items=(notifications||[]).filter(n=>['discord','tools'].includes(n.source)&&!dismissed.includes(n.id)).slice(0,3);
 return <div aria-live="polite">{items.map(n=><div className={'notice '+(n.level==='error'?'error':'')} key={n.id}><strong>{n.source==='tools'?'Tools':'Discord'}</strong><span>{n.text}</span><button aria-label={'Dismiss '+(n.source==='tools'?'tools':'Discord')+' notification'} onClick={()=>setDismissed(old=>[...old.slice(-19),n.id])}>Dismiss</button></div>)}</div>;
}
// Memoized: its parent's updates (a streamed token, a snapshot) render it only when its own props changed.
export default React.memo(DiscordNotifications);
