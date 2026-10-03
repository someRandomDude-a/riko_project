import {useEffect,useState} from 'react';
export function useAvatarEditorOpen(open){
 const [visible,setVisible]=useState(false);
 useEffect(()=>{if(typeof BroadcastChannel==='undefined')return;const channel=new BroadcastChannel('riko-avatar-editor');
  channel.onmessage=({data})=>{if(data?.type==='query'&&open)channel.postMessage({type:'open',value:true});if(data?.type==='open')setVisible(data.value===true);};
  if(open)channel.postMessage({type:'open',value:true});else channel.postMessage({type:'query'});
  const close=()=>{if(open)channel.postMessage({type:'open',value:false});};window.addEventListener('beforeunload',close);
  return()=>{close();window.removeEventListener('beforeunload',close);channel.close();};
 },[open]);return visible;
}
