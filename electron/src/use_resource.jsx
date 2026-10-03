import {useEffect,useState} from 'react';
import {connectEvents} from './event_connection.mjs';

export default function useResource(topic){
  const [data,setData]=useState(null),[error,setError]=useState('');
  useEffect(()=>connectEvents(event=>{
    const value=event.type==='resource.snapshot'?event.payload?.[topic]:event.type==='resource.'+topic?event.payload:undefined;
    if(value!==undefined){setData(value);setError(value===null?'Runtime unavailable':'');}
  },connected=>setError(connected?'':'Backend disconnected')),[topic]);
  return [data,setData,error,setError];
}
