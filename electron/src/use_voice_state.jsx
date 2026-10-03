import {useEffect,useState} from 'react';
import {connectEvents} from './event_connection.mjs';
import {initialVoice,reduceVoice} from './voice_state.mjs';
export default function useVoiceState(){
  const [state,setState]=useState(initialVoice);
  useEffect(()=>connectEvents(event=>setState(old=>reduceVoice(old,event)),connected=>{if(!connected)setState(old=>({...old,enabled:false,status:'disconnected',phase:'stopped'}));}),[]);
  return state;
}
