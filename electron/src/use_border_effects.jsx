import {useEffect,useState} from 'react';
import {connectEvents} from './event_connection.mjs';
import {borderEffect} from './border_effects.mjs';
export default function useBorderEffects(preferences,voice,generating=false){
 const [wake,setWake]=useState(false);
 useEffect(()=>{let timer;const off=connectEvents(event=>{if(event.type==='voice.activated'){setWake(true);clearTimeout(timer);timer=setTimeout(()=>setWake(false),1100);}},connected=>{if(!connected)setWake(false);});return()=>{off();clearTimeout(timer);};},[]);
 return Object.fromEntries(['mic','screen','dock'].map(target=>[target,borderEffect(target,preferences,voice,{wake,generating})]));
}
