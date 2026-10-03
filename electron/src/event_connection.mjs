import {EVENTS} from './api.mjs';

/** One owner for every socket, including sockets created after reconnect. */
function openConnection(onEvent, onConnected = () => {}, {
  Socket = WebSocket, schedule = setTimeout, unschedule = clearTimeout,
  url = EVENTS,
} = {}) {
  let stopped = false, socket, timer;
  function connect() {
    socket = new Socket(url);
    socket.onopen = () => {if (!stopped){globalThis.window?.processBridge?.sync();onConnected(true);}};
    socket.onmessage = event => {
      if (stopped) return;
      let value;
      try {value = JSON.parse(event.data);} catch {return;}
      if (value && typeof value.type === 'string') onEvent(value);
    };
    socket.onclose = () => {
      if (stopped) return;
      onConnected(false);
      timer = schedule(connect, 1000);
    };
  }
  connect();
  return () => {
    stopped = true;
    unschedule(timer);
    socket.onopen = socket.onmessage = socket.onclose = null;
    socket.close();
  };
}

export function createEventHub(open=openConnection){
  const listeners=new Set(),cache=new Map();let close=null,connected=false;
  const deliver=(callback,value)=>{try{callback(value);}catch(error){console.error('Event subscriber failed',error);}};
  return (onEvent,onConnected=()=>{})=>{
    const subscriber={onEvent,onConnected};listeners.add(subscriber);
    if(!close)close=open(event=>{
      if(event.type==='resource.snapshot')for(const key of cache.keys())if(key.startsWith('resource.'))cache.delete(key);
      if(event.type==='state.snapshot'||event.type.startsWith('resource.'))cache.set(event.type,event);
      for(const item of listeners)deliver(item.onEvent,event);
    },value=>{connected=value;for(const item of listeners)deliver(item.onConnected,value);});
    else {deliver(onConnected,connected);if(connected)for(const event of cache.values())deliver(onEvent,event);}
    return ()=>{listeners.delete(subscriber);if(!listeners.size){close?.();close=null;connected=false;cache.clear();}};
  };
}
const shared=createEventHub();
export function connectEvents(onEvent,onConnected=()=>{},options){
  return options?openConnection(onEvent,onConnected,options):shared(onEvent,onConnected);
}
