/** Structural sharing for plain JSON data: next, with every part that equals the same part of previous replaced by previous's object,
 * and previous itself when the two are equal throughout. Two snapshots parsed from the socket share nothing, so this is what lets a
 * component compare the fields it reads by reference. */
export function share(previous,next){
 if(Object.is(previous,next))return previous;
 if(typeof previous!=='object'||typeof next!=='object'||previous===null||next===null)return next;
 const list=Array.isArray(next);if(list!==Array.isArray(previous))return next;
 let same=true,result;
 if(list){
  result=new Array(next.length);same=next.length===previous.length;
  for(let i=0;i<next.length;i++){result[i]=i<previous.length?share(previous[i],next[i]):next[i];if(result[i]!==previous[i])same=false;}
 }else{
  const keys=Object.keys(next);result={};same=keys.length===Object.keys(previous).length;
  for(const key of keys){const had=Object.hasOwn(previous,key);result[key]=had?share(previous[key],next[key]):next[key];if(!had||result[key]!==previous[key])same=false;}
 }
 return same?previous:result;
}
/** Equal by reference key by key (or as values): what a selector's result is compared with once its parts are shared. */
export function shallowEqual(a,b){
 if(Object.is(a,b))return true;
 if(typeof a!=='object'||typeof b!=='object'||a===null||b===null||Array.isArray(a)!==Array.isArray(b))return false;
 const keys=Object.keys(a);return keys.length===Object.keys(b).length&&keys.every(key=>Object.hasOwn(b,key)&&Object.is(a[key],b[key]));
}
/** The window's latest state.snapshot, shared with the previous one (share), for useSyncExternalStore. connect is
 * event_connection's connectEvents; the socket opens with the first subscriber. Listeners hear only snapshots that changed. */
export function createRuntimeStore(connect){
 let state={},close=null;const listeners=new Set();
 function receive(event){
  if(event.type!=='state.snapshot')return;
  const next=share(state,event.payload&&typeof event.payload==='object'?event.payload:{});
  if(next===state)return;state=next;
  for(const listener of [...listeners])listener();
 }
 return {
  get:()=>state,
  subscribe(listener){listeners.add(listener);if(!close)close=connect(receive);return()=>{listeners.delete(listener);if(!listeners.size&&close){close();close=null;}};},
 };
}
/** A selector for some top-level fields of the snapshot (one object per call; compared with shallowEqual). */
export const pick=keys=>state=>Object.fromEntries(keys.map(key=>[key,state[key]]));
