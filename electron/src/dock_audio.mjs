export function audioPopover(state,event){
 if(event==='close')return {open:false,armed:false};
 if(event==='reveal')return {open:true,armed:state.armed};
 if(event==='click')return {open:true,armed:true,toggle:state.open&&state.armed};
 return state;
}
/** Serialize gain writes, keeping only the newest queued slider value. */
export function volumeWriter(write,onError=()=>{},onIdle=()=>{}){
 let next=null,busy=false,closed=false;
 async function pump(){if(busy||closed||next===null)return;busy=true;const value=next;next=null;
  try{await write(value);}catch(error){if(!closed)onError(error);}finally{busy=false;if(!closed){if(next===null)onIdle();else pump();}}
 }
 return {set(value){if(closed)return;next=value;pump();},get pending(){return busy||next!==null;},close(){closed=true;next=null;}};
}
