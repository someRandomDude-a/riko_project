export const initialReply={turn:null,text:'',generating:false};
export function reduceOverlayReply(state,event){
  const p=event.payload||{},turn=event.turn_id;
  if(event.type==='state.snapshot'){
    const runtime=p.runtime||{};
    if(runtime.generating&&(state.turn===null||state.turn===runtime.turn_id))return {turn:runtime.turn_id,text:(runtime.generated_text||'').slice(0,12000),generating:true};
    if(state.turn===null&&p.speech)return {turn:runtime.turn_id,text:p.speech.slice(0,12000),generating:false};
    return state; // Unrelated snapshots cannot replace the latest reply with old speech.
  }
  if(event.type==='model.started')return {turn,text:'',generating:true};
  if(event.type==='chat.completed')return {turn,text:(p.text||'').slice(0,12000),generating:false};
  if(turn!==state.turn)return state;
  if(event.type==='chat.delta')return {...state,text:(state.text+(p.text||'')).slice(0,12000)};
  if(['chat.cancelled','model.error'].includes(event.type))return {...state,generating:false};
  if(event.type==='chat.interrupted'&&Number.isFinite(p.offset))return {...state,text:state.text.slice(0,p.offset),generating:false};
  return state;
}
