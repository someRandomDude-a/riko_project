export const initialTranscriptComposer={id:null,text:'',phase:'idle',startedAt:0};
export function transcriptMatches(text,message){
 const clean=value=>(value||'').trim().replace(/\s+/g,' ').toLocaleLowerCase();
 const a=clean(text),b=clean(message);
 return !!a&&!!b&&(a===b||a.endsWith(b));
}
export function reduceTranscriptComposer(state,event){
 const p=event.payload||{};
 if(event.type==='voice.started')return {id:p.utterance_id,text:'',phase:'live',startedAt:event.timestamp||0};
 if(event.type==='voice.transcript'){
  if(state.id&&p.utterance_id!==state.id)return state;
  if(state.phase==='idle'&&state.id===p.utterance_id)return state;
  if(state.phase==='sending'&&!p.final)return state;
  return {...state,id:p.utterance_id,text:p.text||'',phase:p.final?'sending':'live',startedAt:state.startedAt||event.timestamp||0};
 }
 if(event.type==='voice.stopped'||event.type==='voice.error'||event.type==='connection.closed')return {...state,text:'',phase:'idle'};
 if(event.type==='transcript.dismiss'&&event.id===state.id)return {...state,text:'',phase:'idle'};
 return state;
}
