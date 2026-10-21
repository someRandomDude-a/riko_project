import {mergeHistory} from './chat_history.mjs';
import {transcriptMatches} from './transcript_composer.mjs';
// StreamChat's state: bus events (event_connection.mjs) plus its own `local.*` actions (history pages, send and stop, the
// socket closing). Pure, as reduceOverlayReply is: stream_chat.jsx keeps only refs, scrolling and requests.
export const initialStreamChat={messages:[],busy:false,error:'',notice:null,speechError:'',characterName:'',startupError:'',sessionID:null};
// The errors that belong to the conversation: the chat's error banner shows them until the next completed turn.
const CHAT_ERRORS=['model.error','chat.error'];
const ENDED=['chat.completed','chat.cancelled','model.error'];
// A finished, cancelled or failed turn: the archive now has it, so the chat reloads its latest page.
export const endsTurn=event=>ENDED.includes(event.type);
function interjection(message,p,sequence){
 const items=[...(message.interjections||[])],last=items.at(-1);
 // Speech within debounce_seconds of the previous interjection continues it.
 if(last&&p.started_at>=last.ended_at&&p.started_at-last.ended_at<=p.debounce_seconds)
  items[items.length-1]={...last,text:last.text+' '+(p.display_text||p.text),display_text:(last.display_text||last.text)+' '+(p.display_text||p.text),system_label:p.system_label,ended_at:p.ended_at};
 else items.push(p);
 return {...message,interjections:items,event_sequence:sequence};
}
function reduceMessages(messages,event,sessionID){
 const p=event.payload||{},turn=event.turn_id,sequence=event.sequence,session=p.conversation_id||sessionID;
 const update=change=>messages.map(m=>m.id===turn?{...m,...change,event_sequence:sequence}:m);
 switch(event.type){
  case 'chat.interrupted':return update({interrupted:true,cutoff:p.offset});
  case 'chat.cancelled':return update({interrupted:true});
  case 'chat.interjection':return messages.map(m=>m.id===turn?interjection(m,p,sequence):m);
  case 'chat.input':return mergeHistory(messages,[{id:`${turn}:user`,role:'user',text:p.text,source:p.source,conversation_id:p.conversation_id,timestamp:event.timestamp,session_id:session,event_sequence:sequence}]);
  case 'model.started':return mergeHistory(messages,[{id:turn,role:'assistant',text:'',timestamp:event.timestamp,source:p.source,session_id:session,event_sequence:sequence}]);
  case 'model.metrics':return update({metrics:p});
  case 'chat.delta':case 'chat.completed':{
   const index=messages.findIndex(m=>m.id===turn),previous=index<0?'':messages[index].text;
   const message={...(index<0?{}:messages[index]),id:turn,role:'assistant',source:p.source,conversation_id:p.conversation_id,timestamp:messages[index]?.timestamp||event.timestamp,session_id:session,event_sequence:sequence,text:event.type==='chat.completed'?p.text:previous+p.text};
   return index<0?[...messages,message]:messages.map((m,i)=>i===index?message:m);
  }
 }
 return messages;
}
export function reduceStreamChat(state,event){
 const p=event.payload||{};
 switch(event.type){
  case 'local.busy':return {...state,busy:event.busy};
  case 'local.error':return {...state,error:event.error};
  case 'local.disconnected':return state.busy?{...state,busy:false}:state;
  case 'local.notice_expired':return state.notice?.id===event.id?{...state,notice:null}:state;
  // A page of /api/chat/history. On reconnect, live messages the archive does not hold are dropped, except this session's
  // optimistic ones not yet archived (no sequence).
  case 'local.history':{
   const ids=new Set(event.page.messages.map(m=>m.id));
   const kept=event.reconnect?state.messages.filter(m=>ids.has(m.id)||m.sequence==null&&m.session_id===state.sessionID):state.messages;
   return {...state,messages:mergeHistory(kept,event.page.messages)};
  }
  // The runtime's startup error explains a failed load better.
  case 'local.history_error':return state.startupError?state:{...state,error:event.error};
 }
 let next=state;
 if(event.type==='state.snapshot'){
  const fields={characterName:p.character_name||'',busy:!!p.runtime?.generating,speechError:p.runtime?.speech_error||'',sessionID:p.session_id,startupError:p.startup_error||''};
  if(p.startup_error)fields.error='';
  // A snapshot comes with every desktop change: the chat keeps its state (and renders nothing) unless one of its own fields changed.
  if(Object.keys(fields).some(key=>!Object.is(fields[key],state[key])))next={...next,...fields};
 }
 const messages=reduceMessages(next.messages,event,next.sessionID);
 if(messages!==next.messages)next={...next,messages};
 if(event.type==='model.started')next={...next,busy:true};
 if(endsTurn(event))next={...next,busy:false};
 if(['speech.error','speech.unavailable'].includes(event.type))next={...next,speechError:p.error};
 else if(CHAT_ERRORS.includes(event.type))next={...next,error:p.error};
 // Another part's error (microphone, animation, initiative, wake feedback, a tool) is a notice that passes (StreamChat expires it), never
 // the chat's error: a recoverable microphone hiccup no longer leaves a banner that only a typed message clears.
 else if(event.type?.endsWith('.error'))next={...next,notice:{id:(state.notice?.id||0)+1,text:p.error||'Something went wrong'}};
 if(event.type==='chat.completed'&&next.error)next={...next,error:''};
 if(event.type==='speech.started')next={...next,speechError:''};
 return next;
}
// The user message the inline transcript popup is showing stays out of the list, so the speech appears once.
export function visibleChatMessages(messages,transcript,inlinePopup){
 const pending=inlinePopup?messages.findLast(m=>m.role==='user'&&(!transcript.startedAt||m.timestamp>=transcript.startedAt-.2)&&transcriptMatches(transcript.text,m.text)):null;
 return pending?messages.filter(m=>m.id!==pending.id):messages;
}
// POST /api/chat. A busy backend (409) never took the message, so the composer gets it back to resend.
export async function sendChat(request,value){
 try{await request('/api/chat',{method:'POST',body:{text:value}});return {error:'',restore:''};}
 catch(error){return {error:error.message,restore:error.status===409?value:''};}
}
