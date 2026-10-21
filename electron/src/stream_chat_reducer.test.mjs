import test from 'node:test';
import assert from 'node:assert/strict';
import {initialStreamChat,reduceStreamChat,endsTurn,visibleChatMessages,sendChat} from './stream_chat_reducer.mjs';
const run=(events,state=initialStreamChat)=>events.reduce(reduceStreamChat,state);
const snapshot=(payload={})=>({type:'state.snapshot',payload:{character_name:'Mika',session_id:'s1',runtime:{generating:false},...payload}});
test('a streamed turn: the user line, a growing reply, then the completed text, with busy following the model',()=>{
 let state=run([snapshot(),{type:'chat.input',turn_id:'t1',timestamp:10,sequence:1,payload:{text:'hi',source:'chat'}},{type:'model.started',turn_id:'t1',timestamp:11,sequence:2,payload:{source:'chat'}}]);
 assert.equal(state.characterName,'Mika');assert.equal(state.busy,true);
 assert.deepEqual(state.messages.map(m=>[m.id,m.role,m.text,m.session_id]),[['t1:user','user','hi','s1'],['t1','assistant','','s1']]);
 state=run([{type:'chat.delta',turn_id:'t1',sequence:3,payload:{text:'Hel'}},{type:'chat.delta',turn_id:'t1',sequence:4,payload:{text:'lo'}}],state);
 assert.equal(state.messages[1].text,'Hello');assert.equal(state.messages[1].timestamp,11);assert.equal(state.messages[1].event_sequence,4);
 state=run([{type:'model.metrics',turn_id:'t1',sequence:5,payload:{tokens_per_second:40}},{type:'chat.completed',turn_id:'t1',sequence:6,payload:{text:'Hello there.'}}],state);
 assert.equal(state.messages[1].text,'Hello there.');assert.deepEqual(state.messages[1].metrics,{tokens_per_second:40});assert.equal(state.busy,false);
 // A Discord conversation keeps its own session; a delta for an unknown turn starts its message.
 state=run([{type:'chat.delta',turn_id:'t2',timestamp:20,sequence:7,payload:{text:'from discord',conversation_id:'d9',source:'discord'}}],state);
 assert.deepEqual(state.messages.at(-1),{id:'t2',role:'assistant',source:'discord',conversation_id:'d9',timestamp:20,session_id:'d9',event_sequence:7,text:'from discord'});
});
test('interruptions mark where speech stopped, and close interjections join into one',()=>{
 let state=run([snapshot(),{type:'model.started',turn_id:'t',timestamp:1,payload:{}},{type:'chat.delta',turn_id:'t',payload:{text:'a long reply'}}]);
 state=run([{type:'chat.interjection',turn_id:'t',sequence:4,payload:{text:'wait',started_at:5,ended_at:6,debounce_seconds:2,offset:3}},
  {type:'chat.interjection',turn_id:'t',sequence:5,payload:{text:'stop',display_text:'stop!',system_label:'Speaking over you',started_at:7,ended_at:8,debounce_seconds:2}},
  {type:'chat.interjection',turn_id:'t',sequence:6,payload:{text:'later',started_at:20,ended_at:21,debounce_seconds:2}},
  {type:'chat.interrupted',turn_id:'t',sequence:7,payload:{offset:6}}],state);
 const message=state.messages[0];
 assert.deepEqual(message.interjections.map(i=>[i.text,i.display_text,i.ended_at]),[['wait stop!','wait stop!',8],['later',undefined,21]]);
 assert.equal(message.interjections[0].system_label,'Speaking over you');
 assert.deepEqual([message.interrupted,message.cutoff,message.event_sequence],[true,6,7]);
 state=reduceStreamChat(state,{type:'chat.cancelled',turn_id:'t',sequence:8});assert.equal(state.messages[0].interrupted,true);assert.equal(state.busy,false);
});
test('errors land where they belong: speech apart from chat, and a startup error clears the chat error',()=>{
 let state=run([{type:'model.error',turn_id:'t',payload:{error:'context overflow'}}]);
 assert.equal(state.error,'context overflow');assert.equal(state.busy,false);
 state=run([{type:'speech.unavailable',payload:{error:'GPT-SoVITS offline'}},{type:'tool.error',payload:{error:'calculator failed'}}],state);
 assert.equal(state.speechError,'GPT-SoVITS offline');assert.equal(state.error,'context overflow','another part\'s error is not the chat\'s');
 assert.deepEqual(state.notice,{id:1,text:'calculator failed'});
 state=reduceStreamChat(state,{type:'speech.started'});assert.equal(state.speechError,'');
 state=reduceStreamChat(state,snapshot({startup_error:'Model missing',runtime:{generating:true,speech_error:'no voice'}}));
 assert.deepEqual([state.error,state.startupError,state.busy,state.speechError],['','Model missing',true,'no voice']);
 state=reduceStreamChat(state,{type:'local.history_error',error:'Request failed (503)'});assert.equal(state.error,'','the startup error explains a failed history load');
 state=reduceStreamChat(reduceStreamChat(state,snapshot()),{type:'local.history_error',error:'Request failed (503)'});assert.equal(state.error,'Request failed (503)');
 assert.equal(reduceStreamChat(state,{type:'voice.transcript',payload:{text:'x'}}),state,'unrelated events change nothing');
});
test('history pages merge in; a reconnect drops live messages the archive lacks but keeps this session\'s unarchived ones',()=>{
 let state=run([snapshot(),{type:'chat.input',turn_id:'live',timestamp:5,payload:{text:'pending'}},{type:'chat.input',turn_id:'old',timestamp:4,payload:{text:'gone',conversation_id:'other'}}]);
 const page={messages:[{id:'a',role:'user',text:'archived',timestamp:1,sequence:1,session_id:'s1'}]};
 assert.deepEqual(reduceStreamChat(state,{type:'local.history',page,reconnect:false}).messages.map(m=>m.id).sort(),['a','live:user','old:user']);
 state=reduceStreamChat(state,{type:'local.history',page,reconnect:true});
 assert.deepEqual(state.messages.map(m=>m.id).sort(),['a','live:user']);
 state=run([{type:'local.busy',busy:true},{type:'local.error',error:'x'}],state);assert.deepEqual([state.busy,state.error],[true,'x']);
 assert.equal(reduceStreamChat(state,{type:'local.disconnected'}).busy,false);
});
test('a finished, cancelled or failed turn reloads the archive; nothing else does',()=>{
 assert.deepEqual(['chat.completed','chat.cancelled','model.error','chat.delta','speech.error','state.snapshot'].map(type=>endsTurn({type})),[true,true,true,false,false,false]);
});
test('the inline transcript popup hides the user message it is showing, so speech appears once',()=>{
 const messages=[{id:'1',role:'user',text:'hello there',timestamp:90},{id:'2',role:'assistant',text:'hi',timestamp:91},{id:'3',role:'user',text:'hello there',timestamp:100}];
 const transcript={text:'Hey Mika, hello there',startedAt:99.9};
 assert.deepEqual(visibleChatMessages(messages,transcript,true).map(m=>m.id),['1','2'],'the latest matching user line since the utterance');
 assert.equal(visibleChatMessages(messages,transcript,false),messages,'no popup, every message');
 assert.equal(visibleChatMessages(messages,{text:'something else',startedAt:0},true),messages);
 assert.deepEqual(visibleChatMessages(messages.slice(0,2),{text:'hello there',startedAt:99},true).map(m=>m.id),['1','2'],'older lines are not the utterance');
});
test('a chat the backend refuses as busy (409) gives the typed message back; other failures only explain',async()=>{
 const refused=status=>async()=>{throw Object.assign(new Error('Riko is already handling another turn'),{status});};
 assert.deepEqual(await sendChat(refused(409),'hello'),{error:'Riko is already handling another turn',restore:'hello'});
 assert.deepEqual(await sendChat(refused(500),'hello'),{error:'Riko is already handling another turn',restore:''});
 const sent=[];assert.deepEqual(await sendChat(async(...args)=>{sent.push(args);return null;},'hello'),{error:'',restore:''});
 assert.deepEqual(sent,[['/api/chat',{method:'POST',body:{text:'hello'}}]]);
});
test('only model and chat errors set the chat error; the rest are notices that expire, and a completed turn clears the error',()=>{
 let state=run([{type:'voice.error',payload:{error:'Microphone processing fell behind; recording resumed'}}]);
 assert.equal(state.error,'');assert.deepEqual(state.notice,{id:1,text:'Microphone processing fell behind; recording resumed'});
 state=run([{type:'animation.error',payload:{error:'clip failed'}},{type:'initiative.error',payload:{}}],state);
 assert.deepEqual(state.notice,{id:3,text:'Something went wrong'},'the newest notice replaces the last');
 assert.equal(reduceStreamChat(state,{type:'local.notice_expired',id:2}),state,'an older timer leaves the newer notice');
 state=reduceStreamChat(state,{type:'local.notice_expired',id:3});assert.equal(state.notice,null);
 state=run([{type:'chat.error',payload:{error:'stream broke'}}],state);assert.equal(state.error,'stream broke');
 state=run([{type:'model.started',turn_id:'t',timestamp:1,payload:{}}],state);assert.equal(state.error,'stream broke','still shown while the next turn streams');
 state=run([{type:'chat.completed',turn_id:'t',payload:{text:'ok'}}],state);assert.equal(state.error,'');
});
test('a snapshot that changes none of the chat\'s fields keeps its state object, so nothing renders',()=>{
 const state=run([snapshot()]);
 assert.equal(reduceStreamChat(state,snapshot({emotion:{primary:'joy'},tools:[{id:1}]})),state);
 assert.notEqual(reduceStreamChat(state,snapshot({runtime:{generating:true}})),state);
 assert.equal(reduceStreamChat(state,{type:'local.disconnected'}),state,'not busy: nothing to clear');
});
