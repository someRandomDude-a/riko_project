import React, {useEffect, useLayoutEffect, useReducer, useRef, useState} from 'react';
import FormattedText from './formatted_text.jsx';
import {connectEvents} from './event_connection.mjs';
import {request} from './api.mjs';
import useResource from './use_resource.jsx';
import {Square, ArrowDown, Settings} from './ui/icons.jsx';
import SpeechPopup from './speech_popup.jsx';
import {initialTranscriptComposer,reduceTranscriptComposer,transcriptMatches} from './transcript_composer.mjs';
import {initialStreamChat,reduceStreamChat,endsTurn,visibleChatMessages,sendChat} from './stream_chat_reducer.mjs';
import DockControls from './dock_controls.jsx';
import DiscordLaunch from './discord_launch.jsx';
import DiscordNotifications from './discord_notifications.jsx';

// Memoized, with Message and FormattedText: a snapshot or a streamed token renders only what it changed (the streaming message).
function StreamChat({onSettings,preferences={},compactMode=false,collapsedMode=false}) {
  const [chat, dispatch] = useReducer(reduceStreamChat, initialStreamChat);
  const {messages, busy, error, notice, speechError, characterName, startupError} = chat;
  const setError = error => dispatch({type: 'local.error', error});
  const [text, setText] = useState('');
  const [connected, setConnected] = useState(false);
  const [transcript,setTranscript]=useState(initialTranscriptComposer);
  useEffect(()=>connectEvents(event=>setTranscript(old=>reduceTranscriptComposer(old,event)),connected=>{if(!connected)setTranscript(old=>reduceTranscriptComposer(old,{type:'connection.closed'}));}),[]);
  useEffect(()=>{if(transcript.phase!=='sending')return;const timer=setTimeout(()=>setTranscript(old=>reduceTranscriptComposer(old,{type:'transcript.dismiss',id:transcript.id})),preferences.reduceMotion?80:420);return()=>clearTimeout(timer);},[transcript.id,transcript.phase,preferences.reduceMotion]);
  const inlinePopup=!collapsedMode&&preferences.chatTranscript!==false&&!!transcript.text&&transcript.phase!=='idle';
  const visibleMessages=visibleChatMessages(messages,transcript,inlinePopup);
  const [taskActions,setTaskActions]=useState([]);
  const [taskState]=useResource('tasks');
  const [sessions, setSessions] = useState({}), [hasMore, setHasMore] = useState(false), [loading, setLoading] = useState(false);
  const list = useRef(null), follow = useRef(true), loadingRef = useRef(false), cursor = useRef(null), restore = useRef(null), alive = useRef(true);
  const refreshQueued=useRef(null);
  async function loadHistory(older=false,reconnect=false) {
    if (loadingRef.current) {if(!older)refreshQueued.current={reconnect:reconnect||refreshQueued.current?.reconnect};return;}
    loadingRef.current=true; setLoading(true);
    try {
      const page = await request('/api/chat/history?limit=40' + (older && cursor.current ? '&before='+cursor.current : ''));
      if (!alive.current) return;
      if (older && list.current) restore.current={height:list.current.scrollHeight,top:list.current.scrollTop};
      if(reconnect){follow.current=true;restore.current=null;}
      dispatch({type:'local.history',page,reconnect});
      setSessions(old=>({...old,...page.sessions}));
      if (older || reconnect || cursor.current === null) {cursor.current=page.before;setHasMore(page.has_more);}
    } catch(error) {if(alive.current)dispatch({type:'local.history_error',error:error.message});}
    finally {loadingRef.current=false;if(alive.current){setLoading(false);if(refreshQueued.current){const pending=refreshQueued.current;refreshQueued.current=null;queueMicrotask(()=>loadHistory(false,pending.reconnect));}}}
  }
  useLayoutEffect(()=>{
    const element=list.current;if(!element)return;
    if(restore.current){element.scrollTop=restore.current.top+element.scrollHeight-restore.current.height;restore.current=null;}
    else if(follow.current)element.scrollTop=element.scrollHeight;
   },[messages,transcript.text,transcript.phase,collapsedMode]);
  useEffect(()=>{
    const observer=new ResizeObserver(()=>{if(follow.current&&list.current)list.current.scrollTop=list.current.scrollHeight;});
    if(list.current)observer.observe(list.current);
    return()=>observer.disconnect();
  },[]);
  useEffect(()=>{
    if(!preferences.showTaskActions||!connected||startupError){setTaskActions([]);return;}
    setTaskActions((taskState?.tasks||[]).filter(task=>['active','blocked'].includes(task.status)).slice(0,4));
  },[preferences.showTaskActions,connected,startupError,taskState]);
  // Another part's error passes: shown for 8 s (stream_chat_reducer.mjs keeps it apart from the chat's error).
  useEffect(()=>{if(!notice)return;const timer=setTimeout(()=>dispatch({type:'local.notice_expired',id:notice.id}),8000);return()=>clearTimeout(timer);},[notice?.id]);
  useEffect(() => {
    alive.current=true;
    const disconnect = connectEvents(event => {dispatch(event); if (endsTurn(event)) loadHistory();},
      connected => {setConnected(connected); if (!connected) dispatch({type: 'local.disconnected'}); else loadHistory(false,true);});
    return()=>{alive.current=false;disconnect();};
  }, []);
  async function send(e) {
    e.preventDefault();
    if (!text.trim() || busy) return;
    dispatch({type: 'local.busy', busy: true}); setError('');
    const value = text; setText('');
    const result = await sendChat(request, value);
    if (result.restore) setText(current => current || result.restore);
    if (result.error) setError(result.error);
    dispatch({type: 'local.busy', busy: false});
  }
  async function stop() {
    try {await request('/api/chat/stop', {method: 'POST'});}
    catch (error) {setError(error.status ? 'Unable to stop response' : error.message);}
  }
  return <main className="control stream-chat">
    <header className="conversation-heading"><h1>{characterName||'Chat'}</h1><div className="conversation-indicators"><span className={'connection-badge '+(connected&&!startupError?'online':'')}><i/>{startupError?'Setup needed':connected?busy?'Generating…':'Connected':'Connecting…'}</span><DiscordLaunch disabled={!connected||!!startupError} onError={setError}/></div></header>
    {startupError&&<div className="notice error" role="alert"><div><strong>Your runtime needs attention</strong><p>{startupError}</p></div><button onClick={onSettings}><Settings size={16}/>Open settings</button></div>}
    <div className="chat-status">
      <button className="quiet-button" disabled={!connected||!!startupError} onClick={stop}><Square size={14}/>Stop reply</button>
      <button className="quiet-button" onClick={()=>{follow.current=true;if(list.current)list.current.scrollTop=list.current.scrollHeight;}}><ArrowDown size={14}/>Latest</button></div>
    {error && <p role="alert">{error}</p>}
    {notice && notice.text!==error && <p role="alert">{notice.text}</p>}
    <DiscordNotifications/>
    {speechError && <p role="alert">Voice output unavailable. Chat remains available. Speech retries on the next reply. {speechError}</p>}
    <section ref={list} className="chat" onScroll={e=>{const element=e.currentTarget;follow.current=element.scrollHeight-element.clientHeight-element.scrollTop<60;if(element.scrollTop<80&&hasMore&&!loadingRef.current)loadHistory(true);}}>
      {hasMore&&<button disabled={loading} onClick={()=>loadHistory(true)}>{loading?'Loading…':'Load older messages'}</button>}
       {/* The dock hides the list (morph.css): it is not rendered there at all, so a streamed reply costs the dock nothing. */}
       {!collapsedMode&&visibleMessages.map((message,index)=><React.Fragment key={message.id}>
        {message.session_id!==visibleMessages[index-1]?.session_id&&<details className="session-marker"><summary>{message.source==='discord'?'Discord · ':''}{message.session_id==='legacy'?'Imported conversation':`Session · ${sessions[message.session_id]?.started ? new Date(sessions[message.session_id].started*1000).toLocaleString() : 'Current'}`} {sessions[message.session_id]?.provider}</summary>
          <p>{sessions[message.session_id]?.outcome || 'running'} · {sessions[message.session_id]?.ended ? 'Ended '+new Date(sessions[message.session_id].ended*1000).toLocaleString() : 'No recorded end time'}</p>
        </details>}
           <Message message={message} compact showInferenceStats={preferences.showInferenceStats!==false} {...(inlinePopup&&message.interjections?.length?{pendingTranscript:transcript.text,pendingStartedAt:transcript.startedAt}:{})}/></React.Fragment>)}
     </section><div className="conversation-dock">
      {(preferences.quickActions?.length>0||taskActions.length>0)&&<div className="quick-actions">{(preferences.quickActions||[]).filter(action=>action.label?.trim()&&action.prompt?.trim()).map((action,index)=><button key={action.id||index} onClick={()=>setText(action.prompt)}>{action.label}</button>)}{taskActions.map(task=><button key={task.id} onClick={()=>setText(`Help me work on task ${task.id}. Retrieve its details using the task tools.`)} title={task.title}>{task.title}</button>)}</div>}
        <div className={'composer-slot'+(inlinePopup?' popup-replaces-input':'')}><form id="chat-message-form" className="composer" onSubmit={send}><input aria-label="Message" placeholder="Message" value={text} tabIndex={inlinePopup?-1:undefined} onChange={e => setText(e.target.value)}/></form>
        <SpeechPopup inline sending={transcript.phase==='sending'} prefix="transcript" text={transcript.text} visible={inlinePopup} preferences={preferences}/></div>
       <DockControls collapsed={collapsedMode} generating={busy} preferences={preferences} form={!collapsedMode?'chat-message-form':undefined} canSend={connected&&!busy&&!startupError&&!!text.trim()} disabled={!connected||!!startupError}/>
    </div></main>;
}
export default React.memo(StreamChat);

// A bubble renders again only when its own message (the reducer keeps untouched messages' objects) or its props change. Only a message
// with interjections reads the pending transcript (an interjection it matches stays out until the popup ends).
export const Message=React.memo(function Message({message,compact=false,showInferenceStats=true,pendingTranscript='',pendingStartedAt=0}) {
  message={...message,compact,showInferenceStats};
  let cursor = 0;
  const parts = [];
  for (const [index, item] of (message.interjections || []).entries()) {
    const offset = Math.max(cursor, Math.min(message.text.length, item.offset));
    if (offset > cursor) parts.push(<ReplyPart message={message} start={cursor} end={offset} key={`a${index}`}/>);
    if(!(transcriptMatches(pendingTranscript,item.display_text||item.text)&&(!pendingStartedAt||item.ended_at>=pendingStartedAt-.2)))parts.push(<div className="msg user" key={`u${index}`}>{item.system_label&&<small className="system-extension">{item.system_label}</small>}<FormattedText text={item.display_text||item.text}/>{compact&&<small className="bubble-time">{new Date((item.started_at||message.timestamp)*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</small>}</div>);
    cursor = offset;
  }
  parts.push(<ReplyPart message={message} start={cursor} end={message.text.length} key="tail"/>);
  if (message.interrupted&&!compact) parts.push(<small key="status">Interrupted — grey text was not spoken (word position estimated)</small>);
  if (message.error) parts.push(<p role="alert" key="error">{message.error}</p>);
  return <>{parts}</>;
});

function ReplyPart({message, start, end}) {
  const cutoff = message.interrupted && Number.isFinite(message.cutoff) ? message.cutoff : message.text.length;
  const split = Math.max(start, Math.min(end, cutoff));
  return <div className={'msg ' + message.role} style={{whiteSpace: 'pre-wrap'}}>
    {message.source&&<small className="source-tag">{message.source==='discord'?'Discord':message.source==='microphone'?'Microphone':'Message'}</small>}
    <FormattedText text={message.text.slice(start, split)}/>
    {split < end && <div style={{color: '#96919e', opacity: 0.55}} title="Generated, but not spoken"><FormattedText text={message.text.slice(split, end)}/></div>}
    {message.compact&&message.timestamp&&<small className="bubble-time">{new Date(typeof message.timestamp==='number'?message.timestamp*1000:message.timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</small>}
    {message.role==='assistant'&&message.showInferenceStats&&message.metrics&&<small className="inference-stats">{message.metrics.estimated?'~':''}{message.metrics.tokens_per_second??'…'} tokens/s{message.metrics.estimated?' (estimated)':''} · {message.metrics.output_tokens} tokens · first token {message.metrics.first_token_seconds??'…'}s</small>}
  </div>;
}
