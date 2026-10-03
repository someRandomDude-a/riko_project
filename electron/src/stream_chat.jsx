import React, {useEffect, useLayoutEffect, useRef, useState} from 'react';
import FormattedText from './formatted_text.jsx';
import {connectEvents} from './event_connection.mjs';
import {request} from './api.mjs';
import useResource from './use_resource.jsx';
import {mergeHistory} from './chat_history.mjs';
import {Square, ArrowDown, Settings} from './ui/icons.jsx';
import SpeechPopup from './speech_popup.jsx';
import {initialTranscriptComposer,reduceTranscriptComposer,transcriptMatches} from './transcript_composer.mjs';
import DockControls from './dock_controls.jsx';
import DiscordLaunch from './discord_launch.jsx';
import DiscordNotifications from './discord_notifications.jsx';

export default function StreamChat({onSettings,preferences={},compactMode=false,collapsedMode=false}) {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [speechError, setSpeechError] = useState('');
  const [characterName,setCharacterName]=useState('');
  const [transcript,setTranscript]=useState(initialTranscriptComposer);
  useEffect(()=>connectEvents(event=>setTranscript(old=>reduceTranscriptComposer(old,event)),connected=>{if(!connected)setTranscript(old=>reduceTranscriptComposer(old,{type:'connection.closed'}));}),[]);
  useEffect(()=>{if(transcript.phase!=='sending')return;const timer=setTimeout(()=>setTranscript(old=>reduceTranscriptComposer(old,{type:'transcript.dismiss',id:transcript.id})),preferences.reduceMotion?80:420);return()=>clearTimeout(timer);},[transcript.id,transcript.phase,preferences.reduceMotion]);
  const inlinePopup=!collapsedMode&&preferences.chatTranscript!==false&&!!transcript.text&&transcript.phase!=='idle';
  const pendingMessage=inlinePopup?messages.findLast(m=>m.role==='user'&&(!transcript.startedAt||m.timestamp>=transcript.startedAt-.2)&&transcriptMatches(transcript.text,m.text)):null;
  const visibleMessages=pendingMessage?messages.filter(m=>m.id!==pendingMessage.id):messages;
  const [startupError,setStartupError]=useState('');
  const [taskActions,setTaskActions]=useState([]);
  const [taskState]=useResource('tasks');
  const startupErrorRef=useRef('');
  const [sessions, setSessions] = useState({}), [hasMore, setHasMore] = useState(false), [loading, setLoading] = useState(false);
  const list = useRef(null), follow = useRef(true), loadingRef = useRef(false), cursor = useRef(null), restore = useRef(null), sessionID = useRef(null), alive = useRef(true);
  const refreshQueued=useRef(null);
  async function loadHistory(older=false,reconnect=false) {
    if (loadingRef.current) {if(!older)refreshQueued.current={reconnect:reconnect||refreshQueued.current?.reconnect};return;}
    loadingRef.current=true; setLoading(true);
    try {
      const page = await request('/api/chat/history?limit=40' + (older && cursor.current ? '&before='+cursor.current : ''));
      if (!alive.current) return;
      if (older && list.current) restore.current={height:list.current.scrollHeight,top:list.current.scrollTop};
      if(reconnect){follow.current=true;restore.current=null;}
      const pageIDs=new Set(page.messages.map(message=>message.id));
      setMessages(old=>mergeHistory(reconnect?old.filter(message=>pageIDs.has(message.id)||message.sequence==null&&message.session_id===sessionID.current):old,page.messages));
      setSessions(old=>({...old,...page.sessions}));
      if (older || reconnect || cursor.current === null) {cursor.current=page.before;setHasMore(page.has_more);}
    } catch(error) {if(alive.current&&!startupErrorRef.current)setError(error.message);}
    finally {loadingRef.current=false;if(alive.current){setLoading(false);if(refreshQueued.current){const pending=refreshQueued.current;refreshQueued.current=null;queueMicrotask(()=>loadHistory(false,pending.reconnect));}}}
  }
  useLayoutEffect(()=>{
    const element=list.current;if(!element)return;
    if(restore.current){element.scrollTop=restore.current.top+element.scrollHeight-restore.current.height;restore.current=null;}
    else if(follow.current)element.scrollTop=element.scrollHeight;
   },[messages,transcript.text,transcript.phase]);
  useEffect(()=>{
    const observer=new ResizeObserver(()=>{if(follow.current&&list.current)list.current.scrollTop=list.current.scrollHeight;});
    if(list.current)observer.observe(list.current);
    return()=>observer.disconnect();
  },[]);
  useEffect(()=>{
    if(!preferences.showTaskActions||!connected||startupError){setTaskActions([]);return;}
    setTaskActions((taskState?.tasks||[]).filter(task=>['active','blocked'].includes(task.status)).slice(0,4));
  },[preferences.showTaskActions,connected,startupError,taskState]);
  useEffect(() => {
    alive.current=true;
    const disconnect = connectEvents(event => {
        const p = event.payload || {};
        if (event.type === 'state.snapshot') {setCharacterName(p.character_name||'');setBusy(!!p.runtime?.generating);setSpeechError(p.runtime?.speech_error||'');sessionID.current=p.session_id;startupErrorRef.current=p.startup_error||'';setStartupError(p.startup_error||'');if(p.startup_error)setError('');}
        if (event.type === 'chat.interrupted') {
          setMessages(old => old.map(m => m.id === event.turn_id ? {...m, interrupted: true, cutoff: p.offset,event_sequence:event.sequence} : m));
        }
        if (event.type === 'chat.cancelled') {
          setMessages(old => old.map(m => m.id === event.turn_id ? {...m, interrupted: true,event_sequence:event.sequence} : m));
        }
        if (event.type === 'chat.interjection') {
          setMessages(old => old.map(m => {
            if (m.id !== event.turn_id) return m;
            const items = [...(m.interjections || [])];
            const last = items.at(-1);
            if (last && p.started_at >= last.ended_at && p.started_at - last.ended_at <= p.debounce_seconds) {
              items[items.length - 1] = {...last, text: last.text + ' ' + (p.display_text||p.text),display_text:(last.display_text||last.text)+' '+(p.display_text||p.text),system_label:p.system_label, ended_at: p.ended_at};
            } else items.push(p);
            return {...m, interjections: items,event_sequence:event.sequence};
          }));
        }
        if (event.type === 'chat.input') setMessages(old => mergeHistory(old,[{id: `${event.turn_id}:user`, role: 'user', text: p.text,source:p.source,conversation_id:p.conversation_id,timestamp:event.timestamp,session_id:p.conversation_id||sessionID.current,event_sequence:event.sequence}]));
        if (event.type === 'model.started') setBusy(true);
        if (event.type === 'chat.delta' || event.type === 'chat.completed') {
          setMessages(old => {
            const index = old.findIndex(m => m.id === event.turn_id);
            const previous = index < 0 ? '' : old[index].text;
            const message = {...(index < 0 ? {} : old[index]), id: event.turn_id, role: 'assistant',source:p.source,conversation_id:p.conversation_id,timestamp:old[index]?.timestamp||event.timestamp,session_id:p.conversation_id||sessionID.current,event_sequence:event.sequence, text: event.type === 'chat.completed' ? p.text : previous + p.text};
            return index < 0 ? [...old, message] : old.map((m, i) => i === index ? message : m);
          });
        }
        if (['chat.completed', 'chat.cancelled', 'model.error'].includes(event.type)) setBusy(false);
        if (['chat.completed', 'chat.cancelled', 'model.error'].includes(event.type)) loadHistory();
        if (['speech.error','speech.unavailable'].includes(event.type)) setSpeechError(p.error);
        else if (event.type.endsWith('.error')) setError(p.error);
        if (event.type === 'speech.started') setSpeechError('');
    }, connected => {setConnected(connected); if (!connected) setBusy(false);else loadHistory(false,true);});
    return()=>{alive.current=false;disconnect();};
  }, []);
  async function send(e) {
    e.preventDefault();
    if (!text.trim() || busy) return;
    setBusy(true); setError('');
    const value = text; setText('');
    try {
      const response = await fetch('http://127.0.0.1:8765/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text: value})});
      if (!response.ok) throw new Error(await response.text());
    } catch (error) {setError(error.message);} finally {setBusy(false);}
  }
  async function stop() {
    try {
      const response = await fetch('http://127.0.0.1:8765/api/chat/stop', {method: 'POST'});
      if (!response.ok) throw new Error('Unable to stop response');
    } catch (error) {setError(error.message);}
  }
  return <main className="control stream-chat">
    <header className="conversation-heading"><h1>{characterName||'Chat'}</h1><div className="conversation-indicators"><span className={'connection-badge '+(connected&&!startupError?'online':'')}><i/>{startupError?'Setup needed':connected?busy?'Generating…':'Connected':'Connecting…'}</span><DiscordLaunch disabled={!connected||!!startupError} onError={setError}/></div></header>
    {startupError&&<div className="notice error" role="alert"><div><strong>Your runtime needs attention</strong><p>{startupError}</p></div><button onClick={onSettings}><Settings size={16}/>Open settings</button></div>}
    <div className="chat-status">
      <button className="quiet-button" disabled={!connected||!!startupError} onClick={stop}><Square size={14}/>Stop reply</button>
      <button className="quiet-button" onClick={()=>{follow.current=true;if(list.current)list.current.scrollTop=list.current.scrollHeight;}}><ArrowDown size={14}/>Latest</button></div>
    {error && <p role="alert">{error}</p>}
    <DiscordNotifications/>
    {speechError && <p role="alert">Voice output unavailable. Chat remains available. Speech retries on the next reply. {speechError}</p>}
    <section ref={list} className="chat" onScroll={e=>{const element=e.currentTarget;follow.current=element.scrollHeight-element.clientHeight-element.scrollTop<60;if(element.scrollTop<80&&hasMore&&!loadingRef.current)loadHistory(true);}}>
      {hasMore&&<button disabled={loading} onClick={()=>loadHistory(true)}>{loading?'Loading…':'Load older messages'}</button>}
       {visibleMessages.map((message,index)=><React.Fragment key={message.id}>
        {message.session_id!==visibleMessages[index-1]?.session_id&&<details className="session-marker"><summary>{message.source==='discord'?'Discord · ':''}{message.session_id==='legacy'?'Imported conversation':`Session · ${sessions[message.session_id]?.started ? new Date(sessions[message.session_id].started*1000).toLocaleString() : 'Current'}`} {sessions[message.session_id]?.provider}</summary>
          <p>{sessions[message.session_id]?.outcome || 'running'} · {sessions[message.session_id]?.ended ? 'Ended '+new Date(sessions[message.session_id].ended*1000).toLocaleString() : 'No recorded end time'}</p>
        </details>}
          <Message message={message} compact pendingTranscript={inlinePopup?transcript.text:''} pendingStartedAt={transcript.startedAt}/></React.Fragment>)}
     </section><div className="conversation-dock">
      {(preferences.quickActions?.length>0||taskActions.length>0)&&<div className="quick-actions">{(preferences.quickActions||[]).filter(action=>action.label?.trim()&&action.prompt?.trim()).map((action,index)=><button key={action.id||index} onClick={()=>setText(action.prompt)}>{action.label}</button>)}{taskActions.map(task=><button key={task.id} onClick={()=>setText(`Help me work on task ${task.id}. Retrieve its details using the task tools.`)} title={task.title}>{task.title}</button>)}</div>}
        <div className={'composer-slot'+(inlinePopup?' popup-replaces-input':'')}><form id="chat-message-form" className="composer" onSubmit={send}><input aria-label="Message" placeholder="Message" value={text} tabIndex={inlinePopup?-1:undefined} onChange={e => setText(e.target.value)}/></form>
        <SpeechPopup inline sending={transcript.phase==='sending'} prefix="transcript" text={transcript.text} visible={inlinePopup} preferences={preferences}/></div>
       <DockControls collapsed={collapsedMode} generating={busy} preferences={preferences} form={!collapsedMode?'chat-message-form':undefined} canSend={connected&&!busy&&!startupError&&!!text.trim()} disabled={!connected||!!startupError}/>
    </div></main>;
}

export function Message({message,compact=false,pendingTranscript='',pendingStartedAt=0}) {
  message={...message,compact};
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
}

function ReplyPart({message, start, end}) {
  const cutoff = message.interrupted && Number.isFinite(message.cutoff) ? message.cutoff : message.text.length;
  const split = Math.max(start, Math.min(end, cutoff));
  return <div className={'msg ' + message.role} style={{whiteSpace: 'pre-wrap'}}>
    {message.source&&<small className="source-tag">{message.source==='discord'?'Discord':message.source==='microphone'?'Microphone':'Message'}</small>}
    <FormattedText text={message.text.slice(start, split)}/>
    {split < end && <div style={{color: '#96919e', opacity: 0.55}} title="Generated, but not spoken"><FormattedText text={message.text.slice(split, end)}/></div>}
    {message.compact&&message.timestamp&&<small className="bubble-time">{new Date(typeof message.timestamp==='number'?message.timestamp*1000:message.timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</small>}
  </div>;
}
