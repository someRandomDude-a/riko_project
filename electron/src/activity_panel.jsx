import React, {useEffect, useState} from 'react';
import {Activity, X, Wrench, Brain, Radio, Trash2} from './ui/icons.jsx';
import {connectEvents} from './event_connection.mjs';
import FormattedText from './formatted_text.jsx';
import useRuntime from './use_runtime.jsx';

export default function ActivityPanel({preferences, updatePreferences}) {
  const state=useRuntime();
  const [events,setEvents]=useState([]), [reasoning,setReasoning]=useState({});
  useEffect(()=>connectEvents(event=>{
    if(event.type==='model.reasoning'&&preferences.reasoning)setReasoning(old=>{
      const id=event.turn_id||'current';const next={...old,[id]:((old[id]||'')+(event.payload.text||'')).slice(-60000)};
      const ids=Object.keys(next);if(ids.length>8)delete next[ids[0]];return next;
    });
    if(preferences.system&&/^(runtime\.|memory\.|initiative\.|model\.)/.test(event.type)&&event.type!=='model.reasoning')setEvents(old=>[event,...old].slice(0,80));
  }),[preferences.reasoning,preferences.system]);
  return <aside className="activity-panel" aria-label="Model activity">
    <header><div><Activity size={19}/><h2>Activity</h2></div><button className="icon-button" aria-label="Close activity" onClick={()=>updatePreferences({activity:false})}><X size={18}/></button></header>
    <div className="activity-filters">{[['tools','Tools',Wrench],['reasoning','Reasoning',Brain],['system','Events',Radio]].map(([key,label,Icon])=><button className={preferences[key]?'active':''} aria-pressed={preferences[key]} key={key} onClick={()=>updatePreferences({[key]:!preferences[key]})}><Icon size={14}/>{label}</button>)}</div>
    <div className="activity-scroll">
      {preferences.tools&&<section><h3>Tool calls <span>{(state.tools||[]).length}</span></h3>{!(state.tools||[]).length&&<p className="subtle-empty">Tool calls will appear here as they happen.</p>}
        {(state.tools||[]).map((tool,index)=><details className="activity-card" key={tool.id||index}><summary><span className={'status-dot '+tool.status}/><strong>{tool.name}</strong><span className="tool-status">{tool.status}</span></summary><small>{tool.duration_ms!=null?tool.duration_ms+' ms':''}</small><h4>Arguments</h4><pre>{JSON.stringify(tool.arguments,null,2)}</pre>{tool.result!=null&&<><h4>Result</h4><pre>{tool.result}</pre>{tool.result_truncated&&<small>Result preview truncated to 8,000 characters.</small>}</>}</details>)}
      </section>}
      {preferences.reasoning&&<section><h3>Provider reasoning</h3><p className="caption">Only reasoning explicitly returned by the provider. Never inferred private thoughts. Captured while this toggle is on; not saved to chat history.</p>{!Object.keys(reasoning).length&&<p className="subtle-empty">No reasoning stream received yet. Some models and providers do not expose one.</p>}{Object.entries(reasoning).reverse().map(([id,text])=><details className="activity-card" key={id}><summary>Reasoning · {id.slice(0,8)}</summary><FormattedText text={text}/></details>)}</section>}
      {preferences.system&&<section><h3>Runtime events <button className="icon-button" aria-label="Clear captured events" onClick={()=>{setEvents([]);setReasoning({});}}><Trash2 size={14}/></button></h3>{!events.length&&<p className="subtle-empty">Listening for new events…</p>}{events.map(event=><details className="activity-card" key={event.id}><summary>{event.type}</summary><pre>{JSON.stringify(event.payload,null,2)}</pre></details>)}</section>}
    </div>
  </aside>;
}
