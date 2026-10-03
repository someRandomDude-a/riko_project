import React, {useState} from 'react';
import {Mic, Volume2, Moon} from './ui/icons.jsx';
import Whiteboard from './whiteboard.jsx';
import useRuntime from './use_runtime.jsx';
import {request} from './api.mjs';

export default function Workspace() {
  const state = useRuntime();
  const [tab, setTab] = useState('board'), [error, setError] = useState('');
  async function toggle(path) {
    try {await request(path, {method: 'POST'}); setError('');}
    catch (error) {setError(error.message);}
  }
  return <main className="control workspace"><header className="page-heading"><h1>Canvas</h1><button onClick={()=>window.riko?.showWhiteboard()}>Open separate window</button></header>
    <nav className="tabs">{['board', 'tools'].map(name =>
      <button className={tab===name?'active':''} key={name} onClick={() => setTab(name)}>{name==='board'?'Whiteboard':'Tool activity'}</button>)}</nav>
    {error && <p role="alert">{error}</p>}
    {tab === 'board' && <Whiteboard commands={state.whiteboard} pages={state.whiteboard_pages} modelPage={state.whiteboard_page} loaded={!!state.whiteboard_pages}/>}
    {tab === 'tools' && <section className="panel"><h2>Tool activity</h2>{(state.tools || []).map((tool, index) =>
      <div className="card" key={index}><b>{tool.name}</b><span className={'pill ' + tool.status}>{tool.status}</span>
        <pre>{JSON.stringify(tool.arguments, null, 2)}</pre>{tool.result && <pre>{tool.result}</pre>}</div>)}</section>}
    <nav className="actions">
      <button onClick={() => toggle('/api/mic/toggle')}><Mic/> Mic</button>
      <button onClick={() => toggle('/api/audio/toggle')}><Volume2/> Audio</button>
      <button onClick={() => toggle('/api/sleep/toggle')}><Moon/> Sleep</button>
    </nav>
  </main>;
}
