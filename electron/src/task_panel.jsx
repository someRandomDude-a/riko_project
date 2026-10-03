import React, {useEffect, useState} from 'react';
import FormattedText from './formatted_text.jsx';
import useResource from './use_resource.jsx';
import Explanation from './explanation.jsx';

const API = 'http://127.0.0.1:8765';

export default function TaskPanel() {
  const [tasks, setTasks] = useState([]);
  const [taskState]=useResource('tasks');
  const [selected, setSelected] = useState(null);
  const [history, setHistory] = useState([]);
  const [title, setTitle] = useState('');
  const [query, setQuery] = useState('');
  const [reason, setReason] = useState('User correction');
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  async function request(path, options) {
    const response = await fetch(API + path, options);
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Request failed');
    return body;
  }
  async function load() {
    try {setTasks((await request('/api/tasks?query=' + encodeURIComponent(query))).tasks); setError('');}
    catch (exc) {setError(exc.message);}
  }
   useEffect(()=>{if(taskState?.tasks&&!query)setTasks(taskState.tasks);},[taskState,query]);
  async function inspect(task) {
    try {
      const record = await request('/api/tasks/' + task.id);
      setSelected(record); setHistory(record.history); setError('');
    } catch (exc) {setError(exc.message);}
  }
  async function create() {
    setPending(true);
    try {
      await request('/api/tasks', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({title})});
      setTitle(''); await load();
    } catch (exc) {setError(exc.message);} finally {setPending(false);}
  }
  async function save() {
    setPending(true);
    try {
      const changes = Object.fromEntries(['title', 'description', 'status', 'progress', 'next_step', 'blocker'].map(key => [key, selected[key]]));
      await request('/api/tasks/' + selected.id, {method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({expected_revision: selected.revision, changes, reason})});
      await inspect(selected); await load();
    } catch (exc) {setError(exc.message);} finally {setPending(false);}
  }
   return <section className="panel control task-page">
    <h2>Tasks</h2>
     <Explanation simple="Keep track of your goals and next steps. Pause a task when you do not want reminders."><p className="caption">Chat tools use the same task list. Completed and dismissed tasks do not trigger proactive help. Every update keeps a change history.</p></Explanation>
    {error && <p role="alert">{error}</p>}
     <input aria-label="New task" maxLength="300" value={title} onChange={e => setTitle(e.target.value)} placeholder="What would you like to get done?"/>
    <button disabled={pending || !title.trim()} onClick={create}>Create task</button>
    <button disabled={pending} onClick={load}>Refresh list</button>
     <input aria-label="Search tasks" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search your tasks…"/>
    <button disabled={pending} onClick={load}>Search tasks</button>
    {tasks.map(task => <div className="card" key={task.id}>
      <FormattedText text={task.title}/><p>{task.status} · {Math.round(task.progress * 100)}% · revision {task.revision}</p>
      {task.blocker && <FormattedText text={'Blocker: '+task.blocker}/ >}
      {task.next_step && <FormattedText text={'Next: '+task.next_step}/ >}
       <button disabled={pending} onClick={() => inspect(task)}>Edit task</button>
    </div>)}
    {selected && <fieldset>
      <legend>Edit task · revision {selected.revision}</legend>
      {['title', 'description', 'next_step', 'blocker'].map(key => <label key={key} style={{display: 'block'}}>{key.replace('_', ' ')}<textarea value={selected[key]} onChange={e => setSelected({...selected, [key]: e.target.value})}/></label>)}
      <label>Status<select value={selected.status} onChange={e => setSelected({...selected, status: e.target.value, progress: e.target.value === 'completed' ? 1 : selected.progress === 1 ? 0 : selected.progress})}>{['active', 'blocked', 'paused', 'completed', 'dismissed'].map(status => <option key={status}>{status}</option>)}</select></label>
      <label>Progress (0–1)<input type="number" min="0" max="1" step="0.05" value={selected.progress} onChange={e => setSelected({...selected, progress: Number(e.target.value)})}/></label>
      <label>Reason<input value={reason} onChange={e => setReason(e.target.value)}/></label>
      <button disabled={pending || !reason.trim()} onClick={save}>Save correction</button>
      <button disabled={pending} onClick={() => inspect(selected)}>Reload current revision</button>
       <details><summary>Change history</summary><pre>{JSON.stringify(history, null, 2)}</pre></details>
    </fieldset>}
  </section>;
}
