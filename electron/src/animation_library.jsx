import React,{useState} from 'react';
import {request} from './api.mjs';
import useResource from './use_resource.jsx';
import AvatarModelSettings from './avatar_model_settings.jsx';
import Explanation from './explanation.jsx';

const states=['idle','listening','thinking','speaking','tool','sleeping','held','clicked','settling','walking'];
export default function AnimationLibraryPanel({avatarModel,avatarFormat,onAvatarChange,onAvatarUse,settingsBusy}){
  const [status,setStatus,error,setError]=useResource('animation');
  const [busy,setBusy]=useState(false);
  const [path,setPath]=useState(''),[name,setName]=useState(''),[license,setLicense]=useState(''),[selected,setSelected]=useState('idle');
  const [x,setX]=useState('0'),[y,setY]=useState('0');
  const [drafts,setDrafts]=useState({});
  async function refresh(){try{const data=await request('/api/animation');setStatus(data);setError('');}catch(e){setError(e.message);}}
  async function run(fn){setBusy(true);setError('');try{await fn();}catch(e){setError(e.message);}finally{setBusy(false);}}
  async function choose(){try{const value=await window.riko?.pickPath?.({});if(value)setPath(value);}catch(e){setError(e.message);}}
  const update=(entry,key,value)=>setDrafts(old=>({...old,[entry.id]:{...(old[entry.id]||{}),[key]:value}}));
   return <section className="panel animation-library"><AvatarModelSettings model={avatarModel} format={avatarFormat} onChange={onAvatarChange} onUse={onAvatarUse} busy={settingsBusy}/><h3>Animations</h3>
     <Explanation simple="Add an animation file, then choose when the character uses it. Your original file stays unchanged."><p className="caption">Supported formats: self-contained .vrma clips and .pose.json files. Convert other animation formats to VRMA before importing.</p></Explanation>
    {error&&<p role="alert">{error}</p>}
    {status&&<p>State: <strong>{status.state?.mode||'Initializing'}</strong> · Intent: {status.intent?.intent_id||'Procedural fallback'} · Policy: {status.intent?.source||'rules'} · Rig: {status.capabilities?.bones?.length||0} bones {status.error&&<span role="alert">· {status.error}</span>}</p>}
    <div className="settings-grid"><label className="setting-field"><span>Source file</span><input value={path} onChange={e=>setPath(e.target.value)}/><button onClick={choose}>Choose file</button></label>
      <label className="setting-field"><span>Name</span><input value={name} onChange={e=>setName(e.target.value)}/></label>
      <label className="setting-field"><span>Source/license notes</span><input value={license} onChange={e=>setLicense(e.target.value)}/></label>
      <label className="setting-field"><span>Automatic state (looping base)</span><select value={selected} onChange={e=>setSelected(e.target.value)}><option value="">Preview only</option>{states.map(state=><option key={state}>{state}</option>)}</select></label></div>
    <button disabled={busy||!path||!status} onClick={()=>run(async()=>{await request('/api/animation/import',{method:'POST',body:{path,metadata:{...(name?{name}:{}),license,states:selected?[selected]:[]}}});setPath('');setName('');})}>Import asset</button>
    <button disabled={busy} onClick={refresh}>Refresh</button>
    {(status?.entries||[]).map(entry=>{
      const draft={...entry,...drafts[entry.id]};
      const missing=(entry.mask?.length?entry.mask:entry.bones).filter(bone=>!(status.capabilities?.bones||[]).includes(bone));
      return <article className="card" key={entry.id}><h4>{entry.name} · {entry.kind}</h4>
        <p className="caption">{entry.id} · {entry.bones.length} bone tracks{missing.length?` · Missing: ${missing.join(', ')}`:''}</p>
        <div className="settings-grid"><label className="setting-field"><span>States (comma separated)</span><input value={draft.states.join(', ')} onChange={e=>update(entry,'states',e.target.value.split(',').map(v=>v.trim()).filter(Boolean))}/></label>
          <label className="setting-field"><span>Emotions (empty matches all)</span><input value={draft.emotions.join(', ')} onChange={e=>update(entry,'emotions',e.target.value.split(',').map(v=>v.trim()).filter(Boolean))}/></label>
          <label className="setting-field"><span>Bone mask (empty uses authored tracks)</span><input value={draft.mask.join(', ')} onChange={e=>update(entry,'mask',e.target.value.split(',').map(v=>v.trim()).filter(Boolean))}/></label>
          <label className="setting-field"><span>Layer</span><select value={draft.layer} onChange={e=>update(entry,'layer',e.target.value)}><option>base</option><option>gesture</option></select></label>
          <label className="setting-field"><span>Speed</span><input type="number" min="0.25" max="3" step="0.05" value={draft.speed} onChange={e=>update(entry,'speed',Number(e.target.value))}/></label>
          <label className="setting-field"><span>Transition seconds</span><input type="number" min="0.05" max="3" step="0.05" value={draft.transition_seconds} onChange={e=>update(entry,'transition_seconds',Number(e.target.value))}/></label></div>
        <label className="switch-row">Loop<input type="checkbox" checked={draft.loop} onChange={e=>update(entry,'loop',e.target.checked)}/></label>
        <button disabled={busy||!!missing.length} onClick={()=>run(()=>request('/api/animation/assets/'+entry.id+'/preview',{method:'POST'}))}>Preview</button>
        <button disabled={busy||!drafts[entry.id]} onClick={()=>run(async()=>{await request('/api/animation/assets/'+entry.id,{method:'PATCH',body:drafts[entry.id]});setDrafts(old=>{const next={...old};delete next[entry.id];return next;});})}>Save assignment</button>
        <p className="caption">{entry.license||'No license note recorded. Use assets you have permission to use.'}</p></article>;
    })}
     <h4>Walking</h4><Explanation simple="Choose a destination and select Walk. Dragging the character stops its walk."><p className="caption">Coordinates use display pixels. Imported walking clips can replace the built-in walking motion.</p></Explanation>
    <div className="settings-grid"><label className="setting-field"><span>Destination x</span><input type="number" value={x} onChange={e=>setX(e.target.value)}/></label><label className="setting-field"><span>Destination y</span><input type="number" value={y} onChange={e=>setY(e.target.value)}/></label></div>
    <button disabled={busy||!status} onClick={()=>run(()=>{if(!x.trim()||!y.trim()||!Number.isInteger(Number(x))||!Number.isInteger(Number(y)))throw new Error('Enter integer coordinates');return request('/api/animation/walk',{method:'POST',body:{x:Number(x),y:Number(y)}});})}>Walk to destination</button>
    <button onClick={()=>run(()=>request('/api/animation/stop',{method:'POST'}))}>Stop motion</button>
  </section>;
}
