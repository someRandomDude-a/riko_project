import React,{useEffect,useState} from 'react';
import {request} from './api.mjs';
import useResource from './use_resource.jsx';

export default function AvatarModelSettings({model,format,onChange,onUse,busy}){
  const [library,setLibrary,error,setError]=useResource('avatar_models');
  const [importing,setImporting]=useState(false),[path,setPath]=useState('');
  async function refresh(){try{setLibrary(await request('/api/avatar/models'));setError('');}catch(e){setError(e.message);}}
  useEffect(()=>{refresh();},[]);
  async function choose(){
    try{
      if(!window.riko?.pickPath)throw new Error('Enter a source path when using the browser; the file picker requires Electron.');
      const value=await window.riko.pickPath({extensions:['vrm']});if(value)setPath(value);
    }catch(e){setError(e.message);}
  }
  async function importAndUse(){
    setImporting(true);setError('');
    try{
      const result=await request('/api/avatar/models/import',{method:'POST',body:{value:path}});
      setPath('');await refresh();
      onChange('avatar.model',result.path);onChange('avatar.format','auto');
      await onUse(result.path,'auto');
    }catch(e){setError(e.message);}finally{setImporting(false);}
  }
  const disabled=busy||importing, entries=library?.entries||[];
  return <section className="avatar-model-settings"><h3>Avatar model & format</h3>
    <p className="caption">Choose a .vrm from character_files/models, or import your own. Imports copy the original intact without replacing existing files. Model changes apply to the desktop immediately when saved.</p>
    {error&&<p role="alert">{error}</p>}
    <div className="settings-grid">
      <label className="setting-field"><span>VRM model</span><select disabled={disabled} value={model||''} onChange={e=>onChange('avatar.model',e.target.value)}>
        {!entries.some(entry=>entry.path===model)&&<option value={model||''}>{model||'Choose a model…'} (current configuration)</option>}
        {entries.map(entry=><option key={entry.path} value={entry.path}>{entry.name}</option>)}
      </select></label>
      <label className="setting-field"><span>Model format</span><select disabled={disabled} value={format||'auto'} onChange={e=>onChange('avatar.format',e.target.value)}>
        <option value="auto">Auto-detect (recommended)</option><option value="vrm0">VRM 0.x</option><option value="vrm1">VRM 1.0</option>
      </select><small>Must match the file. This setting does not convert models.</small></label>
    </div>
    <button disabled={disabled||!model} onClick={async()=>{setError('');try{await onUse(model,format||'auto');}catch(e){setError(e.message);}}}>Use selected model</button>
    <button disabled={disabled} onClick={refresh}>Refresh models</button>
    <div className="settings-grid"><label className="setting-field"><span>Import .vrm file</span><input disabled={disabled} value={path} onChange={e=>setPath(e.target.value)} placeholder="Source file path"/><button disabled={disabled} onClick={choose}>Choose VRM file</button></label></div>
    <button disabled={disabled||!path.trim()} onClick={importAndUse}>{importing?'Importing…':'Import & use model'}</button>
    <p className="caption">Self-contained VRM 0.x and VRM 1.0 GLB files, up to 128 MiB. Other formats must be converted first. Your source file is not moved or modified; unrelated unsaved settings are preserved.</p>
  </section>;
}
