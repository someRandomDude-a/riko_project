import React,{useEffect,useId,useRef,useState} from 'react';
import {Search,X,ChevronRight,Save} from './ui/icons.jsx';
import {searchSettings,localSettingValue,localSettingPatch} from './settings_search.mjs';

export function LocalSettingEditor({item,preferences,catalog,onChange}){
 const value=localSettingValue(item,preferences,catalog),[draft,setDraft]=useState(value??''),[error,setError]=useState(''),id=useId();
 useEffect(()=>{setDraft(value??'');setError('');},[value]);
 function edit(input){setDraft(input);const result=localSettingPatch(item,input,preferences,catalog);setError(result.error||'');if(result.changes)onChange(result.changes);}
 return <div className="setting-field"><label htmlFor={id}>{item.label}</label>
  {item.kind==='boolean'?<input id={id} type="checkbox" checked={!!value} onChange={e=>edit(e.target.checked)}/>:item.options?<select id={id} value={value} onChange={e=>edit(item.kind==='number'?Number(e.target.value):e.target.value)}>{item.options.map(option=><option key={option} value={option}>{option}</option>)}</select>:<input id={id} type={item.kind==='number'?'number':'text'} value={draft} min={item.min} max={item.max} step={item.integer?1:'any'} aria-invalid={!!error} onChange={e=>edit(e.target.value)}/>}
  {item.help&&<p className="caption">{item.help}</p>}{error&&<small className="field-error" role="alert">{error}</small>}
 </div>;
}

export function SettingsSearchResults({matches,active,labels,onNavigate,renderRuntime,preferences,catalog,updatePreferences}){
 return matches.map((item,i)=><article className={'settings-search-result'+(i===active?' selected':'')} key={item.id}>
  <button type="button" className="settings-search-jump" aria-label={'Open '+item.label} onClick={()=>onNavigate(item)}><span><strong>{item.label}</strong><small>{labels[item.group]||item.group} · {item.section||'Settings'}{item.advanced?' · Advanced':''}</small></span><ChevronRight size={17}/></button>
  {item.kind!=='section'&&!item.readonly&&<details><summary>Edit here · {item.source==='local'?'applies immediately':'runtime draft'}</summary>{item.source==='runtime'?renderRuntime(item,'search-setting-'):<LocalSettingEditor item={item} preferences={preferences} catalog={catalog} onChange={updatePreferences}/>}</details>}
  {item.readonly&&<small className="caption">Read-only · open to inspect</small>}
 </article>);
}

export default function SettingsSearchMenu({items,groups,onNavigate,renderRuntime,preferences,catalog,updatePreferences,onSave,canSave,saving,status}){
 const dialog=useRef(null),input=useRef(null),[open,setOpen]=useState(false),[query,setQuery]=useState(''),[active,setActive]=useState(0),id=useId();
 const matches=searchSettings(items,query,groups),visible=matches.slice(0,80),labels=Object.fromEntries(groups),index=Math.min(active,Math.max(0,visible.length-1));
 useEffect(()=>{if(open&&!dialog.current.open){dialog.current.showModal();input.current?.focus();}},[open]);
 useEffect(()=>{const shortcut=e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();setOpen(true);input.current?.focus();}};document.addEventListener('keydown',shortcut);return()=>document.removeEventListener('keydown',shortcut);},[]);
 useEffect(()=>{setActive(0);},[query]);
 useEffect(()=>()=>dialog.current?.close(),[]);
 function close(){dialog.current?.close();setOpen(false);}
 function navigate(item){close();onNavigate(item);}
 function keydown(e){if(['ArrowDown','ArrowUp'].includes(e.key)){e.preventDefault();const next=Math.max(0,Math.min(visible.length-1,index+(e.key==='ArrowDown'?1:-1)));setActive(next);dialog.current?.querySelectorAll('.settings-search-result')[next]?.scrollIntoView({block:'nearest'});}else if(e.key==='Enter'&&visible[index]){e.preventDefault();navigate(visible[index]);}}
 return <><button type="button" className="settings-search-trigger" aria-label="Open settings search menu" aria-haspopup="dialog" onClick={()=>setOpen(true)}><Search size={18}/><span>Search settings…</span><kbd>Ctrl K</kbd></button>
  <dialog ref={dialog} className="settings-search-menu" aria-labelledby={id+'-title'} onClose={()=>setOpen(false)} onClick={e=>{if(e.target===dialog.current){const r=dialog.current.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)close();}}}>
   {open&&<><header><h2 id={id+'-title'}>Search settings</h2><button className="icon-button" type="button" aria-label="Close settings search" onClick={close}><X size={18}/></button></header>
    <div className="settings-search-query"><Search size={18}/><input ref={input} type="search" aria-label="Search all settings" aria-controls={id+'-results'} placeholder="Find a setting, bone or category…" value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={keydown}/></div>
    <p className="caption">Open a setting or expand Edit here. Local preferences apply immediately; runtime edits stay in your draft until saved.</p>
    <div id={id+'-results'} className="settings-search-results" aria-label="Matching settings"><p role="status">{matches.length} {query?'matches':'categories and sections'}{matches.length>visible.length?' · Showing the first 80; narrow your search for more.':''}</p>{!matches.length&&<p>No matching settings. Try another word.</p>}<SettingsSearchResults matches={visible} active={index} labels={labels} onNavigate={navigate} renderRuntime={renderRuntime} preferences={preferences} catalog={catalog} updatePreferences={updatePreferences}/></div>
    <footer><span role="status">{status}</span><button type="button" onClick={close}>Done</button><button type="button" disabled={!canSave||saving} onClick={onSave}><Save size={16}/>{saving?'Saving…':'Save runtime settings'}</button></footer>
   </>}
  </dialog>
 </>;
}
