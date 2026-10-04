import React, {useEffect, useMemo, useRef, useState} from 'react';
import {Save, FolderOpen, Check, RefreshCw, AlertCircle, Plus, Trash2} from './ui/icons.jsx';
import {request} from './api.mjs';
import {inputValues, settingsPatch, runtimePresets} from './settings_model.mjs';
import InitiativeSettings from './initiative_settings.jsx';
import DisplaySettings from './display_settings.jsx';
import VoiceInput from './voice_input.jsx';
import AnimationLibraryPanel from './animation_library.jsx';
import GPUResources from './gpu_resources.jsx';
import {ToolApprovalSettings} from './tool_approvals.jsx';
import KVPoolSettings from './kv_pool_settings.jsx';
import {syncPool} from './token_budgets.mjs';
import NumericSetting from './numeric_setting.jsx';
import SettingsNavigation,{SettingsTabs} from './settings_navigation.jsx';
import Explanation from './explanation.jsx';
import {simpleHelp,simpleLabel} from './plain_language.mjs';
import AvatarStudio from './avatar_studio.jsx';
import AvatarHitSettings from './avatar_hit_settings.jsx';
import {useAvatarEditorOpen} from './avatar_editor_channel.mjs';
import GraphicsSettings from './graphics_settings.jsx';
import SettingsSearchMenu from './settings_search.jsx';
import DiscordSettings from './discord_settings.jsx';
import NeuralSettings from './neural_settings.jsx';
import {settingsIndex,focusSetting} from './settings_search.mjs';
import {CATALOG_KEY,readCatalog} from './avatar_studio_settings.mjs';

const groups = [['models','Models'],['neural','Custom neural network settings'],['performance','Performance & logs'],['voice','Microphone'],['speech','Speaking'],['memory','Memory'],['initiative','Initiative'],['tools','Tools & tasks'],['discord','Discord'],['appearance','Desktop model'],['graphics','Graphics'],['character','Character'],['interface','Chat layout']];
const descriptions = {models:'Choose your model and balance speed with memory use.',voice:'Choose your microphone and how to start listening.',speech:'Choose how the character speaks.',memory:'Choose what the character remembers and how it responds.',initiative:'Choose when the character can start a conversation.',tools:'Manage tools, permissions and tasks.',appearance:'Choose your avatar and where it appears.',graphics:'Tune avatar rendering quality and GPU cost.',character:'Set the character’s name and instructions.',interface:'Choose what you see in chat.'};

export default function SettingsPage({preferences, updatePreferences, onDirty}) {
  const [snapshot,setSnapshot]=useState(null), [inputs,setInputs]=useState({}), [group,setGroup]=useState('models');
  useAvatarEditorOpen(group==='appearance');
  const [advanced,setAdvanced]=useState(false), [serverErrors,setServerErrors]=useState({});
  const [catalog,setCatalog]=useState(readCatalog),[destination,setDestination]=useState(null);
  useEffect(()=>{const refresh=()=>setCatalog(readCatalog()),storage=e=>{if(e.key===CATALOG_KEY)refresh();};window.addEventListener('storage',storage);window.addEventListener('avatar-catalog',refresh);return()=>{window.removeEventListener('storage',storage);window.removeEventListener('avatar-catalog',refresh);};},[]);
  const [error,setError]=useState(''), [notice,setNotice]=useState(''), [busy,setBusy]=useState(false), [validating,setValidating]=useState(false);
  const [repos,setRepos]=useState([]), [files,setFiles]=useState([]), [hfNotice,setHfNotice]=useState(''), [devices,setDevices]=useState([]), [pathChecks,setPathChecks]=useState({});
  const [modelSource,setModelSource]=useState('huggingface');
  const generation=useRef(0);
  const page=useRef(null),sectionRefs=useRef(new Map());
  function selectGroup(id){setGroup(id);setDestination(null);page.current?.scrollIntoView({block:'start'});}
  function navigateSetting(item){setGroup(item.target.group);if(item.source==='runtime')setAdvanced(true);setDestination({...item.target});}
  useEffect(()=>{
    if(!destination||!snapshot)return;
    let second;const first=requestAnimationFrame(()=>{if(destination.boneId)window.dispatchEvent(new CustomEvent('avatar-select-bone',{detail:destination.boneId}));second=requestAnimationFrame(()=>{if(page.current)focusSetting(page.current,destination);});});
    return()=>{cancelAnimationFrame(first);if(second)cancelAnimationFrame(second);};
  },[destination,!!snapshot]);
  function jump(section){const target=sectionRefs.current.get(section);target?.scrollIntoView({block:'start',behavior:preferences.reduceMotion?'instant':'auto'});target?.focus({preventScroll:true});}
  const [modelQuery,setModelQuery]=useState(''),[modelGGUF,setModelGGUF]=useState(true);
  const parsed=useMemo(()=>snapshot?settingsPatch(snapshot.fields,snapshot.values,inputs):{changes:{},errors:{}},[snapshot,inputs]);
  const sourceDirty=!!snapshot&&inputs['runtime.provider']==='llama_cpp'&&modelSource!==(snapshot.values['runtime.model_path']?'local':'huggingface');
  const dirty=sourceDirty || Object.keys(parsed.changes).length>0 || Object.keys(parsed.errors).length>0;
  useEffect(()=>{onDirty?.(dirty);},[dirty,onDirty]);
  useEffect(()=>()=>onDirty?.(false),[onDirty]);
  useEffect(()=>{
    if(!dirty)return;
    const handler=event=>{event.preventDefault();event.returnValue='';};
    window.addEventListener('beforeunload',handler);
    return()=>window.removeEventListener('beforeunload',handler);
  },[dirty]);
  const errors={...serverErrors,...parsed.errors};
  if(inputs['runtime.provider']==='llama_cpp'&&modelSource==='local'&&!inputs['runtime.model_path'])errors['runtime.model_path']='Choose a local GGUF file';
  if(inputs['runtime.provider']==='llama_cpp'&&!inputs['runtime.native_library'])errors['runtime.native_library']='Choose a compatible riko-native library; the HTTP server backend is no longer supported';
  if(inputs['runtime.provider']==='llama_cpp'&&inputs['runtime.model_path']&&!String(inputs['runtime.model_path']).toLowerCase().endsWith('.gguf'))errors['runtime.model_path']='Choose a .gguf model file';
  if(inputs['runtime.provider']==='llama_cpp'&&!inputs['runtime.model_path']&&files.length&&inputs['runtime.hf_filename']&&!files.includes(inputs['runtime.hf_filename']))errors['runtime.hf_filename']='File is not in this repository/revision listing';
  async function load() {
    setBusy(true);setError('');
    try {const value=await request('/api/settings');setSnapshot(value);setInputs(inputValues(value));setModelSource(value.values['runtime.model_path']?'local':'huggingface');setServerErrors({});}
    catch(e){setError(e.message);}finally{setBusy(false);}
  }
  useEffect(()=>{load();request('/api/voice/devices').then(v=>setDevices(v.devices)).catch(()=>{});},[]);
  useEffect(()=>{
    if(!snapshot||!dirty||Object.keys(parsed.errors).length){setValidating(false);return;}
    const current=++generation.current;
    let alive=true;
    setValidating(true);
    const timer=setTimeout(()=>request('/api/settings/validate',{method:'POST',body:{changes:parsed.changes}})
      .then(value=>{if(alive&&current===generation.current)setServerErrors(value.errors);})
      .catch(e=>{if(alive)setError(e.message);}).finally(()=>{if(alive)setValidating(false);}),450);
    return()=>{alive=false;clearTimeout(timer);};
  },[parsed,snapshot,dirty]);
  const repo=inputs['runtime.hf_repo_id']||'';
  useEffect(()=>{
    let alive=true;
    const query=modelQuery||repo;
    const timer=setTimeout(()=>{if(query.trim().length>2)request('/api/settings/huggingface/search?query='+encodeURIComponent(query)+'&gguf='+modelGGUF)
      .then(value=>{if(alive)setRepos(value.models);}).catch(()=>{if(alive)setRepos([]);});},400);
    return()=>{alive=false;clearTimeout(timer);};
  },[repo,modelQuery,modelGGUF]);
  useEffect(()=>{
    let alive=true;setFiles([]);setHfNotice('');
    if(!/^[\w.-]+\/[\w.-]+$/.test(repo))return;
    const timer=setTimeout(()=>request('/api/settings/huggingface/files?repo='+encodeURIComponent(repo)+'&revision='+encodeURIComponent(inputs['runtime.hf_revision']||'main'))
      .then(value=>{if(alive){setFiles(value.files);setHfNotice(value.files.length+' GGUF files available'+(value.context_length?' · model context '+value.context_length.toLocaleString():''));
      }})
      .catch(()=>{if(alive)setHfNotice('Repository lookup unavailable. You can enter filenames manually.');}),350);
    return()=>{alive=false;clearTimeout(timer);};
  },[repo,inputs['runtime.hf_revision']]);
  function change(path,value) {
    if(path==='runtime.model_path')setModelSource(value?'local':'huggingface');
    setNotice('');setError('');setServerErrors(old=>({...old,[path]:undefined}));
    setInputs(old=>{const next={...old,[path]:value};
      return next;});
    setPathChecks(old=>({...old,[path]:undefined}));
  }
  async function save() {
    setBusy(true);setError('');
    try {
      const value=await request('/api/settings',{method:'PUT',body:{changes:parsed.changes,revision:snapshot.revision}});
      if(!value.saved){setServerErrors(value.errors);return;}
      setSnapshot(value);setInputs(inputValues(value));setModelSource(value.values['runtime.model_path']?'local':'huggingface');setServerErrors({});setNotice(!value.restart_required?'Saved. Changes are active now.':'Saved. Avatar and background-pause changes are active now. Restart Python for other runtime changes; restart Electron for window changes.');
    }catch(e){setError(e.message);}finally{setBusy(false);}
  }
  async function useAvatar(model,format){
    setBusy(true);setError('');setNotice('');
    try{
      const changes={'avatar.model':model,'avatar.format':format};
      const value=await request('/api/settings',{method:'PUT',body:{changes,revision:snapshot.revision}});
      if(!value.saved)throw new Error(Object.values(value.errors||{}).join('; ')||'Avatar settings could not be saved');
      setSnapshot(value);setInputs(old=>({...old,...changes}));
      setServerErrors(old=>({...old,'avatar.model':undefined,'avatar.format':undefined}));
      setNotice('Avatar model saved and applied to the desktop. Other unsaved settings were kept.');
    }finally{setBusy(false);}
  }
  async function browse(field) {
    if(!window.riko?.pickPath){setError('Native path picker is available in Electron. Enter a path manually in the browser.');return;}
    try {const value=await window.riko.pickPath({directory:field.path.endsWith('_directory')||field.path.endsWith('_dir'),defaultPath:String(inputs[field.path]||'')});if(value)change(field.path,value);}
    catch(e){setError(e.message);}
  }
  async function checkPath(field) {
    try {const value=await request('/api/settings/path',{method:'POST',body:{value:String(inputs[field.path]||'')}});setPathChecks(old=>({...old,[field.path]:value}));}
    catch(e){setError(e.message);}
  }
  function preset(name){setInputs(old=>syncPool({...old,...runtimePresets[name]}));setNotice(name==='compact'?'Compact preset requires compatible flash attention and quantized KV support. No hardware capability is inferred.':'Preset applied to your draft. Review and save when ready.');}
  const searchItems=useMemo(()=>settingsIndex(snapshot?.fields||[],groups,catalog),[snapshot?.fields,catalog]);
  const fields=(snapshot?.fields||[]).filter(field=>field.path===destination?.path||(field.group===group
    &&!['avatar.model','avatar.format','runtime.kv_pool_auto','runtime.kv_pool_tokens'].includes(field.path)
    &&(advanced||!field.advanced)
    &&(advanced||group!=='models'||inputs['runtime.provider']!=='llama_cpp'||(modelSource==='local'?!field.path.startsWith('runtime.hf_'):field.path!=='runtime.model_path'))
    &&(advanced||group!=='models'||!(inputs['runtime.provider']==='llama_cpp'?['runtime.api_mode','runtime.reuse_response_ids','runtime.base_url','runtime.api_key','runtime.model'].includes(field.path):field.path.startsWith('runtime.')&&!['runtime.provider','runtime.api_mode','runtime.reuse_response_ids','runtime.base_url','runtime.api_key','runtime.model','runtime.temperature','runtime.max_output_tokens','runtime.n_ctx','runtime.request_timeout_seconds','runtime.warmup'].includes(field.path)))));
  const sections=[...new Set(fields.map(field=>field.section||'Settings'))].sort((a,b)=>{
    const order=['Model source','Token budgets','Generation','Scheduling & startup','Compute & cache','Speech recognition model','Background models','Embedding model','Emotion model','Renderer assets','Preset fallbacks','Settings','Emotion'];return order.indexOf(a)-order.indexOf(b);
  });
   const jumpSections=[...(group==='models'?['GPU memory',...(inputs['runtime.provider']==='llama_cpp'?['Conversation cache']:[])]:[]),...(group==='appearance'?['Avatar library']:[]),...sections];
   const sectionRef=name=>node=>{if(node)sectionRefs.current.set(name,node);else sectionRefs.current.delete(name);};
   function renderField(field,prefix='setting-'){
    const id=prefix+field.path.replaceAll('.','-'),value=inputs[field.path],message=errors[field.path];
    const modelList=/model(_id)?$/.test(field.path)||field.path==='runtime.hf_repo_id';
    return <div className={'setting-field '+(field.multiline?'wide ':'')+(message?'invalid':'')} key={field.path}>
     <label htmlFor={id}>{preferences.advancedExplanations?field.technicalLabel||field.label:simpleLabel(field)}{preferences.advancedExplanations&&<span className="field-path">{field.path}</span>}</label>
     {field.kind==='boolean'?<div className="switch-row"><span>{value?'Enabled':'Disabled'}</span><input id={id} type="checkbox" role="switch" checked={!!value} disabled={field.readonly} onChange={e=>change(field.path,e.target.checked)}/></div>
      :field.path==='voice.input_device'?<select id={id} value={value} disabled={field.readonly} onChange={e=>change(field.path,e.target.value)}><option value="">Automatic microphone</option>{devices.map(d=><option key={d.index} value={d.index}>{d.name}</option>)}{value!==''&&!devices.some(d=>String(d.index)===String(value))&&<option value={value}>Device {value}</option>}</select>
      :field.options?<select id={id} value={value} disabled={field.readonly} aria-invalid={!!message} onChange={e=>change(field.path,e.target.value)}>{!field.options.includes(value)&&<option value={value}>{value||'Automatic'}</option>}{field.options.map(option=><option key={option}>{option}</option>)}</select>
      :field.kind==='number'?<NumericSetting field={field} id={id} value={value} invalid={!!message} onChange={value=>change(field.path,value)}/>
      :field.multiline?<textarea id={id} rows={field.kind==='json'?5:4} disabled={field.readonly} spellCheck={field.kind!=='json'} value={value} aria-invalid={!!message} onChange={e=>change(field.path,e.target.value)}/>
      :<div className="field-input"><input id={id} disabled={field.readonly} type={field.secret?'password':'text'} value={value} placeholder={field.nullable?'Automatic / not set':undefined} list={field.path==='runtime.hf_filename'?'gguf-files':modelList?'hf-models':undefined} aria-invalid={!!message} onFocus={()=>{if(modelList){setModelQuery(String(value));setModelGGUF(field.path==='runtime.hf_repo_id');}}} onChange={e=>{change(field.path,e.target.value);if(modelList)setModelQuery(e.target.value);}}/>{field.file&&<><button className="icon-button" title="Browse" aria-label={'Browse '+field.label} disabled={field.readonly} onClick={()=>browse(field)}><FolderOpen size={16}/></button><button className="icon-button" title="Check path" aria-label={'Check '+field.label} disabled={!value} onClick={()=>checkPath(field)}><Check size={16}/></button></>}</div>}
     <Explanation simple={simpleHelp(field)}>{field.help&&<p className="caption">{field.help}</p>}</Explanation>
     {message&&<small className="field-error" role="alert">{message}</small>}
     {pathChecks[field.path]&&<small>{pathChecks[field.path].exists||pathChecks[field.path].executable?'✓ Found':'Not found (output paths may be created later)'} · {pathChecks[field.path].resolved}</small>}
     {field.path==='runtime.hf_filename'&&<><small>{hfNotice}</small>{files.length>0&&<select aria-label="Available GGUF files" value={files.includes(value)?value:''} onChange={e=>change(field.path,e.target.value)}><option value="">Choose a GGUF…</option>{files.map(file=><option key={file}>{file}</option>)}</select>}</>}
    </div>;
   }
   return <main ref={page} className="settings-page">
     <header className="page-heading"><SettingsNavigation groups={groups} group={group} sections={jumpSections} onSelect={selectGroup} onJump={jump}/><h1>Settings</h1><button className="icon-button" title="Reload settings" aria-label="Reload settings" disabled={busy} onClick={()=>{if(!dirty||confirm('Discard unsaved changes and reload?'))load();}}><RefreshCw size={18}/></button></header>
      <div className="settings-search"><SettingsSearchMenu items={searchItems} groups={groups} onNavigate={navigateSetting} renderRuntime={renderField} preferences={preferences} catalog={catalog} updatePreferences={updatePreferences} onSave={save} canSave={!!snapshot&&dirty&&!validating&&!Object.values(errors).some(Boolean)} saving={busy} status={error||errors.__all__||notice||(validating?'Checking your changes…':dirty?'Unsaved runtime changes':'Runtime settings saved')} /><label><input type="checkbox" checked={advanced} onChange={e=>setAdvanced(e.target.checked)}/>Advanced controls</label></div>
      <SettingsTabs groups={groups} group={group} onSelect={selectGroup}/>
      {group==='discord'&&<DiscordSettings/>}
    {error&&<div className="notice error" role="alert"><AlertCircle size={18}/>{error}</div>}
    {notice&&<div className="notice" role="status"><Check size={18}/>{notice}</div>}
    {errors.__all__&&<div className="notice error" role="alert">{errors.__all__}</div>}
     {!snapshot?<div className="empty-state"><h2>Settings are not available yet</h2><p>Start the Python backend, then try again.</p><button onClick={load}>Try again</button></div>:<>
        <section id="settings-panel" role="tabpanel" aria-labelledby={'tab-'+group} className="settings-content"><div className="section-heading"><h2>{groups.find(x=>x[0]===group)?.[1]}</h2><p>{descriptions[group]}</p></div>
        {group==='models'&&<div tabIndex={-1} ref={sectionRef('GPU memory')}><GPUResources changes={parsed.changes} valid={!Object.keys(parsed.errors).length}/></div>}
        {group==='models'&&inputs['runtime.provider']==='llama_cpp'&&<div tabIndex={-1} ref={sectionRef('Conversation cache')}><KVPoolSettings values={inputs} onChange={change} onSuggest={()=>{setInputs(old=>syncPool({...old,'runtime.kv_pool_auto':true}));setNotice('Suggested size added to your draft. Save to apply it.');}}/></div>}
        {group==='models'&&inputs['runtime.provider']==='llama_cpp'&&<div className="runtime-overview"><div><span className="eyebrow">YOUR MODEL</span><strong>{inputs['runtime.parallel_slots']} request slots <span>·</span> {Number(inputs['runtime.n_ctx']||0).toLocaleString()} tokens per chat</strong><p>Choose a starting point, then adjust the values below.</p></div><div className="preset-buttons"><button onClick={()=>preset('balanced')}>Balanced</button><button onClick={()=>preset('compact')}>Less memory</button><button onClick={()=>preset('cpu')}>CPU only</button></div></div>}
       {group==='interface'&&<div id="settings-interface" className="settings-grid">{[['activity','Open activity panel'],['tools','Show tool calls'],['reasoning','Show provider reasoning'],['system','Show system events']].map(([key,label])=><label className="setting-field toggle-field" key={key}><span>{label}<small>Interface preference · applies immediately</small></span><input type="checkbox" role="switch" checked={preferences[key]} onChange={e=>updatePreferences({[key]:e.target.checked})}/></label>)}<label className="setting-field"><span>Spacing</span><select value={preferences.density} onChange={e=>updatePreferences({density:e.target.value})}><option value="comfortable">Comfortable</option><option value="compact">Compact</option></select><small>Local to this Electron installation.</small></label></div>}
       {group==='interface'&&<section id="settings-shortcuts" className="shortcut-editor"><h3>Conversation shortcuts</h3><label className="switch-row">Show active task shortcuts<input type="checkbox" role="switch" checked={preferences.showTaskActions} onChange={e=>updatePreferences({showTaskActions:e.target.checked})}/></label><p className="caption">Buttons fill the message box; nothing is sent until you choose Send. Task details are retrieved by tools only after your request.</p>{(preferences.quickActions||[]).map((action,index)=><div className="shortcut-row" key={action.id||index}><input aria-label={'Shortcut '+(index+1)+' label'} placeholder="Button label" value={action.label} maxLength={80} onChange={e=>updatePreferences({quickActions:preferences.quickActions.map((a,i)=>i===index?{...a,label:e.target.value}:a)})}/><input aria-label={'Shortcut '+(index+1)+' message'} placeholder="Message to insert" value={action.prompt} maxLength={2000} onChange={e=>updatePreferences({quickActions:preferences.quickActions.map((a,i)=>i===index?{...a,prompt:e.target.value}:a)})}/><button className="icon-button" aria-label={'Remove shortcut '+(index+1)} onClick={()=>updatePreferences({quickActions:preferences.quickActions.filter((_,i)=>i!==index)})}><Trash2 size={16}/></button></div>)}<button disabled={(preferences.quickActions||[]).length>=12} onClick={()=>updatePreferences({quickActions:[...(preferences.quickActions||[]),{id:crypto.randomUUID(),label:'',prompt:''}]})}><Plus size={16}/>Add shortcut</button></section>}
       {group==='models'&&inputs['runtime.provider']==='llama_cpp'&&<div className="model-source-switch" aria-label="Model source"><button className={modelSource==='huggingface'?'active':''} aria-pressed={modelSource==='huggingface'} onClick={()=>{setModelSource('huggingface');change('runtime.model_path','');}}>Hugging Face</button><button className={modelSource==='local'?'active':''} aria-pressed={modelSource==='local'} onClick={()=>setModelSource('local')}>Local GGUF</button></div>}
          {group==='appearance'&&<><p className="caption">Model setup, animation connections, physics and effects live here. Interface colors and speech popups remain in Appearance. Local model overrides apply immediately.</p><nav className="appearance-links"><a href="#desktop-model-library">Model & animation library</a><a href="#appearance-avatar-hits">Interaction connections</a><a href="#appearance-avatar-studio">Physics, calibration & effects</a></nav><div id="desktop-model-library" tabIndex={-1} ref={sectionRef('Avatar library')}><AnimationLibraryPanel avatarModel={inputs['avatar.model']} avatarFormat={inputs['avatar.format']} onAvatarChange={change} onAvatarUse={useAvatar} settingsBusy={busy}/></div><AvatarHitSettings preferences={preferences} update={updatePreferences}/><AvatarStudio preferences={preferences} update={updatePreferences}/></>}
        {group==='tools'&&<div id="settings-tool-approvals"><ToolApprovalSettings/></div>}
        {group==='neural'&&<NeuralSettings/>}
        {group==='models'&&inputs['runtime.provider']==='llama_cpp'&&<section id="settings-inference-transport" tabIndex={-1}><h3>How llama.cpp runs</h3><p><strong>{inputs['runtime.native_library']?'Draft: in-process native library':'Native library required — select a compatible riko-native library'}</strong> · Changes take effect after Save and a Python restart.</p><p>llama.cpp runs inside Python with one main model, reserved inference slots, streaming, cancellation, tools and prompt caching. There is no separate llama-server process or HTTP listener. A CPU-only library cannot provide CUDA acceleration. The API mode, base URL and API key fields apply to other providers, not native llama.cpp.</p></section>}
        {group==='performance'&&<section id="settings-performance" tabIndex={-1}><h3>Measure before tuning</h3><p>Tokens/second measures generation speed; first-token delay measures how long a reply takes to start. llama.cpp supplies exact per-request stream timings, including reasoning tokens, without polling server-wide /metrics. Other providers may show an explicitly labelled estimate.</p><label className="switch-row">Show live inference statistics<input type="checkbox" role="switch" checked={preferences.showInferenceStats!==false} onChange={e=>updatePreferences({showInferenceStats:e.target.checked})}/></label><p className="caption">Applies immediately to full and mini chat. Hiding statistics does not change inference or logging.</p><p>Deferring Julia reduces competing analysis while the main model generates. Emotion and motion decisions may update later; an already-running call can still overlap. Main-model background slot priority is controlled separately in Models.</p><p>Logs go to <code>logs/debug.log</code>. DEBUG includes slot waits, prompt packing and Julia analysis time; INFO includes generation timings. Review logs before sharing: timing instrumentation does not collect conversation text, but other errors can contain local paths. Choose the level, size and backup count below, then Save and restart Python.</p></section>}
        {group==='graphics'&&<div id="settings-graphics"><GraphicsSettings preferences={preferences} update={updatePreferences}/></div>}
        {sections.map(section=><section tabIndex={-1} ref={sectionRef(section)} className="settings-subsection" key={section}>{sections.length>1&&<h3>{section}</h3>}<div className="settings-grid">{fields.filter(field=>(field.section||'Settings')===section).map(field=>renderField(field))}</div></section>)}
      <datalist id="hf-models">{repos.map(model=><option key={model.id} value={model.id}/>)}</datalist><datalist id="gguf-files">{files.map(file=><option key={file} value={file}/>)}</datalist>
       {group==='voice'&&<details id="settings-voice-live" className="live-settings"><summary>Live calibration & detector testing</summary><VoiceInput/></details>}
       {group==='initiative'&&<details id="settings-initiative-live" className="live-settings"><summary>Live initiative preferences & event rules</summary><p>Saved live preferences override YAML initiative defaults.</p><InitiativeSettings/></details>}
       {group==='appearance'&&<details id="settings-display-live" className="live-settings"><summary>Live monitor & surface placement</summary><DisplaySettings/></details>}
         </section><footer className="settings-save"><div><strong>{dirty?'Unsaved changes':'All changes saved'}</strong><small>{validating?'Checking your changes…':'Some changes need a restart. We’ll tell you after saving.'}</small></div><div><button disabled={!dirty||busy} onClick={()=>{setInputs(inputValues(snapshot));setModelSource(snapshot.values['runtime.model_path']?'local':'huggingface');setServerErrors({});setPathChecks({});setNotice('Changes discarded.');}}>Discard</button><button className="primary" disabled={!dirty||busy||validating||Object.values(errors).some(Boolean)} onClick={save}><Save size={17}/>{busy?'Saving…':'Save settings'}</button></div></footer>
    </>}
  </main>;
}
