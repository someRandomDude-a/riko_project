import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import StreamChat from './stream_chat.jsx';
import SettingsPage from './settings_page.jsx';
import ActivityPanel from './activity_panel.jsx';
import {usePreferences} from './ui_preferences.mjs';
import {MessageCircle, PanelsTopLeft, CheckCheck, Settings, Activity, Palette,ArrowUpRight} from './ui/icons.jsx';
import AppearancePage from './appearance_page.jsx';
import {appearanceStyle} from './ui/skins.mjs';
import useRuntime from './use_runtime.jsx';
import TaskPanel from './task_panel.jsx';
import {ApprovalBubble} from './tool_approvals.jsx';
import Workspace from './workspace.jsx';
import {Overlay, BoardWindow, EffectsWindow} from './runtime_windows.jsx';
import './style.css';
import './ui/theme.css';
import './ui/glass.css';
import './ui/dock.css';
import './ui/morph.css';
import WindowChrome from './window_chrome.jsx';
import {ExplanationContext} from './explanation.jsx';
import CompactScale from './compact_scale.jsx';
import CornerResize from './corner_resize.jsx';

function App() {
  if (location.hash === '#/whiteboard') return <BoardWindow/>;
  if (location.hash === '#/effects') return <EffectsWindow/>;
  if (location.hash === '#/overlay') return <Overlay/>;
  return <Controls/>;
}

function Controls() {
  const [view, setView] = useState('chat');
  const [settingsDirty,setSettingsDirty]=useState(false);
  const [mode,setMode]=useState('full');
  const [dockSize,setDockSize]=useState(88);
  useEffect(()=>{document.documentElement.classList.toggle('transparent-chat',mode!=='full');return()=>document.documentElement.classList.remove('transparent-chat');},[mode]);
  useEffect(()=>{if(mode==='full')return;const hit=event=>{const target=event.target.closest('.msg,.composer,.speech-popup,.glass-dock,.dash-bar,.scale-grip,.corner-resize,.notice,[role=alert]');window.compactBridge?.interactive(!!target&&Number(getComputedStyle(target).opacity)>.08||!!document.querySelector('[data-dragging=true]'));};document.addEventListener('pointermove',hit,true);return()=>{document.removeEventListener('pointermove',hit,true);};},[mode]);
  useEffect(()=>{let alive=true;window.windowBridge?.state().then(value=>{if(alive){setMode(value.mode||'full');if(value.dockWidth)setDockSize(value.dockWidth);}}).catch(()=>{});const off=window.windowBridge?.subscribe(value=>{if(value.mode)setMode(value.mode);if(value.dockWidth)setDockSize(value.dockWidth);});return()=>{alive=false;off?.();};},[]);
  const [preferences, updatePreferences] = usePreferences();
  const runtime=useRuntime();
  function navigate(name){if(view==='settings'&&settingsDirty&&name!==view&&!confirm('Discard unsaved settings changes?'))return;setView(name);}
  useEffect(()=>window.riko?.onNavigate?.(name=>{if(['chat','workspace','tasks','settings','appearance'].includes(name))navigate(name);}),[view,settingsDirty]);
  useEffect(()=>{document.title=runtime.character_name||'Conversation';},[runtime.character_name]);
   return <ExplanationContext.Provider value={preferences.advancedExplanations}><div className={'app-shell mode-'+mode+' density-'+preferences.density+' skin-'+preferences.skin+(preferences.reduceMotion?' reduce-motion':'')} style={{...appearanceStyle(preferences),'--collapsed-size':dockSize+'px'}}>
     <WindowChrome title={runtime.character_name||''} blur={mode==='full'&&preferences.glassBlur>0}><div className="app-header">
       <button className="compact-switch quiet-button" aria-label={mode==='compact'?'Expand to full interface':'Open compact chat'} title={mode==='compact'?'Full interface':'Compact chat'} onClick={()=>{if(mode==='full'&&settingsDirty&&!confirm('Discard unsaved settings changes?'))return;window.windowBridge?.mode(mode==='compact'?'full':'compact').then(()=>{if(mode==='full')setView('chat');}).catch(()=>{});}}>{mode==='compact'?<ArrowUpRight size={19}/>:'Compact chat'}</button>
      <nav aria-label="Main navigation">{[['chat','Conversation',MessageCircle],['workspace','Canvas',PanelsTopLeft],['tasks','Tasks',CheckCheck],['settings','Settings',Settings],['appearance','Appearance',Palette]].map(([name,label,Icon])=>
         <button key={name} className={view===name?'active':''} aria-label={label} title={label} aria-current={view===name?'page':undefined} onClick={()=>navigate(name)}><Icon size={17}/><span>{label}</span></button>)}
       </nav><button className={'activity-toggle '+(preferences.activity?'active':'')} aria-label="Activity" title="Activity" aria-pressed={preferences.activity} onClick={()=>updatePreferences({activity:!preferences.activity})}><Activity size={18}/><span>Activity</span></button>
     </div></WindowChrome>
     {mode==='compact'&&<CompactScale/>}
     {mode!=='full'&&<CornerResize/>}
    <div className="app-body"><div className="primary-view">
    {/* Keep the event consumer mounted while browsing tools/settings: no lost turns. */}
      <div className="chat-host" hidden={mode==='full'&&view !== 'chat'}><StreamChat compactMode={mode!=='full'} collapsedMode={mode==='collapsed'} onSettings={()=>{window.windowBridge?.mode('full');navigate('settings');}} preferences={preferences}/></div>
      <div className="view-host" hidden={mode!=='full'||view === 'chat'}>
      {view === 'workspace' && <Workspace/>}
      {view === 'settings' && <SettingsPage preferences={preferences} updatePreferences={updatePreferences} onDirty={setSettingsDirty}/>}
      {view === 'appearance' && <AppearancePage preferences={preferences} updatePreferences={updatePreferences}/>}
      {view === 'tasks' && <TaskPanel/>}
    </div></div>
     {mode==='full'&&preferences.activity&&<ActivityPanel preferences={preferences} updatePreferences={updatePreferences}/>}
    <ApprovalBubble/>
    </div>
   </div></ExplanationContext.Provider>;
}

createRoot(document.getElementById('root')).render(<App/>);
