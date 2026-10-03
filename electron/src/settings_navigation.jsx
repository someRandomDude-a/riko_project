import React,{useEffect,useRef} from 'react';
import {Menu,X,ChevronRight} from './ui/icons.jsx';
export function SettingsTabs({groups,group,onSelect}){
  function keydown(event){
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();const i=groups.findIndex(([id])=>id===group);
    const next=event.key==='Home'?0:event.key==='End'?groups.length-1:(i+(event.key==='ArrowRight'?1:-1)+groups.length)%groups.length;
    onSelect(groups[next][0]);event.currentTarget.querySelectorAll('[role=tab]')[next].focus();
  }
  return <nav className="settings-tabs" role="tablist" aria-label="Settings categories" onKeyDown={keydown}>{groups.map(([id,label])=><button role="tab" id={'tab-'+id} aria-controls="settings-panel" aria-selected={group===id} tabIndex={group===id?0:-1} key={id} className={group===id?'active':''} onClick={()=>onSelect(id)}>{label}</button>)}</nav>;
}
export default function SettingsNavigation({groups,group,sections,onSelect,onJump}){
  const dialog=useRef(null);
  useEffect(()=>()=>dialog.current?.close(),[]);
  return <><button className="icon-button" aria-label="Open settings menu" title="Settings menu" onClick={()=>dialog.current.showModal()}><Menu size={20}/></button>
    <dialog ref={dialog} className="settings-drawer" aria-labelledby="settings-menu-title" onClick={event=>{if(event.target===dialog.current){const r=dialog.current.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.current.close();}}}>
      <header><h2 id="settings-menu-title">Settings menu</h2><button autoFocus className="icon-button" aria-label="Close settings menu" onClick={()=>dialog.current.close()}><X size={18}/></button></header>
      <nav aria-label="All settings">{groups.map(([id,label])=><button key={id} aria-current={group===id?'page':undefined} className={group===id?'active':''} onClick={()=>{onSelect(id);dialog.current.close();}}>{label}<ChevronRight size={15}/></button>)}</nav>
      {!!sections.length&&<><h3>Jump to a section</h3><nav aria-label="Current settings sections">{sections.map(section=><button key={section} onClick={()=>{dialog.current.close();onJump(section);}}>{section}</button>)}</nav></>}
    </dialog></>;
}
