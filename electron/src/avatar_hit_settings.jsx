import React,{useState,useEffect} from 'react';
import {readCatalog,CATALOG_KEY} from './avatar_studio_settings.mjs';
import {HIT_BONES} from './avatar_bvh.mjs';
import {hitAnimations,hitExpressions,hitEvents} from './avatar_hit_settings.mjs';
import useResource from './use_resource.jsx';
export default function AvatarHitSettings({preferences:p,update}){
  const [bone,setBone]=useState('head'),[event,setEvent]=useState('click');
  const [library]=useResource('animation');
  const [catalog,setCatalog]=useState(readCatalog);useEffect(()=>{const refresh=()=>setCatalog(readCatalog()),storage=e=>{if(e.key===CATALOG_KEY)refresh();};window.addEventListener('storage',storage);window.addEventListener('avatar-catalog',refresh);return()=>{window.removeEventListener('storage',storage);window.removeEventListener('avatar-catalog',refresh);};},[]);
  const bones=[...HIT_BONES,...(catalog?.bones||[]).filter(e=>e.spring&&!e.human).map(e=>e.id)];
 const rule=p.avatarHitRules?.[bone]?.[event]||{animation:'default',expression:'default',intensity:.7};
 const change=patch=>update({avatarHitRules:{...p.avatarHitRules,[bone]:{...p.avatarHitRules?.[bone],[event]:{...rule,...patch}}}});
  return <section id="appearance-avatar-hits"><h2>Interaction connections</h2><p className="caption">Select a bone → choose a trigger → connect animation and expression. Ragdoll starts at your movement threshold; recovery fires on release.</p><div className="interaction-graph" aria-label="Selected interaction connection"><button onClick={()=>document.getElementById('hit-bone')?.focus()}>{bone}</button><span>→</span><button onClick={()=>document.getElementById('hit-event')?.focus()}>{event}</button><span>→</span><strong>{rule.animation} + {rule.expression}</strong></div>
   <div className="settings-grid"><label className="setting-field">Bone<select id="hit-bone" value={bone} onChange={e=>setBone(e.target.value)}>{bones.map(name=><option key={name} value={name}>{catalog?.bones.find(e=>e.id===name)?.name||name}</option>)}</select></label>
   <label className="setting-field">Interaction<select id="hit-event" value={event} onChange={e=>setEvent(e.target.value)}>{hitEvents.map(name=><option key={name}>{name}</option>)}</select></label>
   <label className="setting-field">Procedural animation<select value={rule.animation} onChange={e=>change({animation:e.target.value})}>{hitAnimations.map(name=><option key={name}>{name}</option>)}</select></label>
   <label className="setting-field">Imported animation connection<select value={rule.assetId||''} onChange={e=>change({assetId:e.target.value})}><option value="">Procedural only</option>{(library?.entries||[]).map(asset=><option key={asset.id} value={asset.id}>{asset.name} · {asset.kind}</option>)}</select></label>
   <label className="setting-field">Expression<select value={rule.expression} onChange={e=>change({expression:e.target.value})}>{hitExpressions.map(name=><option key={name}>{name}</option>)}</select></label>
   <label className="setting-field">Color effect<select value={rule.effect||'default'} onChange={e=>change({effect:e.target.value})}>{['default','none','warm','cool','monochrome'].map(name=><option key={name}>{name}</option>)}</select></label>
   <label className="setting-field">Expression strength<input type="range" min="0" max="1" step=".05" value={rule.intensity} onChange={e=>change({intensity:Number(e.target.value)})}/></label>
   <button onClick={()=>{const rules={...p.avatarHitRules,[bone]:{...p.avatarHitRules?.[bone]}};delete rules[bone][event];update({avatarHitRules:rules});}}>Disconnect selected trigger</button>
  <label className="setting-field toggle-field">Show hit-volume outlines<input type="checkbox" checked={p.avatarHitOutlines===true} onChange={e=>update({avatarHitOutlines:e.target.checked})}/></label></div>
   <fieldset><legend>Outline bone filter (none selected shows all)</legend><p className="caption">Visualization only; filtering does not disable hit testing. Every imported spring has a selection volume. Yellow marks the hovered or held bone.</p>{bones.map(name=><label key={name} style={{display:'inline-flex',alignItems:'center',gap:6,margin:6}}><input type="checkbox" checked={p.avatarHitBones?.includes(name)||false} onChange={e=>update({avatarHitBones:e.target.checked?[...(p.avatarHitBones||[]),name]:(p.avatarHitBones||[]).filter(item=>item!==name)})}/>{catalog?.bones.find(e=>e.id===name)?.name||name}</label>)}</fieldset>
  <h3>Saved connections</h3><div className="interaction-connections">{Object.entries(p.avatarHitRules||{}).flatMap(([name,events])=>Object.entries(events).map(([trigger,value])=><button key={name+trigger} className="interaction-graph" onClick={()=>{setBone(name);setEvent(trigger);}} aria-label={'Edit '+name+' '+trigger+' connection'}><span>{name}</span><span>→ {trigger} →</span><strong>{value.assetId?'Imported clip':value.animation} + {value.expression}</strong></button>))}</div></section>;
}
