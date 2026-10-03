import React from 'react';
import {borderDefaults,borderTargets} from './border_effects.mjs';
export default function BorderSettings({preferences,update}){
 const p={...borderDefaults,...preferences};
 const triggers=[['mic','Microphone enabled'],['wake','Wake-up (brief flash)'],['awake','Awake / accepting speech'],['capturing','Recording speech'],['transcribing','Recording or transcribing speech'],['generating','Generating a reply']];
 return <><h3>Voice borders</h3><p className="caption">Each border has its own trigger. A shorter animation period is faster. Audio feedback speeds up rainbow motion or makes a solid-color pulse stronger. Reduce motion disables border animations.</p>{borderTargets.map(target=>{
 const prefix=target+'Border',name=target==='mic'?'Mic button':target==='screen'?'Screen':'Dock';
 const toggle=(key,label)=><label className="switch-row">{label}<input type="checkbox" role="switch" checked={p[prefix+key]} onChange={e=>update({[prefix+key]:e.target.checked})}/></label>;
 const select=(key,label,options)=><label className="setting-field"><span>{label}</span><select value={p[prefix+key]} onChange={e=>update({[prefix+key]:e.target.value})}>{options.map(([value,text])=><option key={value} value={value}>{text}</option>)}</select></label>;
 const range=(key,label,min,max,step)=><label className="setting-field"><span>{label}: {p[prefix+key]}</span><input aria-label={label} type="range" min={min} max={max} step={step} value={p[prefix+key]} onChange={e=>update({[prefix+key]:Number(e.target.value)})}/></label>;
 return <section key={target}><h4>{name} border</h4>{toggle('Enabled','Show '+name.toLowerCase()+' border')}<div className="settings-grid">{select('Trigger',name+' border trigger',triggers)}{select('Style',name+' border style',[['rainbow','Rainbow'],['solid','Solid color']])}{range('Width',name+' border thickness',1,12,1)}{range('Speed',name+' animation period (seconds)',.3,12,.1)}<label className="setting-field"><span>{name} solid color</span><input type="color" value={p[prefix+'Color']} onChange={e=>update({[prefix+'Color']:e.target.value})}/></label></div>{toggle('Pulse','Pulse solid color')}{toggle('Activity','React to microphone audio level')}</section>;
 })}</>;
}
