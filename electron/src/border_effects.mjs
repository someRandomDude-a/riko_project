export const borderTargets=['mic','screen','dock'];
export const borderTriggers=['mic','wake','awake','capturing','transcribing','generating'];
export const borderDefaults=Object.fromEntries(borderTargets.flatMap(target=>Object.entries({Enabled:true,Trigger:target==='mic'?'mic':target==='screen'?'wake':'transcribing',Style:'rainbow',Color:'#b699e6',Width:target==='mic'?6:8,Speed:4,Pulse:false,Activity:false}).map(([key,value])=>[target+'Border'+key,value])));
export function normalizeBorders(raw){
 const value={...borderDefaults,...raw};
 for(const target of borderTargets){const prefix=target+'Border';
  for(const key of ['Enabled','Pulse','Activity'])if(typeof value[prefix+key]!=='boolean')value[prefix+key]=borderDefaults[prefix+key];
  if(!borderTriggers.includes(value[prefix+'Trigger']))value[prefix+'Trigger']=borderDefaults[prefix+'Trigger'];
  if(!['rainbow','solid'].includes(value[prefix+'Style']))value[prefix+'Style']='rainbow';
  if(!/^#[a-f0-9]{6}$/i.test(value[prefix+'Color']))value[prefix+'Color']=borderDefaults[prefix+'Color'];
  for(const [key,min,max] of [['Width',1,12],['Speed',.3,12]])value[prefix+key]=Number.isFinite(value[prefix+key])?Math.min(max,Math.max(min,value[prefix+key])):borderDefaults[prefix+key];
 }
 return Object.fromEntries(Object.keys(borderDefaults).map(key=>[key,value[key]]));
}
export function borderEffect(target,preferences,voice,{wake=false,generating=false}={}){
 const p={...borderDefaults,...preferences},prefix=target+'Border',trigger=p[prefix+'Trigger'];
 const active=p[prefix+'Enabled']&&(trigger==='generating'?generating:trigger==='mic'?voice.enabled:voice.enabled&&!voice.wake?.calibrating&&!voice.wake?.testing&&(trigger==='wake'&&wake||trigger==='awake'&&(voice.wake?.active||['awake','capturing','transcribing','follow_up'].includes(voice.phase))||trigger==='capturing'&&voice.phase==='capturing'||trigger==='transcribing'&&['capturing','transcribing'].includes(voice.phase)));
 const level=p[prefix+'Activity']?Math.max(0,Math.min(1,voice.level||0)):0;
 return {active:!!active,className:'border-effect effect-'+p[prefix+'Style']+(p[prefix+'Pulse']||p[prefix+'Activity']?' effect-pulse':''),style:{'--effect-color':p[prefix+'Color'],'--effect-width':p[prefix+'Width']+'px','--effect-speed':p[prefix+'Speed']/(1+level*3)+'s','--effect-level':level}};
}
