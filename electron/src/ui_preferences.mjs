import {useEffect, useState} from 'react';
import {appearanceDefaults,skins} from './ui/skins.mjs';
import {feedbackDefaults,normalizeFeedback} from './feedback_model.mjs';
import {borderDefaults,normalizeBorders} from './border_effects.mjs';
import {hitDefaults,normalizeHitSettings} from './avatar_hit_settings.mjs';
import {HIT_BONES} from './avatar_bvh.mjs';
import {studioDefaults,normalizeStudio} from './avatar_studio_settings.mjs';
import {normalizeGraphics} from './avatar_graphics.mjs';
const defaults = {activity:false, tools:true, reasoning:false, system:false, density:'comfortable',quickActions:[],showTaskActions:false,...appearanceDefaults,...feedbackDefaults,...borderDefaults,...hitDefaults,...studioDefaults};
export function readPreferences() {
  try {return normalizePreferences(JSON.parse(localStorage.getItem('riko:ui') || '{}'));} catch {return {...defaults};}
}
export function normalizePreferences(raw) {
  const value={...defaults,...raw};
  for(const key of ['background','surface','accent','text','muted','gradientStart','gradientEnd'])if(!/^#[a-f0-9]{6}$/i.test(value[key]))value[key]=defaults[key];
  if(!skins.includes(value.skin))value.skin=defaults.skin;
  if(!['compact','comfortable'].includes(value.density))value.density='comfortable';
  for(const key of ['activity','tools','reasoning','system','showTaskActions','reduceMotion','advancedExplanations'])if(typeof value[key]!=='boolean')value[key]=defaults[key];
  for(const [key,min,max] of [['gradientAngle',0,360],['glassOpacity',.35,1],['windowOpacity',.35,1],['boardOpacity',.35,1],['glassBlur',0,48],['cornerRadius',0,28]])value[key]=typeof value[key]==='number'&&Number.isFinite(value[key])?Math.min(max,Math.max(min,value[key])):defaults[key];
  value.quickActions=Array.isArray(value.quickActions)?value.quickActions.filter(a=>a&&typeof a.label==='string'&&typeof a.prompt==='string').slice(0,12).map(a=>({...a,label:a.label.slice(0,80),prompt:a.prompt.slice(0,2000)})):[];
    return {...value,...normalizeFeedback(value),...normalizeBorders(value),...normalizeHitSettings(value,HIT_BONES),...normalizeStudio(value),...normalizeGraphics(value)};
}
export function usePreferences() {
  const [preferences, set] = useState(readPreferences);
  useEffect(()=>{const changed=event=>{if(event.key==='riko:ui')set(readPreferences());};window.addEventListener('storage',changed);return()=>window.removeEventListener('storage',changed);},[]);
  function update(changes) {set(old=>{const next=normalizePreferences({...old,...readPreferences(),...changes});try{localStorage.setItem('riko:ui',JSON.stringify(next));}catch{}return next;});}
  return [preferences, update];
}
