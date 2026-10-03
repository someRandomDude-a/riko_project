import React from 'react';
import {sliderSpec} from './settings_model.mjs';
export default function NumericSetting({field,id,value,invalid,onChange}){
  const slider=sliderSpec(field,value);
  return <div className="numeric-setting">
    <input id={id} type="number" value={value} disabled={field.readonly} aria-invalid={invalid} step={field.integer?1:'any'} min={field.min} max={field.max} placeholder={field.nullable?'Automatic':undefined} onChange={e=>onChange(e.target.value)}/>
    {slider&&<><input aria-label={field.label+' slider'} type="range" min={slider.min} max={slider.max} step={slider.step} value={Number(value)} onChange={e=>onChange(e.target.value)}/><small>{slider.min.toLocaleString()} — {slider.max.toLocaleString()} · exact values can be typed above</small></>}
  </div>;
}
