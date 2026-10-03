import React,{useEffect,useState} from 'react';
import {request} from './api.mjs';
import {connectEvents} from './event_connection.mjs';
import Explanation from './explanation.jsx';
const memory=value=>value==null?'Unavailable':`${Math.round(value).toLocaleString()} MiB`;

export default function GPUResources({changes={},valid=true}){
  const [live,setLive]=useState(null),[estimate,setEstimate]=useState(null),[error,setError]=useState('');
  const signature=JSON.stringify(changes);
  const liveSignature=JSON.stringify(live?.gpus||[]);
  useEffect(()=>{
    return connectEvents(event=>{
      const value=event.type==='resource.snapshot'?event.payload?.gpu:event.type==='resource.gpu'?event.payload:null;
      if(value)setLive(value);
    },connected=>setError(connected?'':'GPU telemetry disconnected'),{url:'ws://127.0.0.1:8765/ws/resources/gpu'});
  },[]);
  useEffect(()=>{
    let alive=true;if(!valid){setError('Fix invalid draft settings to calculate the estimate.');return;}
    const controller=new AbortController();
    const timer=setTimeout(()=>request('/api/resources/estimate',{method:'POST',body:{changes:JSON.parse(signature)},signal:controller.signal}).then(data=>{if(alive){setEstimate(data.estimate);setError('');}}).catch(e=>{if(alive)setError(e.message);}),180);
    return()=>{alive=false;clearTimeout(timer);controller.abort();};
  },[signature,valid,liveSignature]);
  return <section className="gpu-resources"><div className="section-heading"><h3>GPU memory</h3><span className="eyebrow">LIVE</span></div>
    {error&&<p role="alert">{error}</p>}
    {!live?.available?<p>{live?.error||'Reading GPU…'}</p>:live.gpus.map(gpu=><article key={gpu.uuid}><h4>{gpu.name} · GPU {gpu.index}</h4>
      <progress max={gpu.total_mib} value={gpu.used_mib} aria-label="Total GPU VRAM used"/>
      <p>Total: <strong>{memory(gpu.total_mib)}</strong> · Used: <strong>{memory(gpu.used_mib)}</strong> · Free: {memory(gpu.free_mib)}</p>
      <details className="resource-details"><summary>Process usage</summary><p>Owned processes: {memory(gpu.owned_mib)} · Other programs / driver: {memory(gpu.other_mib)}</p>
       {!gpu.attribution_complete&&<p className="caption">This driver cannot report all usage by program. Some values are unavailable.</p>}
      {(gpu.owned_processes||[]).map(row=><p className="caption" key={row.pid}>{row.category} · PID {row.pid}: {memory(row.used_mib)}</p>)}</details></article>)}
     {estimate&&<><h4>Estimated app memory: {memory(estimate.low_mib)}–{memory(estimate.high_mib)}{!estimate.complete?' + unknown usage':''}</h4>
       {!estimate.projection&&live?.gpus?.[0]&&<><progress max={live.gpus[0].total_mib} value={Math.min(live.gpus[0].total_mib,estimate.high_mib)} aria-label="Estimated owned draft GPU memory"/><p className="caption">Based on your unsaved settings{!estimate.complete?' · some usage is unknown':''}. Current usage is shown above.</p></>}
      {estimate.projection&&<><progress className={estimate.projection.fits_upper_estimate?'':'over-budget'} max={estimate.projection.total_mib} value={Math.min(estimate.projection.total_mib,estimate.projection.high_mib)} aria-label="Estimated draft GPU memory peak"/><p className="caption">Draft projected peak: {memory(estimate.projection.low_mib)}–{memory(estimate.projection.high_mib)} / {memory(estimate.projection.total_mib)}{!estimate.projection.fits_upper_estimate?' · Over estimated capacity':''} · estimate, not measured usage</p></>}
       <Explanation simple="The estimate updates as you change settings. It may differ from actual usage."><div className="resource-details"><h4>Memory breakdown</h4>
      <p className="caption">Uses your settings draft. External providers/TTS are excluded; measurements reflect currently running programs.</p>
      <p className="caption">Estimate confidence: <strong>{estimate.confidence}</strong>. {estimate.confidence_basis}</p>
      {estimate.projection&&<p>Projected GPU total including current Other: <strong>{memory(estimate.projection.low_mib)}–{memory(estimate.projection.high_mib)}</strong> / {memory(estimate.projection.total_mib)}.{!estimate.projection.fits_upper_estimate&&' The upper estimate exceeds GPU capacity; review settings or competing programs.'}</p>}
      {estimate.managed&&<p>{estimate.kv.unified?'Unified KV pool':'Separate equal-sized slot caches'}: {estimate.kv.pool_tokens.toLocaleString()} tokens with all {estimate.kv.slots} slots occupied. Live {estimate.kv.live_context_tokens.toLocaleString()}, initiative {estimate.kv.initiative_context_tokens.toLocaleString()}, reflection {estimate.kv.reflection_context_tokens.toLocaleString()} tokens (each includes output).</p>}
      <div className="resource-components">{estimate.components.map(c=><details key={c.id}><summary><span>{c.label}</span><strong>{c.high_mib==null?'Unknown':`${Math.round(c.low_mib).toLocaleString()}–${Math.round(c.high_mib).toLocaleString()} MiB`}</strong></summary><p className="caption">{c.basis}</p></details>)}</div>
      {estimate.suggestion?<p><strong>{estimate.suggestion.live_context_tokens>0?`Suggested live context ceiling: ${estimate.suggestion.live_context_tokens.toLocaleString()} tokens`:'No estimated context headroom under the current conservative memory budget'}</strong> · {estimate.suggestion.basis}. Your selected size is unchanged.</p>:<p className="caption">Context recommendation unavailable without sufficient model metadata, or not applicable to this configuration.</p>}
       <details className="resource-details"><summary>Assumptions & diagnostics</summary>{estimate.warnings.map((warning,index)=><p className="caption" key={index}>{warning}</p>)}<p className="caption">{estimate.note}</p>{live?.source&&<p className="caption">Measurement source: {live.source}. {live.note}</p>}</details></div></Explanation></>}
  </section>;
}
