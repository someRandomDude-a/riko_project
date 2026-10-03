import React from 'react';
import {tokenBudgets} from './token_budgets.mjs';
import Explanation from './explanation.jsx';
const tokens=value=>Number.isFinite(value)?value.toLocaleString():'—';
export default function KVPoolSettings({values,onChange,onSuggest}){
  const b=tokenBudgets(values),auto=values['runtime.kv_pool_auto']!==false;
   return <section className="gpu-resources"><div className="section-heading"><h3>Conversation cache</h3><span className="eyebrow">MEMORY BUDGET</span></div>
    <div className="resource-components"><div className="budget-row"><span>Live conversation <small>Output reserve {tokens(b.liveOutput)} · prompt limit {tokens(b.livePrompt)}</small></span><strong>{tokens(b.live)}</strong></div>
      <div className="budget-row"><span>Initiative <small>Includes {tokens(b.initiativeOutput)} output tokens</small></span><strong>{tokens(b.initiative)}</strong></div>
      <div className="budget-row"><span>Reflection <small>Includes {tokens(b.reflectionOutput)} output tokens per job</small></span><strong>{tokens(b.reflection)}</strong></div>
      <div className="budget-row"><span>Sum of task limits</span><strong>{tokens(b.sum)} tokens</strong></div>
      <div className="budget-row"><span>Suggested pool · {b.slots} occupied slots</span><strong>{tokens(b.suggested)} tokens</strong></div></div>
     <Explanation simple="Let the app size the cache for your chat and background work, or set a larger size yourself."><p className="caption">{b.unified?`Live + maximum concurrent background demand (${tokens(b.background)} tokens). Initiative and reflection share background slots.`:'Separate caches: largest task limit × slot count.'} Reply and recall budgets are already included. This calculation does not check how much your GPU can hold.</p></Explanation>
     <label className="switch-row">Size cache automatically<input aria-label="Use suggested pool size" type="checkbox" role="switch" checked={auto} onChange={e=>onChange('runtime.kv_pool_auto',e.target.checked)}/></label>
    <div className="field-input"><input aria-label="KV pool tokens" type="number" min={b.suggested} max="4194304" disabled={auto} value={auto?b.suggested:values['runtime.kv_pool_tokens']??b.suggested} onChange={e=>onChange('runtime.kv_pool_tokens',e.target.value)}/><button onClick={onSuggest}>Use suggested size</button></div>
    <small className="caption">Calculated from this draft and saved with other settings. Restart Python to resize the pool.</small>
    <details className="resource-details"><summary>Nested & CPU-only token budgets</summary><p className="caption">Recall: {tokens(b.recall)} tokens inside the live prompt. Emotion window: {tokens(b.emotion)} / {tokens(b.emotionMaximum)} encoder tokens. Memory classifier: {tokens(b.classifier)} encoder tokens. These do not add independent llama-server KV pools.</p></details>
  </section>;
}
