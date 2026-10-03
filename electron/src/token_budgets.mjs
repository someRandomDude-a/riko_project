export function tokenBudgets(values){
  const number=(key,fallback)=>Number(values[key]??fallback);
  const live=number('runtime.n_ctx',8192),slots=number('runtime.parallel_slots',2);
  const initiative=number('initiative.context_window_tokens',4096),reflection=number('memory.reflection_context_window_tokens',4096);
  const unified=values['runtime.kv_unified']!==false;
  const background=Math.max(reflection*(slots-1),initiative+reflection*(slots-2));
  return {live,slots,initiative,reflection,unified,background,sum:live+initiative+reflection,
    suggested:unified?live+background:Math.max(live,initiative,reflection)*slots,
    liveOutput:number('runtime.max_output_tokens',1024),livePrompt:number('memory.context_window_tokens',7168),
    initiativeOutput:number('initiative.max_output_tokens',1024),reflectionOutput:number('memory.reflection_max_output_tokens',1024),
    recall:number('memory.token_budget',1200),emotion:number('emotion.context_tokens',1024),
    emotionMaximum:number('emotion.max_length',8192),classifier:number('memory.system1_max_length',8192)};
}
export function syncPool(values){
  return values['runtime.kv_pool_auto']===false?values:{...values,'runtime.kv_pool_tokens':String(tokenBudgets(values).suggested)};
}
