import test from 'node:test';
import assert from 'node:assert/strict';
import {tokenBudgets,syncPool} from './token_budgets.mjs';
test('two slots share initiative and reflection; output is not double counted',()=>{
  const values={'runtime.n_ctx':16384,'runtime.parallel_slots':2,'initiative.context_window_tokens':4096,'memory.reflection_context_window_tokens':8192};
  const result=tokenBudgets(values);
  assert.equal(result.sum,28672);
  assert.equal(result.suggested,24576);
  assert.equal(syncPool(values)['runtime.kv_pool_tokens'],'24576');
});
test('four-slot suggestion covers both all-reflection and mixed background jobs',()=>{
  const values={'runtime.n_ctx':16384,'runtime.parallel_slots':4,'initiative.context_window_tokens':8192,'memory.reflection_context_window_tokens':4096};
  assert.equal(tokenBudgets(values).suggested,32768);
  values['initiative.context_window_tokens']=2048;
  assert.equal(tokenBudgets(values).suggested,28672);
  values['runtime.kv_unified']=false;
  assert.equal(tokenBudgets(values).suggested,65536);
});
test('manual pool value stays untouched; auto value follows draft changes',()=>{
  const manual={'runtime.kv_pool_auto':false,'runtime.kv_pool_tokens':'65536','runtime.n_ctx':'8192'};
  assert.equal(syncPool(manual),manual);
  assert.equal(syncPool({...manual,'runtime.kv_pool_auto':true})['runtime.kv_pool_tokens'],'12288');
});
