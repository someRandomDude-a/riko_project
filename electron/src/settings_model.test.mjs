import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs';
import {parseSetting,settingsPatch,settingsEdited,inputValues,llamaServerCommand,runtimePresets,ruleErrors,fieldRelevant,shownFields,restartNotice,pendingRestart} from './settings_model.mjs';
// The backend's rules and visible_when with cases, checked against kernel/schema.py by tests/test_electron_settings.py.
const contract=JSON.parse(fs.readFileSync(new URL('./settings_contract.json',import.meta.url),'utf8'));
const typed=inputs=>Object.fromEntries(Object.entries(inputs).map(([path,value])=>[path,typeof value==='number'?String(value):value??'']));

test('numeric inputs validate immediately and do not coerce blanks to zero',()=>{
  const field={kind:'number',integer:true,min:2,max:4};
  assert.deepEqual(parseSetting(field,'3'),{value:3});
  for(const value of ['','NaN','1','5','2.5'])assert.ok(parseSetting(field,value).error);
});
test('nullable automatic fields and JSON drafts are typed correctly',()=>{
  assert.deepEqual(parseSetting({kind:'number',nullable:true},''),{value:null});
  assert.deepEqual(parseSetting({kind:'json'},'[1, 2]'),{value:[1,2]});
  assert.ok(parseSetting({kind:'json'},'[invalid]').error);
});
test('only changed values are submitted and boolean false remains false',()=>{
  const fields=[{path:'enabled',kind:'boolean'},{path:'count',kind:'number',integer:true}];
  assert.deepEqual(settingsPatch(fields,{enabled:true,count:2},{enabled:false,count:'2'}),{changes:{enabled:false},errors:{}});
});
test('the backend\'s rules judge the draft as the backend does, typed or as the form holds it',()=>{
  for(const {inputs,errors} of contract.rule_cases){
    assert.deepEqual(Object.keys(ruleErrors(contract.rules,inputs)).sort(),errors,JSON.stringify(inputs));
    assert.deepEqual(Object.keys(ruleErrors(contract.rules,typed(inputs))).sort(),errors,'as text: '+JSON.stringify(inputs));
  }
  const [rule]=contract.rules,draft={'runtime.provider':'llama_cpp','runtime.n_batch':'128','runtime.n_ubatch':'256'};
  assert.deepEqual(ruleErrors([rule],draft),{'runtime.n_ubatch':rule.message},'the message is the backend\'s');
  for(const value of ['','abc'])assert.deepEqual(ruleErrors([rule],{...draft,'runtime.n_ubatch':value}),{},'a blank or unparsed entry has its own error');
  assert.deepEqual(ruleErrors(undefined,draft),{});
});
test('rule errors join the per-field errors, which win for the same field',()=>{
  const fields=[{path:'runtime.n_ubatch',kind:'number',integer:true},{path:'runtime.n_batch',kind:'number',integer:true}];
  const inputs={'runtime.provider':'llama_cpp','runtime.model_path':'model.gguf','runtime.n_batch':'128','runtime.n_ubatch':'256'};
  assert.deepEqual(settingsPatch(fields,{'runtime.n_batch':128,'runtime.n_ubatch':128},inputs,contract.rules),{changes:{'runtime.n_ubatch':256},errors:{'runtime.n_ubatch':contract.rules[0].message}});
  assert.deepEqual(settingsPatch(fields,{},{...inputs,'runtime.n_ubatch':'2.5'},contract.rules).errors,{'runtime.n_ubatch':'Enter a whole number'});
  assert.deepEqual(settingsPatch(fields,{},inputs).errors,{},'no rules, no checks across settings');
});
test('a field applies to the providers its visible_when names; one without applies to all',()=>{
  for(const [provider,visible] of Object.entries(contract.visible)){
    const shown=Object.entries(contract.visible_when).filter(([path,visible_when])=>fieldRelevant({path,...(visible_when?{visible_when}:{})},{'runtime.provider':provider})).map(([path])=>path);
    assert.deepEqual(shown,visible,provider);
  }
  assert.equal(fieldRelevant({visible_when:{'runtime.provider':['llama_cpp'],'voice.mode':['wake']}},{'runtime.provider':'llama_cpp','voice.mode':'push'}),false,'every condition must hold');
});
test('a group lists the draft provider\'s fields and model source; Advanced controls and a search jump show the others',()=>{
  const when=path=>contract.visible_when[path]?{visible_when:contract.visible_when[path]}:{},native={visible_when:{'runtime.provider':['llama_cpp']}};
  const fields=[...['runtime.provider','runtime.n_ctx','runtime.n_ubatch','runtime.model_path','runtime.parallel_slots','runtime.base_url','runtime.model','runtime.api_mode'].map(path=>({path,group:'models',...when(path)})),
    {path:'runtime.hf_repo_id',group:'models',...native},{path:'runtime.seed',group:'models',advanced:true,...native},{path:'runtime.kv_pool_tokens',group:'models',...native},
    {path:'memory.context_window_tokens',group:'models'},{path:'avatar.model',group:'appearance'},{path:'avatar.enabled',group:'appearance'}];
  const shown=(provider,options={})=>shownFields(fields,{group:'models',inputs:{'runtime.provider':provider},...options}).map(field=>field.path);
  // What the hard-coded provider lists showed before visible_when (same sets, in the backend's order).
  assert.deepEqual(shown('llama_cpp'),['runtime.provider','runtime.n_ctx','runtime.n_ubatch','runtime.parallel_slots','runtime.hf_repo_id','memory.context_window_tokens']);
  assert.deepEqual(shown('llama_cpp',{modelSource:'local'}),['runtime.provider','runtime.n_ctx','runtime.n_ubatch','runtime.model_path','runtime.parallel_slots','memory.context_window_tokens']);
  assert.deepEqual(shown('llama_server'),['runtime.provider','runtime.n_ctx','runtime.parallel_slots','runtime.base_url','memory.context_window_tokens']);
  assert.deepEqual(shown('openai',{modelSource:'local'}),['runtime.provider','runtime.n_ctx','runtime.base_url','runtime.model','runtime.api_mode','memory.context_window_tokens']);
  assert.deepEqual(shown('openai',{advanced:true}),fields.filter(field=>field.group==='models'&&field.path!=='runtime.kv_pool_tokens').map(field=>field.path),'every field but those with panels of their own');
  assert.deepEqual(shown('openai',{destination:'runtime.seed'}),['runtime.provider','runtime.n_ctx','runtime.base_url','runtime.model','runtime.api_mode','runtime.seed','memory.context_window_tokens']);
  assert.deepEqual(shownFields(fields,{group:'appearance',inputs:{}}).map(field=>field.path),['avatar.enabled']);
});
test('runtime presets use llama.cpp automatic flash attention',()=>{
  for(const preset of Object.values(runtimePresets))assert.equal(preset['runtime.flash_attn'],'auto');
});
test('form hydration preserves zero, false and JSON arrays',()=>{
  const values={zero:0,flag:false,split:[1,1]};
  const fields=[{path:'zero',kind:'number'},{path:'flag',kind:'boolean'},{path:'split',kind:'json'}];
  const inputs=inputValues({values,fields});
  assert.equal(inputs.zero,0);assert.equal(inputs.flag,false);assert.deepEqual(JSON.parse(inputs.split),[1,1]);
});

test('the llama-server command matches what the backend checks: largest budget per slot, configured port',()=>{
  assert.equal(llamaServerCommand({'runtime.parallel_slots':2,'runtime.n_ctx':'6144','initiative.context_window_tokens':8192,'memory.reflection_context_window_tokens':4096,'runtime.base_url':'http://127.0.0.1:8081/v1'}),
    'llama-server -m model.gguf --port 8081 --parallel 2 --ctx-size 16384 --jinja');
  assert.equal(llamaServerCommand({'runtime.parallel_slots':'3','runtime.n_ctx':16384,'runtime.base_url':'not a url'}), 'llama-server -m model.gguf --port 8080 --parallel 3 --ctx-size 49152 --jinja');
});
test('only user edits count as unsaved; errors the stored config already has do not',()=>{
  const fields=[{path:'runtime.provider',kind:'string',options:['llama_cpp','openai']},{path:'runtime.model_path',kind:'string'},{path:'runtime.hf_repo_id',kind:'string'},{path:'runtime.hf_filename',kind:'string'},{path:'runtime.n_batch',kind:'number',integer:true},{path:'runtime.n_ubatch',kind:'number',integer:true}];
  const snapshot={fields,values:{'runtime.provider':'llama_cpp','runtime.model_path':'','runtime.hf_repo_id':'','runtime.hf_filename':'','runtime.n_batch':256,'runtime.n_ubatch':512}};
  const inputs=inputValues(snapshot);
  snapshot.rules=contract.rules;
  assert.deepEqual(Object.keys(settingsPatch(fields,snapshot.values,inputs,snapshot.rules).errors).sort(),['runtime.hf_repo_id','runtime.n_ubatch']);
  assert.equal(settingsEdited(snapshot,inputs),false);
  assert.equal(settingsEdited(snapshot,{...inputs,'runtime.n_ubatch':'512'}),false);
  assert.equal(settingsEdited(snapshot,{...inputs,'runtime.n_batch':'abc'}),true);
  assert.equal(settingsEdited(snapshot,{...inputs,'runtime.n_ubatch':'1024'}),true);
  assert.equal(settingsEdited(snapshot,{...inputs,'runtime.provider':'openai'}),true);
});

test('the after-save notice names what applies now, the restart each change awaits and what earlier saves still wait for',()=>{
  const fields=[{path:'avatar.enabled',label:'Avatar',restart_scope:'none'},{path:'runtime.n_ctx',label:'Context',restart_scope:'python'},{path:'runtime.n_batch',label:'Batch',restart_scope:'python'},
    {path:'desktop.debug',label:'Debug',restart_scope:'electron'},{path:'voice.mode',label:'Mode',restart_scope:'microphone'},{path:'runtime.seed',label:'Seed',restart_scope:'python'}];
  assert.equal(restartNotice({'avatar.enabled':false},{restart_required:false,fields}),'Saved. Changes are active now.');
  assert.equal(restartNotice({'avatar.enabled':false,'runtime.n_ctx':4096,'desktop.debug':true,'voice.mode':'push'},{restart_required:true,fields,restart_pending:{'runtime.n_ctx':'python','desktop.debug':'electron','voice.mode':'microphone'}}),
    'Saved. Avatar is active now. Restart Python to apply Context. Restart Electron to apply Debug. Turn the microphone off and on to apply Mode.');
  const before={'runtime.seed':'python','runtime.n_batch':'python'},pending={'runtime.n_ctx':'python','runtime.seed':'python'};
  assert.equal(restartNotice({'runtime.n_ctx':4096,'runtime.n_batch':256},{restart_required:true,fields,restart_pending:pending},before,field=>field.label.toUpperCase()),
    'Saved. Restart Python to apply CONTEXT and BATCH. 1 earlier saved change still waits for a restart.');
  assert.equal(restartNotice({'avatar.enabled':true},{restart_required:false,fields,restart_pending:pending},before),'Saved. Changes are active now. 1 earlier saved change still waits for a restart.');
  const many=Object.fromEntries(['a','b','c','d'].map(key=>['runtime.'+key,1]));
  assert.equal(restartNotice(many,{restart_required:true,fields:[]}),'Saved. Restart Python to apply runtime.a, runtime.b and 2 more.','an unknown field restarts Python, by its path');
});
test('the save bar says how many saved settings still wait for a restart, and of what',()=>{
  assert.equal(pendingRestart({restart_pending:{}}),'');assert.equal(pendingRestart(null),'');
  assert.equal(pendingRestart({restart_pending:{'runtime.n_ctx':'python'}}),'1 saved setting waits for a restart of Python.');
  assert.equal(pendingRestart({restart_pending:{'runtime.n_ctx':'python','desktop.debug':'electron','runtime.seed':'python'}}),'3 saved settings wait for a restart of Python and Electron.');
});

test('a field the draft provider does not use never blocks Save, even with a leftover invalid value', () => {
  const fields = [{path: 'runtime.provider', kind: 'choice', options: ['llama_cpp', 'openai']},
    {path: 'runtime.api_mode', kind: 'choice', options: ['auto', 'responses', 'chat_completions'], visible_when: {'runtime.provider': ['openai']}}];
  const values = {'runtime.provider': 'llama_cpp', 'runtime.api_mode': 'chat'};
  assert.deepEqual(settingsPatch(fields, values, {'runtime.provider': 'llama_cpp', 'runtime.api_mode': 'chat'}).errors, {});
  assert.deepEqual(settingsPatch(fields, values, {'runtime.provider': 'openai', 'runtime.api_mode': 'chat'}).errors, {'runtime.api_mode': 'Choose a supported value'});
});

test('a masked secret left as is sends nothing; clearing it or typing a new one is a change', () => {
  const mask='•'.repeat(8),fields=[{path:'runtime.api_key',kind:'text',secret:true,secret_state:'saved'},{path:'runtime.n_ctx',kind:'number'}];
  const snapshot={fields,values:{'runtime.api_key':mask,'runtime.n_ctx':4096}},inputs=inputValues(snapshot);
  assert.equal(inputs['runtime.api_key'],mask);  // the form holds the mask, never the key
  assert.deepEqual(settingsPatch(fields,snapshot.values,{...inputs,'runtime.n_ctx':'8192'}).changes,{'runtime.n_ctx':8192});
  assert.deepEqual(settingsPatch(fields,snapshot.values,{...inputs,'runtime.api_key':''}).changes,{'runtime.api_key':''});
  assert.deepEqual(settingsPatch(fields,snapshot.values,{...inputs,'runtime.api_key':'sk-new'}).changes,{'runtime.api_key':'sk-new'});
});
