export function parseSetting(field, input) {
  if (field.nullable && (input === '' || input === null)) return {value: null};
  if (field.kind === 'boolean') return {value: !!input};
  if (field.kind === 'number') {
    if (String(input).trim() === '') return {error: 'Enter a number'};
    const value = Number(input);
    if (!Number.isFinite(value)) return {error: 'Enter a finite number'};
    if (field.integer && !Number.isInteger(value)) return {error: 'Enter a whole number'};
    if (field.min != null && value < field.min || field.max != null && value > field.max) return {error: `Use ${field.min ?? '−∞'} to ${field.max ?? '∞'}`};
    return {value};
  }
  if (field.kind === 'json') {
    try {return {value: typeof input === 'string' ? JSON.parse(input) : input};}
    catch {return {error: 'Enter valid JSON'};}
  }
  if (field.options && !field.options.includes(input)) return {error: 'Choose a supported value'};
  return {value: input};
}

export function settingsPatch(fields, values, inputs) {
  const changes = {}, errors = {};
  for (const field of fields) {
    const parsed = parseSetting(field, inputs[field.path]);
    if (parsed.error) errors[field.path] = parsed.error;
    else if (JSON.stringify(parsed.value) !== JSON.stringify(values[field.path])) changes[field.path] = parsed.value;
  }
  if (inputs['runtime.provider'] === 'llama_cpp') {
    if (!inputs['runtime.model_path'] && (!inputs['runtime.hf_repo_id'] || !inputs['runtime.hf_filename'])) errors['runtime.hf_repo_id'] = 'Choose a local GGUF or a Hugging Face repository and file';
    if (Number(inputs['runtime.n_ubatch']) > Number(inputs['runtime.n_batch'])) errors['runtime.n_ubatch'] = 'Physical batch must not exceed logical batch';
    if (!['f16','f32','bf16'].includes(inputs['runtime.type_v']) && !inputs['runtime.flash_attn']) errors['runtime.type_v'] = 'Quantized V requires flash attention';
  }
  return {changes, errors};
}

export const runtimePresets = {
  balanced: {'runtime.n_gpu_layers': -1, 'runtime.n_batch': 512, 'runtime.n_ubatch': 512, 'runtime.flash_attn': false, 'runtime.type_k': 'f16', 'runtime.type_v': 'f16'},
  compact: {'runtime.n_gpu_layers': -1, 'runtime.n_batch': 256, 'runtime.n_ubatch': 128, 'runtime.flash_attn': true, 'runtime.type_k': 'q8_0', 'runtime.type_v': 'q8_0'},
  cpu: {'runtime.n_gpu_layers': 0, 'runtime.n_batch': 256, 'runtime.n_ubatch': 128, 'runtime.flash_attn': false, 'runtime.type_k': 'f16', 'runtime.type_v': 'f16'},
};

export function inputValues(snapshot) {
  return Object.fromEntries(snapshot.fields.map(field=>[field.path, field.kind === 'json' && snapshot.values[field.path] != null
    ? JSON.stringify(snapshot.values[field.path], null, 2) : snapshot.values[field.path] ?? '']));
}

export function sliderSpec(field,value){
  if(field.readonly||field.kind!=='number'||value===''||value==null||!Number.isFinite(Number(value)))return null;
  const budgets=/(_tokens|\.n_ctx|\.context_tokens|\.max_length)$/.test(field.path);
  const compute=['runtime.n_batch','runtime.n_ubatch','runtime.n_gpu_layers'].includes(field.path);
  let min=field.min,max=field.max,step=field.integer?1:.01;
  if(budgets){min=field.min??1;max=Math.min(field.max??65536,65536);step=256;}
  else if(compute){min=field.min??(field.path.endsWith('n_gpu_layers')?-1:1);max=field.path.endsWith('n_gpu_layers')?128:4096;step=1;}
  else if(!Number.isFinite(min)||!Number.isFinite(max)||max-min>1000)return null;
  else if(!field.integer&&max-min>10)step=.1;
  return {min:Math.min(min,Number(value)),max:Math.max(max,Number(value)),step};
}
