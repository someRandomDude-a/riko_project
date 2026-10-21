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

// The backend's lists of allowed values ({path: [values]}: a field's visible_when, a rule's when and one_of) against the
// draft. Inputs hold what the form shows, so a typed '512' matches 512 and a blank matches null.
const same = (allowed, value) => allowed === value || (allowed ?? '') === (value ?? '')
  || typeof allowed !== 'object' && typeof value !== 'object' && String(allowed) === String(value);
const holds = (conditions, inputs) => Object.entries(conditions || {}).every(([path, allowed]) => allowed.some(item => same(item, inputs[path])));
const blank = value => value == null || value === '';

// Whether a field applies to the draft: visible_when names the providers that read a runtime key (configuration/schema.py).
export function fieldRelevant(field, inputs) { return holds(field.visible_when, inputs); }

// The fields a Settings group lists: the one a search jumped to (destination, a path) always; otherwise the group's, less
// those with panels of their own (the avatar library and the conversation cache) and, unless Advanced controls is on,
// advanced fields, fields of providers the draft does not use (visible_when) and the llama.cpp model source it does not use.
const OWN_PANELS = ['avatar.model', 'avatar.format', 'runtime.kv_pool_auto', 'runtime.kv_pool_tokens'];
export function shownFields(fields, {group, inputs, advanced = false, modelSource = 'huggingface', destination}) {
  const source = path => inputs['runtime.provider'] !== 'llama_cpp' || (modelSource === 'local' ? !path.startsWith('runtime.hf_') : path !== 'runtime.model_path');
  return fields.filter(field => field.path === destination || field.group === group && !OWN_PANELS.includes(field.path)
    && (advanced || !field.advanced && fieldRelevant(field, inputs) && source(field.path)));
}

// The checks across settings the backend enforces (/api/settings 'rules', kernel/schema.py Rule.broken), on the draft:
// {path: message}. A rule applies when every `when` holds, and fails when path's value exceeds at_most's, is not below
// below's, a one_of value is not allowed, or no any_set group is fully set. A blank or unparsed number is not compared;
// its field reports it.
export function ruleErrors(rules, inputs) {
  const errors = {};
  const exceeds = (other, fits, path) => {
    if (!other || blank(inputs[path]) || blank(inputs[other])) return false;
    const value = Number(inputs[path]), limit = Number(inputs[other]);
    return Number.isFinite(value) && Number.isFinite(limit) && !fits(value, limit);
  };
  for (const rule of rules || []) {
    if (!holds(rule.when, inputs) || rule.path in errors) continue;
    if (exceeds(rule.at_most, (a, b) => a <= b, rule.path) || exceeds(rule.below, (a, b) => a < b, rule.path) || !holds(rule.one_of, inputs)
      || rule.any_set?.length && !rule.any_set.some(group => group.every(path => !blank(inputs[path])))) errors[rule.path] = rule.message;
  }
  return errors;
}

export function settingsPatch(fields, values, inputs, rules = []) {
  const changes = {}, errors = {};
  for (const field of fields) {
    const parsed = parseSetting(field, inputs[field.path]);
    // A field the draft's provider does not use (visible_when) is hidden, so its value, perhaps left by another provider, never blocks Save.
    if (parsed.error) { if (fieldRelevant(field, inputs)) errors[field.path] = parsed.error; }
    else if (JSON.stringify(parsed.value) !== JSON.stringify(values[field.path])) changes[field.path] = parsed.value;
  }
  for (const [path, message] of Object.entries(ruleErrors(rules, inputs))) errors[path] ??= message;
  return {changes, errors};
}

// What a save needs before it applies, from each field's restart_scope (kernel/schema.py RESTART): 'none' applies on save.
const RESTARTS = [['python', 'Restart Python to apply'], ['electron', 'Restart Electron to apply'], ['microphone', 'Turn the microphone off and on to apply']];
function names(labels) {
  const shown = labels.slice(0, labels.length > 3 ? 2 : 3), more = labels.length - shown.length;
  return more ? `${shown.join(', ')} and ${more} more` : shown.length > 1 ? `${shown.slice(0, -1).join(', ')} and ${shown.at(-1)}` : shown[0];
}

// The notice after a save: what is active now, which restart each change awaits, and whether changes saved earlier still
// wait for one (before: restart_pending of the snapshot the save started from; labelOf: a field's name as the page shows it).
export function restartNotice(changes, result, before = {}, labelOf = field => field.label) {
  const fields = new Map((result.fields || []).map(field => [field.path, field])), pending = result.restart_pending || {};
  const scope = path => fields.get(path)?.restart_scope ?? pending[path] ?? 'python';
  const label = path => (fields.has(path) && labelOf(fields.get(path))) || path;
  const earlier = Object.keys(before || {}).filter(path => !(path in changes) && path in pending).length;
  const waiting = earlier > 0 && `${earlier} earlier saved ${earlier > 1 ? 'changes still wait' : 'change still waits'} for a restart.`;
  if (!result.restart_required) return ['Saved. Changes are active now.', waiting].filter(Boolean).join(' ');
  const paths = Object.keys(changes), live = paths.filter(path => scope(path) === 'none');
  const parts = RESTARTS.map(([name, text]) => [text, paths.filter(path => scope(path) === name).map(label)]).filter(([, labels]) => labels.length).map(([text, labels]) => `${text} ${names(labels)}.`);
  return ['Saved.', live.length > 0 && `${names(live.map(label))} ${live.length > 1 ? 'are' : 'is'} active now.`, ...parts, waiting].filter(Boolean).join(' ');
}

// Saved settings the running app has not loaded yet (restart_pending: {path: scope}), for the save bar.
export function pendingRestart(snapshot) {
  const pending = Object.values(snapshot?.restart_pending || {});
  if (!pending.length) return '';
  const scopes = RESTARTS.filter(([name]) => pending.includes(name)).map(([name]) => ({python: 'Python', electron: 'Electron', microphone: 'the microphone'})[name]);
  return `${pending.length} saved ${pending.length > 1 ? 'settings wait' : 'setting waits'} for a restart${scopes.length ? ' of ' + names(scopes) : ''}.`;
}

// Whether the user edited anything a reload or quit would lose: a change to save, or an entry that does not
// parse yet. Errors the stored values already have (a config the client flags) are not edits.
export function settingsEdited(snapshot, inputs, patch = settingsPatch(snapshot.fields, snapshot.values, inputs, snapshot.rules)) {
  const initial = inputValues(snapshot);
  return Object.keys(patch.changes).length > 0 || Object.keys(patch.errors).some(path => path in initial && String(inputs[path] ?? '') !== String(initial[path] ?? ''));
}

// The llama-server command for these settings: each slot needs room for the largest of the live,
// initiative and reflection budgets (as the backend checks), on the port in the server address.
export function llamaServerCommand(inputs) {
  const slots = Number(inputs['runtime.parallel_slots']) || 2;
  const perSlot = Math.max(Number(inputs['runtime.n_ctx']) || 0, Number(inputs['initiative.context_window_tokens']) || 4096, Number(inputs['memory.reflection_context_window_tokens']) || 4096);
  let port = '8080';
  try { port = new URL(String(inputs['runtime.base_url'] || 'http://127.0.0.1:8080')).port || '8080'; } catch {}
  return `llama-server -m model.gguf --port ${port} --parallel ${slots} --ctx-size ${slots * perSlot} --jinja`;
}

export const runtimePresets = {
  balanced: {'runtime.n_gpu_layers': -1, 'runtime.n_batch': 512, 'runtime.n_ubatch': 512, 'runtime.flash_attn': 'auto', 'runtime.type_k': 'f16', 'runtime.type_v': 'f16'},
  compact: {'runtime.n_gpu_layers': -1, 'runtime.n_batch': 256, 'runtime.n_ubatch': 128, 'runtime.flash_attn': 'auto', 'runtime.type_k': 'q8_0', 'runtime.type_v': 'q8_0'},
  cpu: {'runtime.n_gpu_layers': 0, 'runtime.n_batch': 256, 'runtime.n_ubatch': 128, 'runtime.flash_attn': 'auto', 'runtime.type_k': 'f16', 'runtime.type_v': 'f16'},
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
  else if(compute){min=field.min??(field.path.endsWith('n_gpu_layers')?-2:1);max=field.path.endsWith('n_gpu_layers')?128:4096;step=1;}
  else if(!Number.isFinite(min)||!Number.isFinite(max)||max-min>1000)return null;
  else if(!field.integer&&max-min>10)step=.1;
  return {min:Math.min(min,Number(value)),max:Math.max(max,Number(value)),step};
}
