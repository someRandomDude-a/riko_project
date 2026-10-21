import React, {useEffect, useState} from 'react';
import useResource from './use_resource.jsx';
import {request} from './api.mjs';
import {awarenessToggles,perceptionLine,privacyNote} from './initiative_model.mjs';

export default function InitiativeSettings() {
  const [status, setStatus, streamError] = useResource('initiative');
  const [settings, setSettings] = useState(null);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [context, setContext] = useState('');
  async function load() {
    try {
      const value = await request('/api/initiative');
      setStatus(value); setSettings(value.settings); setError('');
    } catch (exc) {setError(exc.message);}
  }
  useEffect(()=>{if(status?.settings)setSettings(old=>old||status.settings);},[status]);
  function change(key, value) {setSettings(old => ({...old, [key]: value}));}
  function ruleChange(index, key, value) {
    change('rules', settings.rules.map((rule, i) => i === index ? {...rule, [key]: value} : rule));
  }
  async function save() {
    setPending(true);
    try {
      const {context_window_tokens,max_output_tokens,...preferences}=settings;
      const value = await request('/api/initiative', {method: 'PUT', body: preferences});
      setStatus(value); setSettings(value.settings); setError('');
    } catch (exc) {setError(exc.message);} finally {setPending(false);}
  }
  async function testEvent() {
    try {
      await request('/api/initiative/event', {method: 'POST', body: {event: 'user.custom', context}});
      setError('');
    } catch (exc) {setError(exc.message);}
  }
  return <section className="panel">
    <h2>Initiative & event rules</h2>
    <p>Optional background checks can initiate conversation. Disabled by default; bubbles by default.</p>
    <p>{privacyNote(status?.capabilities)}</p>
    {error && <p role="alert">{error}</p>}
    {streamError&&<p role="alert">{streamError}</p>}
    {!settings ? <button onClick={load}>Connect / retry</button> : <>
      {[
        {key: 'enabled', label: 'Enable initiative checks'},
        ...awarenessToggles(status?.capabilities),
        {key: 'spoken_enabled', label: 'Allow spoken presentation for spoken rules'},
        {key: 'allow_urgent_spoken', label: 'Allow model-marked urgent spoken overrides'},
      ].map(({key, label, disabled, reason}) => <label key={key} style={{display: 'block'}} title={reason || undefined}><input type="checkbox" disabled={!!disabled} checked={settings[key] && !disabled} onChange={e => change(key, e.target.checked)}/>{label}{reason && <span className="caption"> · {reason}</span>}</label>)}
      {[
        ['interval_seconds', 'Periodic check interval (seconds)', 10],
        ['idle_seconds', 'Become idle after (seconds)', 5],
        ['cooldown_seconds', 'Minimum time between initiations (seconds)', 10],
      ].map(([key, label, min]) => <label key={key} style={{display: 'block'}}>{label}<input type="number" min={min} max="86400" value={settings[key]} onChange={e => change(key, Number(e.target.value))}/></label>)}
      <p className="caption">Inference budgets are configured under Models & runtime → Token budgets. Invalid output fails the attempt without a repair retry.</p>
      <h3>Event → conditions → model instruction → presentation</h3>
      <p>Rules are declarative, not arbitrary code. The model may remain silent. App conditions require app awareness; idle conditions require idle awareness.</p>
      {settings.rules.map((rule, index) => <fieldset key={rule.id}>
        <legend>Rule {index + 1}</legend>
        <label><input type="checkbox" checked={rule.enabled} onChange={e => ruleChange(index, 'enabled', e.target.checked)}/>Enabled</label>
        <label>Event<select value={rule.event} onChange={e => ruleChange(index, 'event', e.target.value)}>{status.triggers.map(trigger => <option key={trigger.event} value={trigger.event}>{trigger.description}</option>)}</select></label>
        <label>Minimum idle seconds<input type="number" min="0" value={rule.min_idle_seconds || 0} onChange={e => ruleChange(index, 'min_idle_seconds', Number(e.target.value))}/></label>
        <label>App name or window title contains<input value={rule.app_contains || ''} onChange={e => ruleChange(index, 'app_contains', e.target.value)}/></label>
        <label>Rule cooldown seconds<input type="number" min="0" value={rule.cooldown_seconds} onChange={e => ruleChange(index, 'cooldown_seconds', Number(e.target.value))}/></label>
        <label>Model instruction<textarea maxLength="2000" value={rule.instruction} onChange={e => ruleChange(index, 'instruction', e.target.value)}/></label>
        <label>Presentation<select value={rule.presentation} onChange={e => ruleChange(index, 'presentation', e.target.value)}><option value="bubble">Bubble</option><option value="spoken">Spoken (requires permission)</option></select></label>
        <button onClick={() => change('rules', settings.rules.filter((_, i) => i !== index))}>Remove rule</button>
      </fieldset>)}
      <button disabled={settings.rules.length >= 50} onClick={() => change('rules', [...settings.rules, {id: crypto.randomUUID(), enabled: true, event: 'user.custom', instruction: 'Consider offering relevant help based on this event.', presentation: 'bubble', cooldown_seconds: 300}])}>Add rule</button>
      <button disabled={pending} onClick={save}>Save & apply</button>
      <button disabled={pending} onClick={load}>Reload saved settings/status</button>
      <h3>Test custom event</h3>
      <p>Save an enabled “Custom user event” rule first. Cooldowns, busy-state checks and the model's decision still apply.</p>
      <input maxLength="2000" value={context} onChange={e => setContext(e.target.value)} placeholder="Event context"/>
      <button onClick={testEvent}>Publish custom event</button>
      <p>{!status.settings.enabled ? 'Checks disabled' : status.busy ? 'Evaluating an event…' : 'Observer ready'}{status.error && ` · ${status.error}`}</p>
      {status.perception && <p className="caption">{perceptionLine(status.perception)}</p>}
      <p>Last model check: {status.last_check || 'None yet'}{status.last_decision && ` · ${status.last_decision.initiate ? 'Initiate' : 'Remain silent'}`}</p>
      <pre>{JSON.stringify(status.environment, null, 2)}</pre>
    </>}
  </section>;
}
