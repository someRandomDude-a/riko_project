import React, {useState} from 'react';
import useRuntime from './use_runtime.jsx';
import {request} from './api.mjs';

export default function DisplaySettings() {
  const state = useRuntime();
  const [error, setError] = useState(''), [saving, setSaving] = useState(false);
  const screens = state.displays || [];
  async function select(event) {
    setSaving(true);
    try {
      await request('/api/surfaces/avatar', {method: 'PATCH', body: {screen: Number(event.target.value)}});
      setError('');
    } catch (error) {setError(error.message);}
    finally {setSaving(false);}
  }
  return <section className="panel"><h2>Avatar display settings</h2>
    <p>Drag the avatar to reposition it. Transparent pixels remain click-through.</p>
    <select aria-label="Avatar monitor" disabled={saving || !screens.length}
      value={state.avatar_geometry?.screen ?? 0} onChange={select}>
      {screens.map(display => <option key={display.id} value={display.index}>
        {display.index}: {display.label}{display.primary ? ' (primary)' : ''} · {display.bounds.width}×{display.bounds.height}
      </option>)}
    </select>
    {!screens.length && <p>Waiting for display information from Electron…</p>}
    {error && <p role="alert">{error}</p>}
  </section>;
}
