import React, {useEffect, useState} from 'react';
import {Mic, MicOff, AudioLines} from './ui/icons.jsx';
import {connectEvents} from './event_connection.mjs';
import useVoiceState from './use_voice_state.jsx';
import {voiceLabel,microphoneAction} from './voice_state.mjs';

const API = 'http://127.0.0.1:8765';

export default function VoiceInput({compact=false, disabled=false}) {
  const voice=useVoiceState(),enabled=voice.enabled,level=voice.level;
  const micAction=microphoneAction(voice);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [wake, setWake] = useState(null);
  const [threshold, setThreshold] = useState(0.9);
  const [testScores, setTestScores] = useState([]);
  useEffect(() => {if (Number.isFinite(wake?.threshold)) setThreshold(wake.threshold);}, [wake?.threshold]);
  useEffect(() => {
    const disconnect=connectEvents(event=>{
        const status=event.type==='resource.snapshot'?event.payload?.voice:event.type==='resource.voice'?event.payload:null;
        if(status){setWake(status.wake);}
        if (event.type === 'voice.error') setError(event.payload.error);
        if (event.type === 'voice.wake_status') setWake(event.payload);
        if (event.type === 'voice.wake_score') setWake(old => old ? {...old, last_score: event.payload.score} : old);
        if (event.type === 'voice.wake_test') setTestScores(old => [...old.slice(-9), event.payload]);
        if (event.type === 'voice.calibration_discarded') setError(event.payload.error);
      });
    return disconnect;
  }, []);
  async function toggle() {
    setPending(true); setError('');
    try {
      const response = await fetch(API + micAction.path, {method: 'POST'});
      if (!response.ok) throw new Error(await response.text());
    } catch (e) {setError(e.message);} finally {setPending(false);}
  }
  async function calibration(action, extra = {}) {
    setPending(true); setError('');
    try {
      const response = await fetch(API + '/api/voice/calibration', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action, ...extra})});
      if (!response.ok) throw new Error(await response.text());
      setWake(await response.json());
    } catch (e) {setError(e.message);} finally {setPending(false);}
  }
  const tones = ['natural voice', 'slightly softer', 'slightly louder', 'a little higher', 'a little lower', 'natural voice again'];
  return <section className={'voice-input '+(compact?'compact':'')}>
    <div className="voice-actions"><button className={'mic-button '+(enabled?'listening':'')} aria-label={micAction.label} disabled={pending||disabled||voice.status==='starting'||(micAction.icon==='speak'&&(wake?.calibrating||wake?.testing))} onClick={toggle}>{micAction.icon==='off'?<MicOff size={22}/>:micAction.icon==='speak'?<AudioLines size={22}/>:<Mic size={22}/>}<span>{pending?'Please wait…':micAction.label}</span></button>
    <div className="voice-indicator"><meter aria-label="Microphone activity" min="0" max="1" value={level}/><small role="status" aria-live="polite">{voiceLabel(voice)}</small></div></div>
    {enabled && wake && <details className="voice-calibration">
      <summary>Wake-name setup & testing</summary><div className="card">
      <h3>Wake-name calibration: {wake.wake_word}</h3>
      <p>Microphone: {wake.device?.name || 'Connecting…'}. At least six local recordings; add as many as you need. No browser audio or uploads.</p>
      {Number.isFinite(wake.last_score) && <p>Last wake score: {wake.last_score.toFixed(3)} · threshold: {wake.threshold}</p>}
      {wake.error && <p role="alert">{wake.error}</p>}
      {!wake.calibrating ? <button disabled={pending || !wake.device} onClick={()=>calibration('begin')}>{wake.enrolled ? 'Add more samples' : 'Set up wake name'}</button> : <>
        <p>{wake.samples} samples (minimum 6). {wake.recording ? wake.speech_detected ? 'Speech detected — stop speaking to finish. Five-second limit.' : 'Waiting for speech — say the name once. Five-second limit.' : wake.processing ? 'Processing sample…' : `Next: say “${wake.wake_word}” in your ${tones[wake.samples % tones.length]}.`}</p>
        <button disabled={pending || wake.recording || wake.processing} onClick={()=>calibration('record')}>Record another sample</button>
        <button disabled={pending || wake.recording || wake.processing || wake.samples < 6} onClick={()=>calibration('save')}>Save detector</button>
        <button disabled={pending || wake.processing} onClick={()=>calibration('cancel')}>Cancel setup</button>
      </>}
      {wake.enrolled && !wake.calibrating && <>
        <button disabled={pending} onClick={()=>{setTestScores([]); calibration(wake.testing ? 'stop_test' : 'test');}}>{wake.testing ? 'Finish testing' : 'Test wake name (no replies)'}</button>
        <label style={{display:'block'}}>Activation threshold: {threshold.toFixed(2)}
          <input type="range" min="0.01" max="0.99" step="0.01" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}/>
        </label>
        <button disabled={pending} onClick={()=>calibration('threshold', {threshold})}>Save threshold</button>
        <p>Lower values wake more easily but may match unrelated speech. Save the threshold to apply it immediately and keep it after restart.</p>
        {wake.testing && <><p>Say the wake name in different tones, then unrelated words. Check that only the wake name matches.</p><ul>{testScores.map((score, index)=><li key={index}>{score.score.toFixed(3)} — {score.matched ? 'MATCH' : 'No match'} (threshold {score.threshold.toFixed(2)})</li>)}</ul></>}
      </>}
    </div></details>}
    {error && !disabled && <p role="alert">{error}</p>}
  </section>;
}
