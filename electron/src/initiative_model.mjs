// Initiative's computer-awareness toggles from the backend's capabilities (runtime/activity.py through /api/initiative): a
// toggle this OS or desktop cannot serve is disabled with the reason, and the copy names what is really shared.
const supported=entry=>entry?.supported!==false;

export function activeAppLabel(capabilities){
  const fields=capabilities?.active_app?.fields||['window_title','process_id'];
  return fields.includes('window_title')?'Allow foreground window title/process awareness':'Allow foreground app name/process awareness';
}

export function awarenessToggles(capabilities){
  return [['observe_idle','Allow computer idle/return awareness','idle'],['observe_active_app',activeAppLabel(capabilities),'active_app']]
    .map(([key,label,field])=>{const entry=capabilities?.[field];return {key,label,disabled:!supported(entry),reason:supported(entry)?'':entry.reason||'Not available here'};});
}

export function privacyNote(capabilities){
  const app=capabilities?.active_app;
  const shared=supported(app)?`App monitoring shares ${app?.description||'the foreground window title and process ID'} with your configured model provider.`:'App monitoring is not available on this computer.';
  return `Idle monitoring reads time since keyboard/mouse activity, not keys. ${shared} No screenshots are captured.`;
}

const STATUS={off:'off',unsupported:'not available here',waiting:'waiting for the first reading',available:'reading',error:'failed (see the error)'};
export function perceptionLine(perception){
  if(!perception)return '';
  return `Idle awareness: ${STATUS[perception.idle]||perception.idle} · App awareness: ${STATUS[perception.active_app]||perception.active_app}`;
}
