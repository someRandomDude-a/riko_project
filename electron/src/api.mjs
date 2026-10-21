// The backend origin Electron main resolved once (electron/backend_origin.cjs) and preload hands every window before its
// first script runs. Every request and socket goes there; main adds the API token for that origin alone.
export const API = String(globalThis.rikoConfig?.backend || '');
export const socketURL = path => API.replace(/^http/, 'ws') + path;
export const EVENTS = socketURL('/ws/events');
export const mediaURL = path => API + '/api/media?path=' + encodeURIComponent(path);

// The one renderer fetch wrapper. body is sent as JSON, or as is with type (a Content-Type) for binary uploads. A failure
// throws an Error carrying the backend's detail (or its raw text, whatever the body is) and the HTTP status.
// Electron main adds the API token to every backend request; renderer code never handles it.
export async function request(path, {method = 'GET', body, signal, type} = {}, confirmation) {
  const raw = type !== undefined;
  const headers = {...(body === undefined ? {} : {'Content-Type': raw ? type : 'application/json'}), ...(confirmation ? {'X-Riko-Confirmation': confirmation} : {})};
  const response = await fetch(API + path, {
    method,
    ...(signal?{signal}:{}),
    headers,
    ...(body === undefined ? {} : {body: raw ? body : JSON.stringify(body)}),
  });
  if (response.status === 428 && !confirmation) {
    // A security-sensitive change: Electron main asks the user and signs the exact change if they allow it.
    let challenge; try {challenge=(await response.clone().json()).detail?.confirm;} catch {}
    if (challenge) {
      const signature = await globalThis.riko?.confirmChange?.(challenge);
      if (!signature) throw new Error('Change cancelled. Nothing was saved.');
      return request(path, {method, body, signal, type}, signature);
    }
  }
  if (!response.ok) {
    const text=await response.text();
    let message=text;
    try {const value=JSON.parse(text);if(typeof value.detail==='string')message=value.detail;else if(typeof value.detail?.detail==='string')message=value.detail.detail;}catch{}
    if(response.status===404&&path.startsWith('/api/discord/'))message='This Python backend does not support the current Discord controls. Restart Python from this updated checkout, then retry.';
    throw Object.assign(new Error(message||`Request failed (${response.status})`), {status: response.status});
  }
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

export function reportSurface(surface, command_id, status, error = '', bounds) {
  return request('/api/surfaces/result', {method: 'POST', body: {surface, command_id, status, error, bounds}}).catch(() => {});
}
