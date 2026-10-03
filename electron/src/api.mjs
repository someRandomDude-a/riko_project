export const API = 'http://127.0.0.1:8765';
export const EVENTS = 'ws://127.0.0.1:8765/ws/events';
export const mediaURL = path => API + '/api/media?path=' + encodeURIComponent(path);

export async function request(path, {method = 'GET', body, signal} = {}) {
  const response = await fetch(API + path, {
    method,
    ...(signal?{signal}:{}),
    ...(body === undefined ? {} : {headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}),
  });
  if (!response.ok) {
    const text=await response.text();
    let message=text;
    try {const value=JSON.parse(text);if(typeof value.detail==='string')message=value.detail;}catch{}
    throw new Error(message);
  }
  return response.json();
}

export function reportSurface(surface, command_id, status, error = '', bounds) {
  return request('/api/surfaces/result', {method: 'POST', body: {surface, command_id, status, error, bounds}}).catch(() => {});
}
