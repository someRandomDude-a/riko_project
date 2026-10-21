// The backend's address, defined once. Main resolves the origin once (RIKO_BACKEND_URL in development, else this default;
// packaged, a free port it hands the backend it spawns as RIKO_PORT), passes it to every window as a --riko-backend=
// argument that preload exposes as rikoConfig.backend, and src/api.mjs builds every request and socket URL from that.
// Only http://127.0.0.1:<port>: main adds the API token for exactly this origin, and 'localhost' may resolve to ::1, where
// another account could listen. src/backend_origin.test.mjs fails on the default port anywhere else under electron/.
const DEFAULT_BACKEND='http://127.0.0.1:8765';
const ARGUMENT='--riko-backend=';
// 1024-65535, as the backend's RIKO_PORT (backend_port in Code/process/app_core/desktop/api_guard.py).
function originForPort(port){
 if(!Number.isInteger(port)||port<1024||port>65535)throw new Error('The backend port must be 1024-65535, not '+port);
 const host='127.0.0.1:'+port;
 return Object.freeze({origin:'http://'+host,socket:'ws://'+host,host,port,patterns:Object.freeze(['http://'+host+'/*','ws://'+host+'/*'])});
}
function backendOrigin(value=DEFAULT_BACKEND){
 const text=String(value).trim(),match=/^http:\/\/127\.0\.0\.1:([0-9]{1,5})\/?$/.exec(text);
 if(!match)throw new Error('RIKO_BACKEND_URL must be http://127.0.0.1:<port> (no localhost, path or credentials), not '+JSON.stringify(text));
 return originForPort(Number(match[1]));
}
// Development only: a packaged app ignores RIKO_BACKEND_URL and talks to the backend it spawns.
const resolveBackend=(env=process.env)=>backendOrigin(env.RIKO_BACKEND_URL||DEFAULT_BACKEND);
// Whether a renderer request goes to this backend (http or ws, exact host and port): the only requests that carry the token.
function ownsRequest(backend,url){try{const target=new URL(url);return ['http:','ws:'].includes(target.protocol)&&target.host===backend.host;}catch{return false;}}
// webPreferences.additionalArguments reach process.argv in the renderer, where a sandboxed preload reads them synchronously.
const rendererArguments=backend=>backend?[ARGUMENT+backend.origin]:[];
module.exports={DEFAULT_BACKEND,ARGUMENT,backendOrigin,originForPort,resolveBackend,ownsRequest,rendererArguments};
