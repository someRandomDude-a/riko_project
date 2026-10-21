import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
const {DEFAULT_BACKEND,ARGUMENT,backendOrigin,originForPort,resolveBackend,ownsRequest,rendererArguments}=require('../backend_origin.cjs');
const electron=fileURLToPath(new URL('..',import.meta.url));
const read=file=>fs.readFileSync(path.join(electron,file),'utf8');
// Taken from the one definition, so this file holds no copy of it either.
const port=new URL(DEFAULT_BACKEND).port;

test('development resolves RIKO_BACKEND_URL, defaulting to the one definition',()=>{
  assert.deepEqual(resolveBackend({}),backendOrigin(DEFAULT_BACKEND));
  const backend=resolveBackend({RIKO_BACKEND_URL:' http://127.0.0.1:9123/ '});
  assert.deepEqual({...backend,patterns:[...backend.patterns]},{origin:'http://127.0.0.1:9123',socket:'ws://127.0.0.1:9123',host:'127.0.0.1:9123',port:9123,patterns:['http://127.0.0.1:9123/*','ws://127.0.0.1:9123/*']});
  assert.ok(Object.isFrozen(backend)&&Object.isFrozen(backend.patterns));
  assert.deepEqual(originForPort(9123),backend);
});

test('only http://127.0.0.1:<port> is a backend origin: the token goes to it alone',()=>{
  for(const value of [`https://127.0.0.1:${port}`,`http://localhost:${port}`,`http://[::1]:${port}`,`http://0.0.0.0:${port}`,`http://127.0.0.2:${port}`,'http://127.0.0.1',`http://127.0.0.1:${port}/v1`,`http://127.0.0.1:${port}?x=1`,`http://127.0.0.1:${port}#x`,`http://user:pass@127.0.0.1:${port}`,'http://127.0.0.1:80','http://127.0.0.1:1023','http://127.0.0.1:65536','http://127.0.0.1:0','127.0.0.1:9123','',' '])
    assert.throws(()=>backendOrigin(value),/RIKO_BACKEND_URL must be http:\/\/127\.0\.0\.1:<port>|backend port must be 1024-65535/,value);
  for(const value of [0,80,1023,65536,9123.5,NaN,'9123'])assert.throws(()=>originForPort(value),/1024-65535/,String(value));
  assert.throws(()=>resolveBackend({RIKO_BACKEND_URL:`http://localhost:${port}`}),/no localhost/);
});

test('a request carries the token only to the exact backend host and port, over http or ws',()=>{
  const backend=originForPort(9123);
  for(const url of ['http://127.0.0.1:9123/api/status','ws://127.0.0.1:9123/ws/events'])assert.equal(ownsRequest(backend,url),true,url);
  for(const url of ['http://127.0.0.1:9124/api/status','https://127.0.0.1:9123/','wss://127.0.0.1:9123/','http://localhost:9123/','http://127.0.0.1:9123.evil/','file:///index.html','not a url'])assert.equal(ownsRequest(backend,url),false,url);
});

// main_wiring.test.mjs loads main.cjs and preload.cjs: every window gets the preload, isolation and this argument, the preload
// exposes it as rikoConfig.backend (requiring only electron, as a sandboxed preload must), and main resolves the origin once,
// scoping the token to it before any window loads.
test('every window receives the origin as an argument, none before a backend exists',()=>{
  assert.deepEqual(rendererArguments(originForPort(9123)),[ARGUMENT+'http://127.0.0.1:9123']);
  assert.deepEqual(rendererArguments(null),[]);  // the first-run setup window, before any backend exists
});

function* sources(directory){
  for(const entry of fs.readdirSync(directory,{withFileTypes:true})){
    if(['node_modules','dist','release'].includes(entry.name))continue;
    const file=path.join(directory,entry.name);
    if(entry.isDirectory())yield* sources(file);
    else if(entry.isFile()&&entry.name!=='package-lock.json'&&/\.(?:[cm]?js|jsx|html|css|json|ya?ml|md)$/.test(entry.name))yield file;
  }
}

test('the default backend port appears nowhere under electron/ but its one definition',()=>{
  const literal=new RegExp(`\\b${port}\\b`),offenders=[],files=[...sources(electron)];
  for(const file of files)if(literal.test(fs.readFileSync(file,'utf8'))&&path.relative(electron,file)!=='backend_origin.cjs')offenders.push(path.relative(electron,file));
  assert.ok(files.some(file=>file.endsWith(path.join('src','api.mjs')))&&files.some(file=>file.endsWith('main.cjs')),'the scan covers main and the renderer');
  assert.deepEqual(offenders,[],'take the origin from backend_origin.cjs (main) or src/api.mjs (renderer)');
  assert.equal((read('backend_origin.cjs').match(new RegExp(literal,'g'))||[]).length,1);
});

test('main-process code reaches the backend only through backend_supervisor.cjs, which adds the token',()=>{
  const offenders=fs.readdirSync(electron).filter(name=>name.endsWith('.cjs')).filter(name=>(read(name).match(/\bfetch\(/g)||[]).length!==(name==='backend_supervisor.cjs'?1:0));
  assert.deepEqual(offenders,[]);
});

test('src/api.mjs request() is the renderer\'s one fetch wrapper, and api.mjs its one origin',()=>{
  const src=path.join(electron,'src'),offenders=[];
  for(const name of fs.readdirSync(src).filter(name=>/\.(?:mjs|jsx)$/.test(name)&&!name.endsWith('.test.mjs'))){
    const text=fs.readFileSync(path.join(src,name),'utf8');
    if(name!=='api.mjs'&&/\bfetch\(|ws:\/\/|rikoConfig/.test(text))offenders.push(name);
  }
  assert.deepEqual(offenders,[]);
});
