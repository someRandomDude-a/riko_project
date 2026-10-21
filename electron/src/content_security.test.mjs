import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {devCsp} from '../vite.config.mjs';
const require=createRequire(import.meta.url);
const {DEFAULT_BACKEND,originForPort}=require('../backend_origin.cjs');
const electron=fileURLToPath(new URL('..',import.meta.url));
const html=fs.readFileSync(path.join(electron,'index.html'),'utf8');

// index.html's Content-Security-Policy: the renderer shows text from the model, tasks, Discord and errors, so nothing in it
// may load a remote resource (FormattedText makes remote images links; this refuses any that slip through) or run a script
// the bundle did not ship. The backend origin is dynamic (RIKO_PORT, RIKO_BACKEND_URL, a free port when packaged) and a meta
// tag cannot read it, so the policy names loopback with any port; backend_origin.cjs admits only http://127.0.0.1:<port>.
const EXPECTED={
  'default-src':["'self'"],'script-src':["'self'"],'style-src':["'self'","'unsafe-inline'"],
  'img-src':["'self'",'data:','blob:','http://127.0.0.1:*'],'media-src':["'self'",'blob:','http://127.0.0.1:*'],
  'font-src':["'self'",'data:'],'connect-src':["'self'",'data:','blob:','http://127.0.0.1:*','ws://127.0.0.1:*'],
  'object-src':["'none'"],'base-uri':["'none'"],'form-action':["'none'"],'frame-src':["'none'"]};
const cspOf=text=>{
  const match=/<meta http-equiv="Content-Security-Policy" content="([^"]*)">/.exec(text);
  assert.ok(match,'index.html has a CSP meta');
  return Object.fromEntries(match[1].split(';').map(part=>part.trim().split(/\s+/)).filter(part=>part[0]).map(([name,...sources])=>[name,sources]));
};
const policy=cspOf(html);

// A small CSP source matcher (CSP3 as Chromium applies it, enough for the sources above): 'self' is the page's own origin,
// which for a file:// page is any file: URL (and never blob: or data:); a scheme source matches the scheme; a host source matches
// scheme (or its secure upgrade), host and port.
function allows(directive,url,page){
  const sources=policy[directive]??policy['default-src'],target=new URL(url),self=new URL(page);
  return sources.some(source=>{
    if(source==="'none'"||source.startsWith("'unsafe"))return false;
    // blob: and data: never match 'self' in Chromium (their scheme is not the page's); they need their own scheme source.
    if(source==="'self'")return self.protocol==='file:'?target.protocol==='file:':target.protocol===self.protocol&&target.host===self.host;
    if(/^[a-z][a-z0-9+.-]*:$/.test(source))return target.protocol===source;
    const host=/^([a-z]+):\/\/([^:/]+)(?::(\*|[0-9]+))?$/.exec(source);
    const schemes={http:['http:','https:'],ws:['ws:','wss:']}[host?.[1]]??[host?.[1]+':'];  // CSP3 lets http: upgrade to https:
    return !!host&&schemes.includes(target.protocol)&&target.hostname===host[2]&&(host[3]==='*'||target.port===(host[3]||''));
  });
}

test('index.html is a standards-mode page with its charset before the policy',()=>{
  assert.match(html,/^<!doctype html>\n<html lang="en">/);
  const charset=html.indexOf('<meta charset="utf-8">'),csp=html.indexOf('http-equiv="Content-Security-Policy"'),script=html.indexOf('<script');
  assert.ok(charset>0&&charset<csp&&csp<script,'charset, then the CSP, then the only script');
  assert.doesNotMatch(html,/<title>/);  // a <title> would override each window's title option
});

test('the policy is exactly the reviewed one: no inline or eval script, no host beyond loopback',()=>{
  assert.deepEqual(policy,EXPECTED);
  for(const value of policy['script-src'])assert.doesNotMatch(value,/unsafe|data:|blob:|\*/);
  for(const [name,sources] of Object.entries(policy))for(const source of sources){
    assert.notEqual(source,'*',name);
    assert.ok(!/^(https?|wss?):$/.test(source),`${name} allows every ${source} URL`);
    if(/:\/\//.test(source))assert.match(source,/^(http|ws):\/\/127\.0\.0\.1:\*$/,name);
  }
});

const pages={development:'http://localhost:5173/#/control',packaged:pathToFileURL(path.join(electron,'dist','index.html')).href+'#/control'};
const backends=[DEFAULT_BACKEND,originForPort(1024).origin,originForPort(65535).origin,originForPort(49152).origin];
for(const [kind,page] of Object.entries(pages))test(`every load the ${kind} renderer makes is allowed`,()=>{
  const asset=name=>new URL('./assets/'+name,page).href;
  for(const name of ['index-abc.js','motion-abc.js'])assert.ok(allows('script-src',asset(name),page),name);
  assert.ok(allows('style-src',asset('index-abc.css'),page));
  for(const url of [asset('KaTeX_Main-Regular-abc.woff2'),'data:font/woff;base64,AAAA'])assert.ok(allows('font-src',url,page),url);  // KaTeX
  assert.ok(allows('img-src','data:image/svg+xml,%3Csvg%3E',page));  // the logo Vite inlines
  for(const origin of backends){
    const socket=origin.replace(/^http/,'ws');
    for(const url of [origin+'/api/media?path=character_files%2Fa.png'])assert.ok(allows('img-src',url,page),url);  // whiteboard, Markdown
    assert.ok(allows('media-src',origin+'/api/media?path=effects%2Fa.mp4',page));  // effects video
    // request(), GLTFLoader (/api/avatar/model), VRMA (/api/avatar/animation, /api/animation/assets) and the event socket.
    for(const url of [origin+'/api/status',origin+'/api/avatar/model',origin+'/api/animation/assets/x.vrma',socket+'/ws/events',socket+'/ws/chat'])
      assert.ok(allows('connect-src',url,page),url);
  }
  // GLTFLoader turns a VRM's embedded textures into blob: URLs it fetches, and fetches embedded data: buffers.
  for(const url of ['blob:'+new URL(page).origin+'/1234','data:application/octet-stream;base64,AAAA'])assert.ok(allows('connect-src',url,page),url);
  assert.ok(allows('img-src','blob:'+new URL(page).origin+'/1234',page));
});

for(const [kind,page] of Object.entries(pages))test(`the ${kind} renderer refuses remote and injected loads`,()=>{
  for(const url of ['https://attacker.example/leak.png?k=v','http://attacker.example/x.png','http://localhost:9000/x.png','http://127.0.0.2:9000/x.png'])
    assert.ok(!allows('img-src',url,page),url);
  for(const url of ['https://attacker.example/x','wss://attacker.example/x','ws://localhost:9000/x','ws://127.0.0.2:9000/x'])assert.ok(!allows('connect-src',url,page),url);
  for(const url of ['https://cdn.example/x.js','data:text/javascript,alert(1)','blob:'+new URL(page).origin+'/1','http://127.0.0.1:9000/x.js'])
    assert.ok(!allows('script-src',url,page),url);
  for(const url of ['https://fonts.example/x.woff2'])assert.ok(!allows('font-src',url,page),url);
  for(const url of ['https://attacker.example/','http://127.0.0.1:9000/'])assert.ok(!allows('frame-src',url,page)&&!allows('object-src',url,page),url);
});

test('Vite adds its HMR socket to the policy only while it serves the page',()=>{
  assert.equal(devCsp.apply,'serve');  // never in a build: the packaged page keeps the file's policy
  const served=cspOf(devCsp.transformIndexHtml(html));
  assert.deepEqual(served,{...EXPECTED,'connect-src':[...EXPECTED['connect-src'],'ws://localhost:*']});
});

test('the renderer source loads nothing the policy refuses',()=>{
  const files=fs.readdirSync(path.join(electron,'src')).filter(name=>/\.(jsx|mjs|css)$/.test(name)&&!name.endsWith('.test.mjs'));
  for(const name of files){
    const text=fs.readFileSync(path.join(electron,'src',name),'utf8');
    assert.doesNotMatch(text,/\bnew Worker\b|\bnew Function\b|\beval\(|importScripts|WebAssembly\.(instantiate|compile)/,name);  // script-src 'self'
    assert.doesNotMatch(text,/url\(\s*['"]?(https?:)?\/\/|@import\s+(url\()?['"]?https?:/i,name);  // remote CSS (fonts, images)
    assert.doesNotMatch(text,/\bsrc=["']https?:\/\//i,name);  // a remote image or media literal
  }
});

// After npm run build: Vite keeps the policy and puts the bundle's script and stylesheet after it (a meta CSP governs only what
// follows it) and inlines no script.
const built=path.join(electron,'dist','index.html');
test('the built page keeps the policy ahead of the bundle',{skip:!fs.existsSync(built)&&'run npm run build first'},()=>{
  const text=fs.readFileSync(built,'utf8');
  assert.deepEqual(cspOf(text),EXPECTED);
  const csp=text.indexOf('Content-Security-Policy');
  for(const match of text.matchAll(/<(script|link)\b[^>]*>/g))assert.ok(match.index>csp,match[0]);
  for(const match of text.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)){
    assert.match(match[1],/\bsrc="\.\/assets\//,match[0]);
    assert.equal(match[2].trim(),'',match[0]);
  }
});
