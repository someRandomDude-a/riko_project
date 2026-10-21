import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import fs from 'node:fs';
import {simpleHelp} from './plain_language.mjs';
const {windowAction}=createRequire(import.meta.url)('../window_actions.cjs');

test('custom controls minimize, toggle maximize and close through existing window handling',()=>{
  const calls=[];let maximized=false;
  const w={minimize:()=>calls.push('minimize'),maximize:()=>{maximized=true;calls.push('maximize');},unmaximize:()=>{maximized=false;calls.push('restore');},close:()=>calls.push('close'),isMaximized:()=>maximized,isDestroyed:()=>false};
  windowAction(w,'minimize');assert.equal(windowAction(w,'maximize').maximized,true);assert.equal(windowAction(w,'maximize').maximized,false);windowAction(w,'close');
  assert.deepEqual(calls,['minimize','maximize','restore','close']);assert.throws(()=>windowAction(w,'quit'));
});
test('simple help explains effects without exposing the full technical note',()=>{
  const field={path:'runtime.n_ctx',help:'KV pool demand etc.'};
  assert.match(simpleHelp(field),/Larger values use more memory/);
  assert.equal(field.help,'KV pool demand etc.');
  assert.match(simpleHelp({path:'custom',kind:'boolean',help:'JSON override.'}),/Turn this feature on or off/);
});
// The tray and window icon path (resources/assets/tray.png when packaged) is checked in main_wiring.test.mjs.
test('authored branding images come from the root assets folder',()=>{
 const branding=fs.readFileSync(new URL('./branding.mjs',import.meta.url),'utf8');
 assert.match(branding,/\.\.\/\.\.\/assets\/logo.svg/);
 assert.ok(fs.statSync(new URL('../../assets/tray.png',import.meta.url)).size>0);
});
