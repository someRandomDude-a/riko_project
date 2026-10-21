import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const sign=require('../build/mac_sign.cjs');

test('macOS builds sign with the identity electron-builder found, else ad hoc, keeping its per-file options',async()=>{
  const optionsForFile=()=>({hardenedRuntime:true,entitlements:'build/entitlements.mac.plist'}),app='/release/mac-arm64/Riko.app';
  assert.deepEqual(sign.signOptions({app,optionsForFile,identity:undefined}),{app,optionsForFile,identity:'-',identityValidation:false});
  assert.equal(sign.signOptions({app,optionsForFile,identity:'0123456789ABCDEF0123456789ABCDEF01234567'}).identity,'0123456789ABCDEF0123456789ABCDEF01234567');
  assert.equal(typeof sign.osxSign().signAsync,'function');
  // As electron-builder calls it, with the packager second: osx-sign itself runs, and refuses an app that does not exist.
  sign.retry.delay=0;
  await assert.rejects(sign({app:'/nonexistent/Riko.app',optionsForFile,platform:'darwin'},{platform:{nodeName:'darwin'}}),error=>!/is not a function/.test(error.message)&&/Riko\.app/.test(error.message));
  // electron-builder takes a module's `sign` export over the module itself, so the hook must not have one.
  assert.equal(typeof sign,'function');assert.equal(sign.sign,undefined);
});
