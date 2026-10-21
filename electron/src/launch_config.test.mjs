import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {parseConfig,launchSettings}=require('../launch_config.cjs');
const {shortcutBindings}=require('../shortcuts.cjs');
// tests/test_electron_settings.py checks these readings against PyYAML itself.

test('yes, no, on and off are booleans, as PyYAML reads them; y and n and quoted words stay text',()=>{
  for(const word of ['yes','Yes','YES','on','On','ON','true','True'])assert.equal(parseConfig(`v: ${word}\n`).v,true,word);
  for(const word of ['no','No','NO','off','Off','OFF','false','False'])assert.equal(parseConfig(`v: ${word}\n`).v,false,word);
  for(const word of ['y','Y','n','N','"yes"',"'off'"])assert.equal(typeof parseConfig(`v: ${word}\n`).v,'string',word);
});

test('numbers, dates and repeated keys follow PyYAML, not the wider YAML 1.1 of the yaml package',()=>{
  assert.deepEqual(parseConfig('a: 017\nb: 1:30\nc: 0x1F\nd: ~\ne: 1_000\nf: 1.0e+3\ng: .5\n'),{a:15,b:90,c:31,d:null,e:1000,f:1000,g:.5});
  assert.deepEqual(parseConfig('a: 09\nb: 1e3\nc: 1.0e3\nd: 0:30\ne: 2024-1-5\nf: -.5\n'),{a:'09',b:'1e3',c:'1.0e3',d:'0:30',e:'2024-1-5',f:'-.5'});
  assert.ok(parseConfig('d: 2024-01-05\n').d instanceof Date);
  assert.deepEqual(parseConfig('desktop:\n  debug: false\ndesktop:\n  debug: true\n'),{desktop:{debug:true}},'the last of a repeated key, where YAML 1.2 refused the file');
});

test('the launch settings follow the backend: debug: yes opens debug mode, off unbinds a shortcut, auto_start: no stays off',()=>{
  const settings=launchSettings('desktop:\n  debug: yes\n  shortcuts:\n    popup: off\n    mic: Alt+M\npresets:\n  default:\n    name: Mika\nsovits_ping_config:\n  auto_start: no\n');
  assert.equal(settings.debug,true);assert.equal(settings.name,'Mika');assert.equal(settings.sovits.auto_start,false);
  assert.deepEqual(shortcutBindings(settings.shortcuts,['popup','mic']),[['mic','Alt+M']]);  // YAML 1.2 read popup as the text 'off'
  assert.equal(launchSettings('desktop:\n  debug: "yes"\n').debug,false,'only a boolean opens debug mode');
});

test('the name is the backend\'s, as text, and a missing or malformed file reads as defaults',()=>{
  assert.equal(launchSettings("presets:\n  default:\n    name: 'Yes'\n").name,'Yes');
  assert.equal(launchSettings('presets:\n  default:\n    name: Yes\n').name,'True','PyYAML reads an unquoted Yes as True');
  assert.equal(launchSettings('presets:\n  default:\n    name: 42\n').name,'42');
  assert.equal(launchSettings('character_name: Mika\n').name,'Mika','the older top-level name, as load_config falls back to it');
  assert.equal(launchSettings('character_name: Mika\npresets:\n  default:\n    name: Aoi\n').name,'Aoi');
  for(const source of ['','- a list\n','plain text\n','presets: 3\n','presets:\n  default:\n    name: [a]\n','desktop: 2024-01-01\n']){
    const settings=launchSettings(source);
    assert.equal(settings.name,'',source);assert.equal(settings.debug,false,source);assert.equal(settings.shortcuts,undefined,source);
  }
  assert.throws(()=>launchSettings('desktop: [\n'),'a file the backend cannot read either stops the launch, as before');
});
