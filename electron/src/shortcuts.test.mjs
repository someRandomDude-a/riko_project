import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {DEFAULT_SHORTCUTS,shortcutBindings}=require('../shortcuts.cjs');
const names=['popup','quit','whiteboard','settings','mic','audio','sleep'];

test('only the summon shortcut is global by default; no OS or browser chord is taken',()=>{
  assert.deepEqual(shortcutBindings(undefined,names),[['popup','CommandOrControl+Shift+Space']]);
  assert.deepEqual(Object.keys(DEFAULT_SHORTCUTS),['popup']);
  for(const chord of Object.values(DEFAULT_SHORTCUTS))assert.doesNotMatch(chord,/Shift\+[QW]$/);
});
test('configured shortcuts bind actions, and null, empty or false unbinds even the default',()=>{
  assert.deepEqual(shortcutBindings({quit:'CommandOrControl+Shift+Q',mic:' Alt+M ',unknown:'Alt+U'},names),[['popup','CommandOrControl+Shift+Space'],['quit','CommandOrControl+Shift+Q'],['mic','Alt+M']]);
  for(const off of [null,'',false,0])assert.deepEqual(shortcutBindings({popup:off},names),[]);
  assert.deepEqual(shortcutBindings(['Alt+X'],names),[['popup','CommandOrControl+Shift+Space']]);
});
