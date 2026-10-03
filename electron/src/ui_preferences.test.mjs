import {test} from 'node:test';
import assert from 'node:assert/strict';
import {normalizePreferences} from './ui_preferences.mjs';
import {appearanceStyle} from './ui/skins.mjs';

test('default interface uses glass without canned shortcuts',()=>{
  const preferences=normalizePreferences({});
  assert.equal(preferences.skin,'glass');assert.equal(preferences.activity,false);
  assert.deepEqual(preferences.quickActions,[]);assert.equal(preferences.showTaskActions,false);
});
test('appearance values are bounded and corrupted storage is recoverable',()=>{
  const preferences=normalizePreferences({skin:'unknown',background:'url(remote)',glassOpacity:0,gradientAngle:999,quickActions:'invalid'});
  assert.equal(preferences.skin,'glass');assert.match(preferences.background,/^#/);
  assert.equal(preferences.glassOpacity,.35);assert.equal(preferences.gradientAngle,360);assert.deepEqual(preferences.quickActions,[]);
});
test('window materials and explanation preferences are bounded without replacing existing choices',()=>{
  const p=normalizePreferences({skin:'solid',windowOpacity:-1,boardOpacity:9,glassBlur:Infinity,cornerRadius:99,advancedExplanations:'yes',reduceMotion:true});
  assert.equal(p.skin,'solid');assert.equal(p.windowOpacity,.35);assert.equal(p.boardOpacity,1);
  assert.equal(p.glassBlur,24);assert.equal(p.cornerRadius,28);assert.equal(p.advancedExplanations,false);assert.equal(p.reduceMotion,true);
  const style=appearanceStyle(p);assert.equal(style['--window-alpha'],.35);assert.equal(style['--glass-blur'],'24px');
});
test('theme style exposes central CSS variables without global avatar styling',()=>{
  const style=appearanceStyle({skin:'glass',accent:'#123456'});
  assert.equal(style['--accent'],'#123456');assert.ok(!('background' in style));
});
