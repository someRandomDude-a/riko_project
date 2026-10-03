import test from 'node:test';
import assert from 'node:assert/strict';
import {borderDefaults,borderEffect,normalizeBorders} from './border_effects.mjs';
const voice={enabled:true,phase:'waiting',level:1};
test('default mic, screen and dock borders have independent triggers',()=>{
 assert.equal(borderEffect('mic',borderDefaults,voice).active,true);
 assert.equal(borderEffect('screen',borderDefaults,voice).active,false);
 assert.equal(borderEffect('screen',borderDefaults,voice,{wake:true}).active,true);
 assert.equal(borderEffect('dock',borderDefaults,voice).active,false);
 for(const phase of ['capturing','transcribing'])assert.equal(borderEffect('dock',borderDefaults,{...voice,phase}).active,true);
 assert.equal(borderEffect('mic',borderDefaults,{...voice,enabled:false}).active,false);
});
test('each border is independently disableable and can follow reply generation',()=>{
 assert.equal(borderEffect('mic',{...borderDefaults,micBorderEnabled:false},voice).active,false);
 assert.equal(borderEffect('dock',{...borderDefaults,dockBorderTrigger:'generating'},voice,{generating:true}).active,true);
 assert.equal(borderEffect('screen',{...borderDefaults,screenBorderTrigger:'capturing'},{...voice,phase:'capturing'}).active,true);
});
test('audio response is optional and solid pulse can be enabled separately',()=>{
 assert.equal(borderEffect('mic',borderDefaults,voice).style['--effect-speed'],'4s');
 const reactive=borderEffect('mic',{...borderDefaults,micBorderActivity:true,micBorderStyle:'solid'},voice);
 assert.equal(reactive.style['--effect-speed'],'1s');assert.match(reactive.className,/effect-pulse/);
 assert.match(borderEffect('mic',{...borderDefaults,micBorderPulse:true,micBorderStyle:'solid'},voice).className,/effect-pulse/);
});
test('border preferences reject invalid values and preserve independent widths',()=>{
 const p=normalizeBorders({micBorderWidth:99,dockBorderWidth:2,screenBorderSpeed:NaN,screenBorderTrigger:'invalid',micBorderColor:'bad'});
 assert.equal(p.micBorderWidth,12);assert.equal(p.dockBorderWidth,2);assert.equal(p.screenBorderSpeed,4);assert.equal(p.screenBorderTrigger,'wake');assert.equal(p.micBorderColor,borderDefaults.micBorderColor);
});
