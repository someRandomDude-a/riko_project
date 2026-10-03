import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {beginPopupDrag,movePopupDrag} from './popup_drag.mjs';
const {overlayInput}=createRequire(import.meta.url)('../overlay_input.cjs');
test('drag keeps overlay interactive when hover leaves, without enabling focus',()=>{
 const changes=[],input=overlayInput(on=>changes.push(on));
 input.hover('avatar',true);input.drag('avatar',true);input.hover('avatar',false);
 assert.equal(input.interactive,true);input.release();assert.equal(input.buttons,0);
 input.drag('avatar',false);assert.deepEqual(changes,[true,false]);
});
test('popup drag uses viewport coordinates and clamps to screen',()=>{
 const drag=beginPopupDrag({x:100,y:100},{left:40,top:50});
 const position=movePopupDrag(drag,{x:130,y:140},{width:80,height:60},{width:500,height:400});
 assert.equal(position.left,70);assert.equal(position.top,90);
});
