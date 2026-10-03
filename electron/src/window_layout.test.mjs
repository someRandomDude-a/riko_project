import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {chatBounds,clampBounds}=createRequire(import.meta.url)('../window_layout.cjs');
const area={x:0,y:0,width:1920,height:1080},anchor={x:960,y:900},current={x:916,y:858,width:88,height:100};
test('mini defaults to quarter display width with free height and a square minimum',()=>{
 const mini=chatBounds('compact',current,area,anchor);
 assert.equal(mini.width,480);assert.ok(mini.height>=mini.width);
 const resized=chatBounds('compact',current,area,anchor,{width:100,height:100});
 assert.equal(resized.width,320);assert.equal(resized.height,320);
});
test('all modes preserve the dock centre when it fits on screen',()=>{
 for(const mode of ['full','compact','collapsed']){
   const bounds=chatBounds(mode,current,area,anchor,{width:mode==='full'?800:mode==='compact'?480:88,height:600});
   assert.equal(bounds.x+bounds.width/2,anchor.x);
   assert.ok(Math.abs(bounds.y+bounds.height-(mode==='collapsed'?bounds.height*.6:58)-anchor.y)<1);
 }
});
test('collapsed size is retained and scales as a unit',()=>{
 const dock=chatBounds('collapsed',current,area,anchor,{width:176});
 assert.equal(dock.width,176);assert.equal(dock.height,176);
 assert.ok(Math.abs(dock.y+dock.height-105.6-anchor.y)<1);
});
test('bounds cannot leave the active display and mini remains square on short screens',()=>{
 assert.deepEqual(clampBounds({x:-500,y:2000,width:200,height:300},area),{x:0,y:780,width:200,height:300});
 const mini=chatBounds('compact',current,{...area,height:600},anchor,{width:1000,height:300});
 assert.equal(mini.width,600);assert.equal(mini.height,600);
});
