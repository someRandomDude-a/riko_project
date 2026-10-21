import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {beginPopupDrag,movePopupDrag} from './popup_drag.mjs';
const {overlayInput,pointerForwarder,forwardTargets}=createRequire(import.meta.url)('../overlay_input.cjs');
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
test('Linux forwarding hands moved cursor positions to each click-through window under it',()=>{
 let cursor={x:5,y:5},targets=[],timer=null,cancelled=0;const sent=[];
 const overlay={name:'overlay',bounds:{x:0,y:0,width:1920,height:1080}},dock={name:'dock',bounds:{x:1840,y:1010,width:88,height:88}};
 const forward=pointerForwarder({cursor:()=>cursor,targets:()=>targets,bounds:w=>w.bounds,send:(w,point)=>sent.push([w.name,point]),
  schedule:fn=>(timer=fn),cancel:()=>{cancelled++;timer=null;}});
 targets=[overlay,dock];forward.start();forward.start();
 assert.deepEqual(sent,[['overlay',{x:5,y:5}]]);
 timer();assert.equal(sent.length,1,'an unmoved cursor sends nothing');
 cursor={x:1850,y:1020};timer();
 assert.deepEqual(sent.slice(1),[['overlay',{x:1850,y:1020}],['dock',{x:10,y:10}]]);
 targets=[dock];cursor={x:100,y:100};timer();assert.equal(sent.length,3,'a window taking the mouse, or not under the cursor, gets nothing');
 targets=[overlay,dock];cursor={x:1928,y:1098};timer();assert.equal(sent.length,3,'right and bottom edges are outside');
 cursor={x:7,y:9};timer();assert.equal(sent.length,4);
 forward.stop();assert.equal(cancelled,1);assert.equal(timer,null);
 forward.start();assert.deepEqual(sent.slice(4),[['overlay',{x:7,y:9}]],'a restart forwards the current position again');
});
test('forwarded moves go only to windows ignoring the mouse that nothing else covers',()=>{
 const window=(name,bounds,visible=true)=>({name,getBounds:()=>bounds,isVisible:()=>visible,isDestroyed:()=>false});
 const overlay=window('overlay',{x:0,y:0,width:1920,height:1080}),control=window('control',{x:1800,y:950,width:88,height:88}),board=window('board',{x:100,y:100,width:400,height:300});
 const names=options=>forwardTargets({overlay,control,covering:[board],point:{x:1810,y:960},overlayInteractive:false,controlIgnoring:true,gesture:null,...options}).map(w=>w.name);
 assert.deepEqual(names({}),['overlay','control'],'a click-through dock passes the mouse to the overlay too');
 assert.deepEqual(names({controlIgnoring:false}),[],'a chat taking the mouse covers the overlay and gets real events');
 assert.deepEqual(names({overlayInteractive:true}),['control']);assert.deepEqual(names({gesture:{}}),['overlay'],'never a dock being dragged');
 assert.deepEqual(names({point:{x:150,y:150}}),['control'],'the whiteboard covers the overlay there');
 assert.deepEqual(names({covering:[window('hidden',{x:0,y:0,width:1920,height:1080},false),null]}),['overlay','control'],'hidden or missing windows cover nothing');
 assert.deepEqual(forwardTargets({point:{x:1,y:1},overlay:window('o',{x:0,y:0,width:9,height:9},false),control:null,controlIgnoring:false}),[]);
});
