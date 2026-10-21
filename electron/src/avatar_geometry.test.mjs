import test from 'node:test';
import assert from 'node:assert/strict';
import {slidePixels,dragPosition,resizeDrag} from './avatar_geometry.mjs';

const VIEWPORT={width:3840,height:2160};  // big enough that resizeHeldAvatar's clamp to the screen never moves the canvas off the cursor
// A point's screen position from its canvas NDC (x right, y up) and the canvas geometry (screen px, y down).
const screen=(ndc,g)=>({x:g.x+(ndc.x+1)/2*g.width,y:g.y+(1-ndc.y)/2*g.height});
const ndcOf=(point,g)=>({x:(point.x-g.x)/g.width*2-1,y:1-(point.y-g.y)/g.height*2});

test('the scruff slide in px flips NDC y (up) to screen y (down) and scales with the canvas',()=>{
 assert.deepEqual(slidePixels({x:.5,y:.5},{width:480,height:720}),{x:120,y:-180});
 assert.deepEqual(slidePixels({x:-.25,y:-.1},{width:960,height:1440}),{x:-120,y:72});
 assert.deepEqual(slidePixels(null,{width:480,height:720}),{x:0,y:0});
});

test('a slide of cursor NDC minus S NDC puts S under the cursor, and the release geometry is the last placed one',()=>{
 const g={x:600,y:200,width:480,height:720},s={x:.05,y:.42},cursor={x:700,y:830};  // grabbed by the foot, S is the upper chest
 const drag={x:cursor.x,y:cursor.y,left:g.x,top:g.y,geometry:g},target=ndcOf(cursor,g),slide={x:target.x-s.x,y:target.y-s.y};
 for(const u of [0,.5,1]){const slid=slidePixels({x:slide.x*u,y:slide.y*u},g),placed={...g,...dragPosition(drag,cursor,slid)},at=screen(s,placed);
  if(u===1)assert.ok(Math.hypot(at.x-cursor.x,at.y-cursor.y)<=1,`S at ${JSON.stringify(at)}, the cursor at ${JSON.stringify(cursor)}`);
  else assert.ok(at.y<cursor.y-100,'S on its way up to the cursor');}
 const moved={x:900,y:500},slid=slidePixels(slide,g);assert.deepEqual(dragPosition(drag,moved,slid),{x:g.x+200+slid.x,y:g.y-330+slid.y});
});

test('a wheel resize mid-slide keeps the slide in NDC and S still converges under the cursor',()=>{
 const g={x:600,y:200,width:480,height:720},s={x:.05,y:.42},cursor={x:700,y:830},target=ndcOf(cursor,g),slide={x:target.x-s.x,y:target.y-s.y};
 let drag={x:cursor.x,y:cursor.y,point:{...cursor},left:g.x,top:g.y,geometry:g};
 const half={x:slide.x/2,y:slide.y/2};drag.geometry={...g,...dragPosition(drag,cursor,slidePixels(half,g))};
 for(const delta of [-200,300]){
  drag=resizeDrag(drag,delta,cursor,VIEWPORT,half);
  assert.deepEqual(dragPosition(drag,cursor,drag.slid),{x:drag.geometry.x,y:drag.geometry.y},'until the cursor moves, the drag keeps the resized geometry');
  const done={...drag.geometry,...dragPosition(drag,cursor,slidePixels(slide,drag.geometry))},at=screen(s,done);
  assert.ok(Math.hypot(at.x-cursor.x,at.y-cursor.y)<=2,`after a ${delta} resize S ends ${Math.hypot(at.x-cursor.x,at.y-cursor.y).toFixed(2)} px from the cursor`);
 }
 assert.notEqual(drag.geometry.width,480);
});
