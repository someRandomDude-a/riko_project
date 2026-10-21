import test from 'node:test';
import assert from 'node:assert/strict';
import {Drop,Release,workArea,floorTop,flightWalls,fallGravity,chooseRelease,setDownTarget,hopHeight,THROW,FLOOR_SNAP} from './avatar_drop.mjs';
import {HoldBehaviour} from './avatar_hold.mjs';

const ground=(bottom,foot=0)=>({bottom,foot});
const run=(drop,floor,seconds,dt=1/60)=>{const steps=[];for(let i=0,n=Math.round(seconds/dt);i<n;i++){steps.push(drop.step(dt,ground(floor)));if(drop.standing&&steps.at(-1).landed)break;}return steps;};

test('free fall from rest covers g t^2 / 2 and lands exactly on the floor with impact g t, at any frame rate',()=>{
 const g=2500,floor=900;
 for(const dt of [1/60,1/144,1/30]){
  const drop=new Drop({x:100,y:100},{fall:true,gravity:g,bottom:floor});assert.equal(drop.flying,true);
  assert.ok(Math.abs(drop.seconds-Math.sqrt(2*800/g))<1e-9,'the expected flight time');
  let t=0;for(let i=0;i<12;i++){drop.step(dt);t+=dt;}
  assert.ok(Math.abs((drop.y-100)-.5*g*t*t)<=.01*.5*g*t*t,`after ${t.toFixed(3)} s: ${(drop.y-100).toFixed(1)} px against ${(.5*g*t*t).toFixed(1)}`);
  const steps=run(drop,floor,2,dt),last=steps.at(-1),tFall=Math.sqrt(2*800/g);
  assert.equal(last.landed,true);assert.equal(drop.y,floor);assert.equal(drop.standing,true);assert.equal(drop.seconds,0);
  assert.ok(Math.abs(last.impact-g*tFall)<1e-6*g,`impact ${last.impact.toFixed(2)} against ${(g*tFall).toFixed(2)}`);
  assert.equal(steps.filter(s=>s.landed).length,1,'it lands once');
 }
});

test('the floor is the work area bottom minus her foot line, re-measured every step',()=>{
 const drop=new Drop({x:0,y:0},{fall:true,gravity:2500,bottom:1040,foot:600});assert.ok(Math.abs(drop.seconds-Math.sqrt(2*440/2500))<1e-9);
 let s;do s=drop.step(1/60,{bottom:1040,foot:610});while(!s.landed);assert.equal(s.y,430,'landed on 1040 - 610');
 assert.equal(drop.step(1/60,{bottom:1040,foot:620}).y,420,'standing on the floor follows the foot line');assert.equal(drop.step(1/60,{bottom:1000,foot:620}).y,380,'and the floor');
});

test('an upward throw rises, then falls; drag slows only the sideways speed; the speed is capped',()=>{
 const drop=new Drop({x:0,y:500,vx:1000,vy:-1500},{fall:true,gravity:2500,bottom:800});
 const ys=[],xs=[];for(let i=0;i<120&&drop.flying;i++){const s=drop.step(1/60);ys.push(s.y);xs.push(s.x);}
 const apex=ys.indexOf(Math.min(...ys));assert.ok(apex>0&&apex<ys.length-1&&ys[apex]<500-400,`apex ${ys[apex].toFixed(0)} at frame ${apex}`);
 assert.ok(Math.abs(ys[apex]-(500-1500**2/(2*2500)))<15,'the apex is v^2 / 2g above the release');
 assert.equal(drop.standing,true);assert.ok(xs.every((x,i)=>!i||x>xs[i-1]),'she keeps moving sideways');
 const t=Math.round(.5*60)/60,expected=1000*(1-Math.exp(-THROW.drag*t))/THROW.drag;assert.ok(Math.abs(xs[Math.round(.5*60)-1]-expected)<1e-6,'exact exponential air drag');
 const fast=new Drop({x:0,y:0,vx:9000,vy:-9000},{fall:true,bottom:5000});assert.ok(Math.abs(Math.hypot(fast.vx,fast.vy)-THROW.maxSpeed)<1e-9);
});

test('the sides and top reflect her speed with restitution 0.35 and report each hit once; live walls move in with her arms',()=>{
 const drop=new Drop({x:500,y:400,vx:-3000,vy:0},{fall:true,gravity:100,walls:{left:0,right:1000,top:0},bottom:1e6});
 let hit=null,t=0;while(!hit&&t<1){const s=drop.step(1/60);t+=1/60;if(s.hits.length)hit={...s,vx:drop.vx};}
 assert.deepEqual(hit.hits,['left']);assert.ok(hit.x>=0&&hit.x<20,`she ends the frame just off the wall (${hit.x.toFixed(1)} px)`);assert.ok(hit.vx>0&&Math.abs(hit.vx-THROW.restitution*3000*Math.exp(-THROW.drag*t))<.05*hit.vx,`bounced at ${hit.vx.toFixed(0)} px/s`);
 assert.ok(drop.step(1/60).x>0,'and moves away from the wall');
 const up=new Drop({x:500,y:100,vx:0,vy:-2000},{fall:true,gravity:2500,walls:{left:0,right:1000,top:0},bottom:900}),hits=[];
 while(up.flying)hits.push(...up.step(1/60).hits);assert.deepEqual(hits,['top']);assert.equal(up.y,900);
 // her arms fling out on the way: the box grows on the leading side, so the wall she meets is 60 px further in
 const live=new Drop({x:300,y:100,vx:-2000},{fall:true,gravity:100,walls:{left:0,right:1000,top:0},bottom:1e6});let first=null;
 for(let i=0;i<60&&!first;i++){const s=live.step(1/60,{walls:{left:i<3?0:60,right:1000,top:0}});if(s.hits.length)first=s;}
 assert.deepEqual(first.hits,['left']);assert.ok(first.x>=60&&first.x<80,`stopped at the moved wall (${first.x.toFixed(1)})`);
});

test('a release beyond a wall is never pushed back in, until she is back inside it',()=>{
 const drop=new Drop({x:-200,y:100,vx:-100},{fall:true,gravity:100,walls:{left:0,right:1000,top:0},bottom:1e6});assert.equal(drop.walls.left,-200);
 drop.step(1/60,{walls:{left:0,right:1000,top:0}});assert.ok(drop.x>=-200&&drop.x<0,'the wall stays out at her');
 const back=new Drop({x:-50,y:100,vx:3000},{fall:true,gravity:100,walls:{left:0,right:1000,top:0},bottom:1e6});
 for(let i=0;i<10;i++)back.step(1/60,{walls:{left:0,right:1000,top:0}});assert.equal(back.walls.left,0,'back inside, the wall holds again');
});

test('a throw from the floor (at, below, or within the snap above it) hops above it first, then lands on it; the hop grows with the speed',()=>{
 for(const y of [900,977,1256]){
  const drop=new Drop({x:0,y,vx:2500,vy:200},{fall:true,gravity:2450,bottom:900,hop:hopHeight(2500)});assert.equal(drop.flying,true);
  const ys=[];let s;do{s=drop.step(1/60);ys.push(s.y);}while(!s.landed&&ys.length<200);
  assert.equal(s.landed,true);assert.equal(drop.y,900);assert.ok(Math.abs(Math.min(...ys)-(900-100))<3,`it rose to ${Math.min(...ys).toFixed(1)}, 100 px above the floor`);
  assert.ok(drop.x>300,'and flew sideways');
 }
 const near=new Drop({x:0,y:894,vx:2000},{fall:true,gravity:2450,bottom:900,snap:43,hop:80});let n;do n=near.step(1/60);while(!n.landed);assert.ok(near.x>500,`6 px above the floor a flick still flies (${near.x.toFixed(0)} px)`);
 const high=new Drop({x:0,y:800,vx:2000},{fall:true,gravity:2450,bottom:900,snap:43,hop:80});assert.equal(high.vy,0,'100 px above it she simply falls');
 assert.equal(hopHeight(0),40);assert.equal(hopHeight(2500),100);assert.equal(hopHeight(1e5),160);
 const strong=new Drop({x:0,y:950,vx:0,vy:-3000},{fall:true,gravity:2500,bottom:900,hop:40});assert.equal(strong.vy,-3000,'a stronger upward flick keeps its own speed');
});

test('no flight on a set-down: she glides up onto the floor; one just above it lands on it, measured with or without the scruff slide; elsewhere she stays',()=>{
 for(const y of [900,1250]){const drop=new Drop({x:0,y},{bottom:900});assert.equal(drop.flying,false);
  const ys=[];let landed=0;while(!drop.standing){const s=drop.step(1/60);ys.push(s.y);if(s.landed){landed++;assert.equal(s.impact,null,'a glide lands with no impact');}}
  assert.equal(drop.y,900);assert.ok(ys.length<=Math.ceil(.45*60));assert.equal(landed,y===900?0:1,'a glide reports its touchdown once');
  for(let i=1;i<ys.length;i++)assert.ok(ys[i]<=ys[i-1]+1e-9,'only upward');}
 const below=new Drop({x:0,y:1250},{bottom:900}),steps=[];assert.ok(Math.abs(below.seconds-below.duration)<1e-12);while(!below.standing)steps.push(below.step(1/60).y);
 assert.ok(Math.max(...steps.map((y,i)=>i?steps[i-1]-y:1250-y))<1.875*350/below.duration/60+1,'a smootherstep glide, never a jump');
 const near=new Drop({x:0,y:870},{bottom:900,snap:40});assert.equal(near.gliding,true);while(!near.standing)near.step(1/60);assert.equal(near.y,900);
 const mid=new Drop({x:0,y:400},{bottom:900,snap:40});assert.equal(mid.standing,true);mid.step(1/60);assert.equal(mid.y,400,'set down where she is');
 // picked up off the floor by the head: the slide lifted the canvas 62 px; let go there she lands back on the floor
 const head=new Drop({x:0,y:838},{bottom:900,snap:40,slid:-62});assert.equal(head.gliding,true);while(!head.standing)head.step(1/60);assert.equal(head.y,900);
 assert.equal(setDownTarget(400,900,40),400);assert.equal(setDownTarget(861,900,40),900);assert.equal(setDownTarget(950,900,0),900);assert.equal(setDownTarget(808,900,40,-62),900);assert.equal(setDownTarget(700,900,40,-62),700);
});

test('standing she keeps her soles on their line while the camera eases back: on the floor, or where she was set down in mid-air',()=>{
 const drop=new Drop({x:0,y:0},{fall:true,gravity:2500,bottom:600});while(drop.flying)drop.step(1/60);
 assert.equal(drop.step(1/60,ground(590)).y,590);assert.equal(drop.step(1/60,ground(580)).y,580);
 const mid=new Drop({x:0,y:300},{bottom:1040,foot:620});assert.equal(mid.standing,true);assert.equal(mid.onFloor,false);
 assert.equal(mid.step(1/60,{bottom:1040,foot:640}).y,280,'her soles stay at 920 as the foot line grows');assert.equal(mid.step(1/60,{bottom:1000,foot:640}).y,280,'and the floor moving never moves her');
});

test('the work area comes from the screen\'s available rectangle relative to the overlay, cut to the viewport',()=>{
 const viewport={width:1920,height:1080};
 assert.deepEqual(workArea({availLeft:0,availTop:0,availWidth:1920,availHeight:1040},{x:0,y:0},viewport),{left:0,top:0,right:1920,bottom:1040},'a Windows taskbar at the bottom');
 assert.deepEqual(workArea({availLeft:1920,availTop:25,availWidth:1920,availHeight:1000},{x:1920,y:0},viewport),{left:0,top:25,right:1920,bottom:1025},'a second display with a menu bar and a dock');
 assert.deepEqual(workArea({availLeft:0,availTop:25,availWidth:1440,availHeight:875},{x:0,y:25},{width:1440,height:875}),{left:0,top:0,right:1440,bottom:875},'macOS put the overlay below the menu bar');
 assert.deepEqual(workArea({availWidth:1920,availHeight:1040},{x:0,y:0},viewport),{left:0,top:0,right:1920,bottom:1040},'availLeft/availTop missing');
 for(const screen of [null,{},{availWidth:0,availHeight:1040},{availLeft:5000,availTop:0,availWidth:1920,availHeight:1080}])assert.deepEqual(workArea(screen,{x:0,y:0},viewport),{left:0,top:0,right:1920,bottom:1080});
 assert.equal(floorTop({bottom:1040},616),424);
 assert.deepEqual(flightWalls({left:0,right:1920,top:0},{left:160,right:320,top:90}),{left:-160,right:1600,top:-90});
 assert.ok(Math.abs(fallGravity(.00324,.7)-9.8/.00324*.7)<1e-9);assert.ok(fallGravity(0,1)>0);
});

test('chooseRelease: throw falls only at or above the throw speed (from the floor too), place never falls, fall always does except slowly at the floor',()=>{
 assert.equal(chooseRelease({x:600,y:-700},{release:'throw',throwSpeed:900}),'fall');
 assert.equal(chooseRelease({x:500,y:500},{release:'throw',throwSpeed:900}),'place');
 assert.equal(chooseRelease({x:2500,y:0},{release:'throw',throwSpeed:900},{grounded:true}),'fall','a flick off the taskbar flies');
 assert.equal(chooseRelease({x:5000,y:0},{release:'place',throwSpeed:900}),'place');
 assert.equal(chooseRelease({x:0,y:0},{release:'fall'}),'fall');
 assert.equal(chooseRelease({x:100,y:0},{release:'fall'},{grounded:true}),'place');assert.equal(chooseRelease({x:5000,y:0},{release:'fall'},{grounded:true}),'fall');
 assert.equal(chooseRelease({x:NaN,y:undefined},{}),'place');
});

/** A Release with recording hooks, a stand-in carry rig and the real hold. */
function harness({foot=600,camera=false,ragdoll=false}={}){
 const log={places:[],dragging:[],interacts:[],pickup:[]},m={foot,rest:foot};
 const pickup={cameraActive:camera,ragdoll,release(o={}){log.pickup.push(['release',o]);this.ragdoll=false;},airborne(seconds){log.pickup.push(['airborne',seconds]);},land(){log.pickup.push(['land']);}};
 const hold=new HoldBehaviour({random:()=>.5});
 const release=new Release({place:(g,final)=>log.places.push({...g,final}),dragging:v=>log.dragging.push(v),interact:k=>log.interacts.push(k),hold,pickup,
  measure:{area:()=>({left:0,top:0,right:1920,bottom:1040}),foot:rest=>rest?m.rest:m.foot,box:()=>({left:160,right:320,top:90}),worldPerPixel:()=>.0028,height:()=>720}});
 const grab=now=>{hold.press(now);hold.grab(now,{kind:'body'});};
 return {release,hold,pickup,log,m,grab};
}
const settings={release:'throw',throwSpeed:900,fallGravity:.7};

test('Release: a flick throws her, places the canvas every frame and sends the one final place when she has landed and the camera rests',()=>{
 const {release,hold,pickup,log,grab}=harness({camera:true});grab(10);
 assert.equal(release.letGo(10.5,{x:700,y:100,width:480,height:720},{x:1500,y:-300},{settings}),'fall');
 assert.equal(release.flying,true);assert.equal(hold.state,'falling');assert.equal(log.pickup[0][0],'release');assert.equal(log.pickup[0][1].flight,true);
 assert.ok(log.pickup[0][1].seconds>.3&&log.pickup[0][1].seconds<1,'the carry rig learns the expected flight time');
 let now=10.5,landed=false;for(let i=0;i<120&&!landed;i++){now+=1/60;landed=!!release.step(now,1/60)?.landed;}
 assert.ok(landed);assert.equal(hold.state,'landing');assert.deepEqual(log.interacts,['recover']);assert.ok(log.pickup.some(([k])=>k==='land'));
 assert.equal(log.places.filter(p=>p.final).length,0,'the camera is still easing back: no final place yet');assert.equal(release.active,true);
 for(let i=0;i<5;i++){now+=1/60;release.step(now,1/60);}assert.equal(log.places.filter(p=>p.final).length,0);
 pickup.cameraActive=false;now+=1/60;release.step(now,1/60);
 const finals=log.places.filter(p=>p.final);assert.equal(finals.length,1);assert.equal(finals[0].y,1040-600);assert.deepEqual(log.dragging,[false]);assert.equal(release.active,false);
 assert.ok(log.places.slice(0,-1).every(p=>!p.final),'every earlier place was a move only');
});

test('Release: a gentle set-down lands at once in place, or after its glide onto the floor; mid-air it waits for the camera and keeps her soles still',()=>{
 let h=harness();h.grab(10);assert.equal(h.release.letGo(11,{x:700,y:200,width:480,height:720},{x:100,y:0},{settings}),'place');
 assert.equal(h.hold.state,'landing','set down in mid-air: the squash starts now');assert.deepEqual(h.log.interacts,['recover']);
 assert.deepEqual(h.log.places,[{x:700,y:200,width:480,height:720,final:true}]);assert.deepEqual(h.log.dragging,[false]);
 // the foot grab: the scruff slide put the canvas 356 px below the floor; she glides up and lands at the end, not before
 h=harness();h.grab(10);h.release.letGo(11,{x:700,y:796,width:480,height:720},{x:0,y:0},{slid:{x:0,y:356},settings});
 assert.equal(h.hold.state,'grabbed','still held as she was until her feet arrive');assert.equal(h.hold.released,true);assert.deepEqual(h.log.interacts,[]);
 assert.equal(h.log.pickup[0][1].flight,true,'the carry rig treats the glide as a short flight');
 let now=11,steps=0;while(h.hold.state!=='landing'&&steps<60){now+=1/60;h.release.step(now,1/60);steps++;
  if(h.hold.state!=='landing')assert.ok(h.log.places.at(-1).y>440,'no squash while she is still rising');}
 assert.equal(h.log.places.at(-1).y,440);assert.equal(h.log.places.at(-1).final,true);assert.deepEqual(h.log.interacts,['recover']);assert.ok(h.log.pickup.some(([k])=>k==='land'));
 // set down mid-air while the camera is zoomed out: she waits, and as the foot line grows her canvas rises so her soles stay put
 h=harness({camera:true});h.grab(10);h.release.letGo(11,{x:700,y:200,width:480,height:720},{x:0,y:0},{settings});
 assert.equal(h.log.places.filter(p=>p.final).length,0);h.m.foot=630;h.release.step(11.1,1/60);assert.equal(h.log.places.at(-1).y,170);
 h.pickup.cameraActive=false;h.m.foot=640;h.release.step(11.2,1/60);assert.deepEqual(h.log.places.at(-1),{x:700,y:160,width:480,height:720,final:true});
 // just above the floor, measured while zoomed: the soles land on it
 h=harness({foot:640,camera:true});h.grab(10);h.release.letGo(11,{x:700,y:370,width:480,height:720},{x:0,y:0},{settings});assert.equal(h.release.drop.gliding,true);
});

test('Release: a press catches a flight or a glide where she is, sends a standing release\'s place at once, and dispose() saves where she would end',()=>{
 let h=harness();h.grab(10);h.release.letGo(10.5,{x:700,y:100,width:480,height:720},{x:2000,y:0},{settings});for(let i=0;i<6;i++)h.release.step(10.5+i/60,1/60);
 assert.deepEqual(h.release.grab(),{caught:true,grabbed:true});assert.equal(h.release.active,false);assert.equal(h.log.places.filter(p=>p.final).length,0,'caught mid-air: nothing saved');
 h=harness();h.grab(10);h.release.letGo(11,{x:700,y:900,width:480,height:720},{x:0,y:0},{settings});assert.deepEqual(h.release.grab(),{caught:false,grabbed:true});
 h=harness({camera:true});h.grab(10);h.release.letGo(11,{x:700,y:200,width:480,height:720},{x:0,y:0},{settings});
 assert.deepEqual(h.release.grab(),{caught:false,grabbed:false});assert.equal(h.log.places.filter(p=>p.final).length,1,'standing, only waiting for the camera');
 assert.deepEqual(h.release.grab(),{caught:false,grabbed:false},'nothing running');
 h=harness({camera:true});h.m.rest=616;h.grab(10);h.release.letGo(10.5,{x:700,y:100,width:480,height:720},{x:2000,y:-500},{settings});h.release.step(10.52,1/60);
 h.release.dispose();const last=h.log.places.at(-1);assert.equal(last.final,true);assert.equal(last.y,1040-616,'the floor, measured with the camera at rest');assert.equal(h.release.active,false);assert.deepEqual(h.log.dragging,[false]);
 h=harness({camera:true});h.m.rest=616;h.grab(10);h.release.letGo(11,{x:700,y:200,width:480,height:720},{x:0,y:0},{settings});h.release.dispose();assert.equal(h.log.places.at(-1).y,800-616,'a mid-air set-down keeps her soles where they stood');
 h=harness();h.release.dispose();assert.equal(h.log.places.length,0,'nothing running, nothing sent');
});

test('Release: a wall bounce winces her once and tells the carry rig the new time to the contact',()=>{
 const {release,hold,log,grab}=harness();grab(10);release.letGo(10.5,{x:1200,y:200,width:480,height:720},{x:3000,y:-2400},{settings});  // the top 7 frames in, the right side 1 frame later
 const before=log.pickup[0][1].seconds;let now=10.5,blinks=0;for(let i=0;i<20;i++){now+=1/60;release.step(now,1/60);blinks+=(hold.step(now,1/60,{},{enabled:true})?.blinks||[]).filter(b=>b.kind==='startle').length;}
 const airborne=log.pickup.filter(([k])=>k==='airborne');assert.equal(airborne.length,2,'the top, then the side');assert.ok(airborne[0][1]<before,`after the bounce off the top ${airborne[0][1].toFixed(2)} s instead of ${before.toFixed(2)}`);
 assert.equal(blinks,1,'one blink for the corner');
});

test('Release: a flick off the floor hops; the ragdoll\'s own recover is never posted twice; reactions off still fly and land',()=>{
 let h=harness();h.grab(10);assert.equal(h.release.letGo(11,{x:700,y:520,width:480,height:720},{x:2500,y:0},{slid:{x:0,y:80},settings}),'fall');assert.ok(h.release.drop.vy<0,'she hops up off the taskbar');
 h=harness({ragdoll:true});h.grab(10);h.release.letGo(11,{x:700,y:100,width:480,height:720},{x:2000,y:0},{settings});let now=11;
 while(h.release.flying){now+=1/60;h.release.step(now,1/60);}assert.deepEqual(h.log.interacts,[],'the ragdoll\'s release already posted it');
 h=harness({ragdoll:true});h.grab(10);h.release.letGo(11,{x:700,y:200,width:480,height:720},{x:0,y:0},{settings});assert.deepEqual(h.log.interacts,[]);
 h=harness();h.release.letGo(11,{x:700,y:100,width:480,height:720},{x:2000,y:0},{settings});now=11;let landed=false;while(!landed){now+=1/60;landed=h.release.step(now,1/60).landed;}
 assert.equal(h.hold.state,null);assert.deepEqual(h.log.interacts,['recover'],'a flight\'s landing always posts it');
 h=harness();h.release.letGo(11,{x:700,y:200,width:480,height:720},{x:0,y:0},{settings});assert.deepEqual(h.log.interacts,[],'a set-down with no hold reaction posts only the release');
 assert.equal(FLOOR_SNAP,.06);
});
