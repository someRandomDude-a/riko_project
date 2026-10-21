import test from 'node:test';
import assert from 'node:assert/strict';
import {VRMExpression,VRMExpressionManager} from '@pixiv/three-vrm';
import {HoldBehaviour,HoldFace,passiveHold,HELD_STATES} from './avatar_hold.mjs';
import {AvatarFace,FACE_PARTS,INTENTS} from './avatar_face.mjs';

const seeded=(seed=3)=>()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};
const SETTINGS={enabled:true,protestSeconds:6,resignSeconds:18,recoverySeconds:2.5};
// An injected clock: every step is at an exact multiple of dt, so state times are exact.
function scene(options={}){
 const hold=new HoldBehaviour({random:seeded(options.seed)}),dt=options.dt||1/60;let frame=0;const outs=[],events=[];
 const now=()=>frame*dt;
 const run=(seconds,motion={},settings=SETTINGS)=>{let out=null;for(let i=0,n=Math.round(seconds/dt);i<n;i++){frame++;out=hold.step(now(),dt,typeof motion==='function'?motion(now()):motion,settings);outs.push(out);if(out)events.push(...out.events);}return out;};
 return {hold,run,now,outs,events,dt};
}

test('a press pops the brows and blinks; a grab startles for 0.12 s, surprises until 0.9 s and is grabbed for exactly 0.35 s, then dangling',()=>{
 const {hold,run,now}=scene({dt:.01});
 hold.press(now());let out=run(.01);assert.equal(out.state,'pressed');assert.equal(out.face.intent,'poke');assert.deepEqual(out.blinks,[{kind:'press',delay:.03}]);assert.equal(out.pose,null);
 out=run(.1);assert.deepEqual(out.blinks,[]);
 const grabbed=now();hold.grab(grabbed,{region:'torso',kind:'body'});out=run(.01);
 assert.equal(out.state,'grabbed');assert.equal(out.face.intent,'startle');assert.ok(out.suppress>=.9,'spontaneous blinks wait until the surprise is over');assert.equal(out.transition,.25);
 const states=[],faces=[],looks=[],startle=[];for(let i=0;i<120;i++){out=run(.01);states.push(out.state);faces.push(out.face.intent);looks.push(out.look);startle.push(out.pose.startle);}
 const at=t=>Math.round((t-.01)/.01)-1;  // index of the step at t seconds after the grab
 assert.equal(states[at(.34)],'grabbed');assert.equal(states[at(.35)],'dangling');
 assert.equal(faces[at(.11)],'startle');assert.equal(faces[at(.12)],'surprise');assert.equal(faces[at(.89)],'surprise');assert.equal(faces[at(.9)],'worried');
 assert.equal(looks[at(.1)],'viewer');assert.equal(looks[at(.2)],'holder');assert.equal(looks[at(.95)],'viewer');
 assert.ok(Math.abs(startle[at(.12)]-1)<.01&&startle[at(.35)]===0,'the startle rises by 0.12 s and is gone by 0.35 s');
});

test('carried when fast, back to dangling 0.6 s after slowing; four hard reversals in 1.5 s make her dizzy, and 1.2 s after the shaking she is annoyed',()=>{
 const {hold,run,now,events}=scene();hold.grab(now(),{kind:'body'});run(1);assert.equal(hold.state,'dangling');
 run(.2,{vx:1000});assert.equal(hold.state,'carried');assert.equal(run(1/60,{vx:1000}).face.intent,'alarm');
 run(.55,{vx:200});assert.equal(hold.state,'carried');run(.1,{vx:200});assert.equal(hold.state,'dangling');
 assert.equal(run(.2,{ax:9500}).state,'carried','a hard jolt counts as fast too');run(.7);assert.equal(hold.state,'dangling');
 const shake=t=>({vx:Math.sin(t*2*Math.PI*1.6)>0?1500:-1500});run(1.4,shake);assert.equal(hold.state,'dizzy');
 const out=run(1/60,shake);assert.equal(out.pose.limp,1);assert.equal(out.face.intent,'dizzy');assert.equal(out.look,'ahead','dizzy, her eyes rest instead of tracking');
 run(1.1);assert.equal(hold.state,'dizzy');run(.15);assert.equal(hold.state,'dangling');assert.equal(run(1/60).face.intent,'annoyed');run(3);assert.equal(run(1/60).face.intent,'worried');
 assert.deepEqual(events.map(e=>e.to),['grabbed','dangling','carried','dangling','carried','dangling','carried','dizzy','dangling']);
});

test('6 s still she protests and kicks in bursts, a 25 px move resets it, 18 s still she resigns with a sigh, and a move from resigned is carried',()=>{
 const {hold,run,now}=scene();hold.grab(now(),{kind:'body'});
 run(5.9);assert.equal(hold.state,'dangling');run(.12);assert.equal(hold.state,'protesting');
 let burst=0,inside=0,gap=0;
 for(let i=0;i<60*11.5;i++){const out=run(1/60),t=now();
  assert.ok(hold.burstEnd-hold.burstStart>=.9&&hold.burstEnd-hold.burstStart<=1.5,'bursts last 0.9-1.5 s');assert.ok(hold.nextBurst-hold.burstEnd>=2&&hold.nextBurst-hold.burstEnd<=4,'gaps of 2-4 s');
  if(t>=hold.burstStart+1/60&&t<hold.burstEnd){assert.ok(out.pose.kick>0,`no kick inside a burst at ${t.toFixed(2)}`);inside++;}
  if(t>=hold.burstEnd+.3+1e-6&&t<hold.nextBurst){assert.equal(out.pose.kick,0,`kicking in a gap at ${t.toFixed(2)}`);gap++;}
  if(out.pose.kick===1)burst++;
  assert.equal(out.face.intent,'pout');assert.ok(Math.abs(out.breath.rate-1/1.4)<1e-9);
 }
 assert.ok(inside>60&&gap>60&&burst>0,`${inside} burst and ${gap} gap frames`);
 run(.25,{vx:100});assert.equal(hold.state,'dangling','a slow 25 px move goes back to dangling and restarts the timer');
 run(5.8);assert.equal(hold.state,'dangling');run(.3);assert.equal(hold.state,'protesting');
 run(11.8);assert.equal(hold.state,'protesting');run(.2);assert.equal(hold.state,'resigned');
 let out=run(1/60);assert.equal(out.face.intent,'sigh');assert.equal(out.pose.limp,1);assert.equal(out.look,'floor');
 const sighing=run(2).breath.depth;out=run(2);assert.equal(out.face.intent,'resigned');assert.ok(sighing>2.5&&out.breath.depth<sighing,`one deep sigh (${sighing.toFixed(2)}) and then ${out.breath.depth.toFixed(2)}`);
 run(.25,{vx:100});assert.equal(hold.state,'carried');assert.equal(run(1/60).face.intent,'perk');run(.5);assert.equal(hold.state,'carried');run(.1);assert.equal(hold.state,'dangling','0.6 s after slowing');
});

test('a gentle release sets her down at once: legs reach, a 0.4 squash by 0.26 s, a small undershoot, gone by 1.2 s; recovery ends at recoverySeconds',()=>{
 const {hold,run,now,events}=scene();hold.grab(now(),{kind:'body'});run(1);
 assert.equal(hold.release(now(),{fall:false}),'landing');let out=run(1/60);
 assert.equal(out.state,'landing');assert.deepEqual(out.blinks,[{kind:'relief',delay:.05}]);assert.equal(out.face.intent,'relief');assert.equal(out.transition,.35);assert.equal(out.look,null);
 const squash=[out.pose.squash],states=[out.state],transitions=[out.transition];for(let i=0;i<70;i++){out=run(1/60);squash.push(out.pose.squash);states.push(out.state);transitions.push(out.transition);}
 assert.ok(squash[4]<.23,'the legs reach first (about 10-degree knees)');assert.ok(Math.abs(squash[15]-.4)<.01,`the squash is 0.4 by 0.26 s (${squash[15].toFixed(3)})`);
 assert.equal(states[14],'landing');assert.equal(states[16],'recovering');assert.ok(transitions.every(t=>t===.35),'recovering keeps the landing crossfade');
 assert.ok(Math.abs(Math.max(...squash)-.4)<.01);assert.ok(Math.min(...squash)<0&&Math.min(...squash)>=-.15,`undershoot ${Math.min(...squash).toFixed(3)}`);assert.ok(Math.abs(squash.at(-1))<.02);
 out=run(2.5-71/60-1/60-.02);assert.equal(out.state,'recovering');out=run(.04);assert.equal(out.state,null);assert.deepEqual(out.events.map(e=>[e.from,e.to]),[['recovering',null]]);
 assert.equal(run(1/60),null,'nothing to show afterwards');
 assert.deepEqual(events.map(e=>e.to),['grabbed','dangling','landing','recovering',null],'each transition reports exactly one event');
});

test('thrown she falls (eyes wide, looking at the floor, no blinking, the falling pose); a landing squashes by the impact, overshoots at most 0.15 and settles by 1.2 s',()=>{
 for(const impact of [1000,2500,4000]){
  const {hold,run,now,events}=scene();hold.grab(now(),{kind:'body'});run(1);
  assert.equal(hold.release(now(),{fall:true,thrown:false}),'falling');let out=run(1/60);
  assert.equal(out.state,'falling');assert.equal(out.face.intent,'falling');assert.equal(out.look,'floor');assert.equal(out.pose.fall,1);assert.ok(out.suppress>0,'no spontaneous blinks on the way down');assert.equal(out.transition,.2);
  out=run(.5);assert.equal(out.state,'falling','she falls until the renderer reports the contact');
  hold.land(now(),impact);const goal=Math.min(1.2,Math.max(.5,impact/2500)),squash=[];let first=null;
  for(let i=0;i<72;i++){out=run(1/60);first??=out;squash.push(out.pose.squash);}
  assert.equal(first.state,'landing');assert.equal(first.transition,.08,'into the landing in 0.08 s');assert.equal(first.look,'floor');
  assert.ok(Math.abs(squash[15]-goal)<.01,`impact ${impact}: squash ${squash[15].toFixed(3)} by 0.26 s, ${goal} expected`);
  assert.ok(Math.min(...squash)<0&&Math.min(...squash)>=-.15,`overshoot ${Math.min(...squash).toFixed(3)}`);assert.ok(Math.abs(squash.at(-1))<.02,`${squash.at(-1).toFixed(3)} at 1.2 s`);
  assert.deepEqual(events.map(e=>e.to).slice(-3),['falling','landing','recovering']);
 }
 const scaled=scene();scaled.hold.grab(scaled.now(),{kind:'body'});scaled.run(1);scaled.hold.release(scaled.now(),{fall:true});scaled.run(.3);scaled.hold.land(scaled.now(),2500);
 assert.ok(Math.abs(scaled.run(.27,{},{...SETTINGS,landingSquash:.5}).pose.squash-.5)<.02,'landingSquash scales it');
 const none=scene();none.hold.grab(none.now(),{kind:'body'});none.run(1);none.hold.release(none.now());assert.equal(none.run(.3,{},{...SETTINGS,landingSquash:0}).pose.squash,0,'landingSquash 0: no squash');
});

test('after a landing: thrown, hard or after shaking her eyes shut and she pouts at the cursor; a soft fall gets a relief blink, a sigh and a look back at the viewer',()=>{
 const land=({thrown=false,impact=1500,setup=s=>s.run(1)}={})=>{const s=scene();s.hold.grab(s.now(),{kind:'body'});setup(s);s.hold.release(s.now(),{fall:true,thrown});s.run(.4);s.hold.land(s.now(),impact);return s;};
 const timeline=s=>{const faces=[],looks=[];for(let i=0;i<60*2.4;i++){const out=s.run(1/60);faces.push(out?.face.intent??null);looks.push(out?.look??null);}const at=t=>Math.round(t*60)-1;return {face:t=>faces[at(t)],look:t=>looks[at(t)]};};
 let s=land({thrown:true,impact:3000});let first=s.run(1/60);assert.deepEqual(first.blinks,[],'no relief blink: the eyes squeeze shut');
 let t=timeline(s);assert.equal(t.face(.1),'squeeze');assert.equal(t.face(.5),'pout');assert.equal(t.face(1.8),'pout');assert.equal(t.face(2),null);
 assert.equal(t.look(.15),'floor');assert.equal(t.look(.5),'cursor');assert.equal(t.look(1.6),null);
 s=land({thrown:false,impact:1200});first=s.run(1/60);assert.deepEqual(first.blinks,[{kind:'relief',delay:.05}]);assert.equal(first.face.intent,'relief');
 t=timeline(s);assert.equal(t.look(.15),'floor');assert.equal(t.look(.5),'viewer');assert.equal(t.look(1.2),null);assert.equal(t.face(1.7),'relief');assert.equal(t.face(1.9),null);
 const sighs=land({impact:1200});sighs.run(.3);assert.ok(sighs.run(1.5).breath.rate<.34,'a slow sigh after a soft landing');
 assert.equal(timeline(land({impact:2600})).face(.5),'pout','a hard landing pouts even if she was dropped');
 s=land({thrown:true,impact:1200});first=s.run(1/60);assert.deepEqual(first.blinks,[],'a soft throw shuts her eyes too');assert.equal(first.face.intent,'squeeze');
 s=land({impact:2100});first=s.run(1/60);assert.equal(first.face.intent,'squeeze','a big squash (0.84) does, dropped or not');assert.equal(timeline(s).face(.5),'relief');
 assert.equal(timeline(land({impact:1200,setup:x=>x.run(6.2)})).face(.5),'pout','so does a fall after a protest');
});

test('a wall bonk winces and blinks only in the air; a catch is a 1.2x surprise; land() outside a fall and reactions off do nothing',()=>{
 const s=scene();s.hold.grab(s.now(),{kind:'body'});s.run(1);s.hold.release(s.now(),{fall:true});s.run(.1);
 s.hold.bonk(s.now());let out=s.run(1/60);assert.equal(out.face.intent,'wince');assert.deepEqual(out.blinks,[{kind:'startle',delay:0}]);s.hold.bonk(s.now());assert.deepEqual(s.run(1/60).blinks,[],'a second wall 0.02 s later: no second blink');assert.equal(s.run(.4).face.intent,'falling');
 s.hold.press(s.now());s.hold.grab(s.now(),{kind:'body',caught:true});out=s.run(1/60);assert.equal(out.state,'grabbed');assert.equal(out.face.intensity,1.2);assert.equal(s.run(1).face.intensity,1,'only for the surprise');
 s.hold.bonk(s.now());assert.equal(s.run(1/60).face.intent,'worried','no bonk while held');
 const idle=scene();idle.hold.land(idle.now(),3000);assert.equal(idle.run(1/60),null);
 const off=scene();off.hold.grab(off.now(),{kind:'body'});off.run(.5);off.hold.release(off.now(),{fall:true});assert.equal(off.run(.1,{},{...SETTINGS,enabled:false}),null);off.hold.land(off.now(),3000);assert.equal(off.hold.state,null);
});

test('a set-down that glides onto the floor first stays held as it was, with no tracking, until land() at the glide\'s end',()=>{
 const s=scene();s.hold.grab(s.now(),{kind:'body'});s.run(1);
 assert.equal(s.hold.release(s.now(),{defer:true}),'dangling');let out=s.run(.3,{vx:2000,vy:-1800});
 assert.equal(out.state,'dangling','a glide at 2000 px/s is not a fast carry');assert.equal(out.pose.squash,0);assert.equal(out.face.intent,'worried');
 assert.equal(s.hold.land(s.now(),null),'landing');out=s.run(1/60);assert.equal(out.state,'landing');assert.deepEqual(out.blinks,[{kind:'relief',delay:.05}]);
 assert.ok(Math.abs(s.run(.25).pose.squash-.4)<.01,'the 0.4 squash builds from the touchdown');assert.equal(s.hold.released,false);
 const caught=scene();caught.hold.grab(caught.now(),{kind:'body'});caught.run(1);caught.hold.release(caught.now(),{defer:true});caught.hold.press(caught.now());caught.hold.grab(caught.now(),{kind:'body'});
 assert.equal(caught.hold.released,false,'caught mid-glide she is held again');caught.run(.5,{vx:1500});assert.equal(caught.hold.state,'carried');
 const idle=scene();assert.equal(idle.hold.release(idle.now(),{defer:true}),null);assert.equal(idle.hold.land(idle.now(),null),null,'nothing held, nothing lands');
});

test('after a protest or dizziness the release is annoyed and looks at the cursor; a calm one is relief',()=>{
 for(const [setup,face,look] of [[s=>s.run(6.2),'annoyed','cursor'],[s=>s.run(1),'relief',null]]){
  const s=scene();s.hold.grab(s.now(),{kind:'body'});setup(s);assert.equal(s.hold.release(s.now()),'landing');const out=s.run(1/60);
  assert.equal(out.face.intent,face);assert.equal(out.look,look);assert.equal(s.run(1.9).face.intent,null,'then her own face');
 }
});

test('mirror eases toward the drag direction only past 200 px/s, over 0.4 s and never in a jump; the breath phase never jumps; the variant follows her mood',()=>{
 const {hold,run,now}=scene();hold.grab(now(),{kind:'body',emotion:{primary:'joy'}});assert.equal(hold.variant,'swing');
 const mirror=vx=>run(1/60,{vx}).pose.mirror;
 assert.equal(mirror(150),1);assert.equal(mirror(-150),1);assert.equal(hold.mirrorTarget,1,'nothing under 200 px/s flips it');
 const flip=[mirror(-250)];for(let i=0;i<30;i++)flip.push(mirror(-80));
 assert.equal(hold.mirrorTarget,-1,'latched past 200 px/s');assert.ok(flip[0]>.95,`the first frame barely moves (${flip[0].toFixed(3)})`);
 assert.ok(flip.every((m,i)=>!i||m<=flip[i-1]&&flip[i-1]-m<=1.5*2/24+1e-9),'monotonic, at most 1.5 times the average speed');
 assert.ok(flip[22]>-1&&flip[23]===-1,`fully mirrored after 0.4 s (${flip.slice(20,25).map(m=>m.toFixed(3))})`);
 const back=[mirror(250)];for(let i=0;i<10;i++)back.push(mirror(0));assert.ok(back.every((m,i)=>!i||m>=back[i-1]),'a reversal mid-way turns around without a jump');
 let phase=run(1/60).breath.phase;
 for(let i=0;i<240;i++){const out=run(1/60,{vx:i<120?0:1000}),jump=((out.breath.phase-phase)%(2*Math.PI)+2*Math.PI)%(2*Math.PI);assert.ok(jump<=2*Math.PI*out.breath.rate/60+1e-9,`breath jumped ${jump}`);phase=out.breath.phase;}
 const counts={tuck:0,reach:0};for(let seed=1;seed<=200;seed++){const h=new HoldBehaviour({random:seeded(seed)});h.grab(0,{kind:'body',emotion:{primary:'sadness'}});counts[h.variant]++;}
 assert.ok(counts.tuck>120&&counts.reach>40&&counts.tuck+counts.reach===200,JSON.stringify(counts));
});

test('protest always lasts a while: with protestSeconds above resignSeconds she still protests 3 s before resigning',()=>{
 const {hold,run,now,events}=scene();hold.grab(now(),{kind:'body'});const settings={...SETTINGS,protestSeconds:30,resignSeconds:10};
 run(29.9,{},settings);assert.equal(hold.state,'dangling');run(.2,{},settings);assert.equal(hold.state,'protesting');
 run(2.8,{},settings);assert.equal(hold.state,'protesting');run(.2,{},settings);assert.equal(hold.state,'resigned');
 assert.deepEqual(events.map(e=>e.to),['grabbed','dangling','protesting','resigned']);
});

test('a spring grab is a tug (face only), a click pokes for 0.6 s, reactions off return null, and ragdoll makes her limp',()=>{
 const tug=scene();tug.hold.press(tug.now());tug.hold.grab(tug.now(),{kind:'spring'});let out=tug.run(1/60);
 assert.equal(out.state,'tugged');assert.equal(out.pose,null);assert.equal(out.face.intent,'wince');assert.equal(out.look,'cursor');
 tug.hold.release(tug.now());out=tug.run(1/60);assert.equal(out.state,null);assert.equal(out.face.intent,null);assert.equal(tug.run(1/60),null);
 const poke=scene();poke.hold.press(poke.now());poke.run(.1);poke.hold.click(poke.now());assert.equal(poke.run(.5).face.intent,'poke');assert.equal(poke.run(.12),null);assert.equal(poke.hold.state,null);assert.equal(poke.events.at(-1).from,'poked');
 const off=scene();off.hold.grab(off.now(),{kind:'body'});assert.equal(off.run(.1,{},{...SETTINGS,enabled:false}),null);assert.equal(off.hold.state,null,'switching reactions off clears the hold');
 const rag=scene();rag.hold.grab(rag.now(),{kind:'body'});rag.run(.5);out=rag.run(1/60,{ragdoll:true});assert.equal(out.state,'dizzy');assert.equal(out.pose.limp,1);
 rag.run(3,{ragdoll:true});assert.equal(rag.hold.state,'dizzy','limp while the ragdoll lasts');rag.run(1.1);assert.equal(rag.hold.state,'dizzy');rag.run(.15);assert.equal(rag.hold.state,'dangling');
 const passive=passiveHold(true);assert.equal(passive.pose.limp,1);assert.equal(passiveHold().pose.limp,0);assert.ok(HELD_STATES.has(passiveHold().state));
});

test('HoldFace reacts only on a change of intent and forwards blinks and suppression; with a real face the grab reads startle, surprise, worried',()=>{
 const calls=[],fake={react:(intent,options)=>calls.push(['react',intent,options.intensity]),blink:(kind,delay)=>calls.push(['blink',kind,delay]),blinker:{suppress:s=>calls.push(['suppress',s])}};
 const driver=new HoldFace(fake),{hold,run,now}=scene();hold.press(now());driver.apply(run(1/60));driver.apply(run(1/60));
 hold.grab(now(),{kind:'body'});for(let i=0;i<60;i++)driver.apply(run(1/60));hold.release(now());driver.apply(run(1/60));
 assert.deepEqual(calls.filter(c=>c[0]==='react').map(c=>c[1]),['poke','startle','surprise','worried','relief']);
 assert.deepEqual(calls.filter(c=>c[0]!=='react'),[['blink','press',.03],['suppress',.9],['blink','relief',.05]]);
 driver.apply(null);assert.deepEqual(calls.at(-1),['react',null,1],'nothing to show clears the reaction once');new HoldFace(null).apply(run(1/60));
 const names=['neutral','aa','ih','ou','ee','oh','blink','blinkLeft','blinkRight','angry','relaxed','happy','sad','Surprised',...Object.keys(FACE_PARTS)],manager=new VRMExpressionManager();for(const name of names)manager.registerExpression(new VRMExpression(name));
 const face=new AvatarFace(manager,{random:()=>.999}),real=new HoldFace(face),s=scene();let out;
 const step=seconds=>{for(let i=0,n=Math.round(seconds*60);i<n;i++){const held=s.run(1/60);real.apply(held);out=face.step(1/60,{breathPhase:held?.breath.phase,settings:{}});}return out;};
 s.hold.press(s.now());step(.08);assert.ok(out.get('blink')>0&&out.get('browUp')>.1,'press: blink and brow pop');
 s.hold.grab(s.now(),{kind:'body'});step(.1);assert.ok(out.get('browAngry')>.15,'startle');
 step(.4);assert.ok(out.get('browUp')>.9&&out.get('eyeWide')>.6&&out.get('mouthSurprised')>.4,'surprise');
 step(2);assert.ok(out.get('browWorry')>.45&&out.get('browUp')<.05,'worried while dangling');
 s.hold.release(s.now());step(.6);assert.ok(out.get('relaxed')>.35,'relief on release');step(3);assert.ok(Math.abs(out.get('relaxed')-.15)<.02,'and the resting smile is back');
 assert.ok(INTENTS.perk&&INTENTS.sigh&&INTENTS.relief.oh>0);
});
