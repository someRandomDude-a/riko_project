import test from 'node:test';
import assert from 'node:assert/strict';
import {Group,Mesh,BufferGeometry,Float32BufferAttribute} from 'three';
import {VRMExpression,VRMExpressionManager,VRMExpressionMorphTargetBind} from '@pixiv/three-vrm';
import {resolveExpression,registerFaceParts,measureLidClosure,reportedExpressions,FaceMixer,Blinker,AvatarFace,FACE_PARTS,INTENTS,MAX_EXPRESSIONS,GAZE_BLINK_GAP} from './avatar_face.mjs';

const seeded=(seed=7)=>()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};
const never=()=>.999;  // the longest blink interval, so no spontaneous blink lands inside a short test
const manager=(...names)=>{const m=new VRMExpressionManager();for(const name of names)m.registerExpression(new VRMExpression(name));return m;};
const VROID=['neutral','aa','ih','ou','ee','oh','blink','blinkLeft','blinkRight','angry','relaxed','happy','sad','Surprised'];
const run=(mixer,seconds,options={},dt=1/60)=>{let out=new Map();for(let t=0;t<seconds-1e-9;t+=dt)out=mixer.step(dt,options);return out;};

test('resolveExpression finds Surprised for surprised, prefers an exact match and returns null for an unknown name',()=>{
 const m=manager('happy','Happy','Surprised','Joy');
 assert.equal(resolveExpression(m,'surprised'),'Surprised');assert.equal(resolveExpression(m,'happy'),'happy');assert.equal(resolveExpression(m,'HAPPY'),'happy');
 assert.equal(resolveExpression(m,'unknown'),null);assert.equal(resolveExpression(m,''),null);assert.equal(resolveExpression(null,'happy'),null);
 m.registerExpression(new VRMExpression('surprised'));assert.equal(resolveExpression(m,'surprised'),'surprised');  // the memo follows new names
 assert.equal(resolveExpression(manager('Joy','Sorrow'),'happy'),'Joy');assert.equal(resolveExpression(manager('Sorrow'),'sad'),'Sorrow');
 assert.equal(resolveExpression(m,'constructor'),null);assert.equal(resolveExpression(m,'toString'),null);  // never a prototype member
});

function vroidRig(existing=[]){
 const scene=new Group(),meshes=[new Mesh(),new Mesh()];
 meshes[0].morphTargetDictionary={Fcl_BRW_Surprised:0,Fcl_EYE_Joy:1,Fcl_MTH_Small:2};meshes[0].morphTargetInfluences=[0,0,0];
 meshes[1].morphTargetDictionary={Fcl_MTH_Small:0,Fcl_BRW_Surprised:3};meshes[1].morphTargetInfluences=[0,0,0,0];  // another index on this primitive
 scene.add(...meshes);const expressionManager=manager(...existing);return {vrm:{scene,expressionManager},meshes};
}

test('registerFaceParts registers only parts whose morph exists, never replaces a name, and drives every primitive',()=>{
 const {vrm,meshes}=vroidRig(['happy','relaxed','mouthSmall']),kept=vrm.expressionManager.getExpression('mouthSmall');
 assert.deepEqual(registerFaceParts(vrm),['browUp','eyeSmile']);
 assert.equal(vrm.expressionManager.getExpression('mouthSmall'),kept);assert.equal(vrm.expressionManager.getExpression('eyeWide'),null);
 assert.equal(vrm.expressionManager.getExpression('eyeSmile').overrideBlink,'blend');assert.equal(vrm.expressionManager.getExpression('happy').overrideBlink,'blend');assert.equal(vrm.expressionManager.getExpression('relaxed').overrideBlink,'none');
 assert.ok(vrm.scene.children.includes(vrm.expressionManager.getExpression('browUp')));
 vrm.expressionManager.setValue('browUp',.7);vrm.expressionManager.update();
 assert.equal(meshes[0].morphTargetInfluences[0],.7);assert.equal(meshes[1].morphTargetInfluences[3],.7);assert.equal(meshes[1].morphTargetInfluences[0],0);
 assert.deepEqual(registerFaceParts(vrm),[]);  // a second load adds nothing
});

test('registerFaceParts stops at the backend limit of 64 names and leaves models without Fcl morphs alone',()=>{
 const {vrm}=vroidRig(Array.from({length:MAX_EXPRESSIONS-1},(_,i)=>'e'+i));
 assert.equal(registerFaceParts(vrm).length,1);assert.equal(Object.keys(vrm.expressionManager.expressionMap).length,MAX_EXPRESSIONS);
 const plain={scene:new Group(),expressionManager:manager('happy')};plain.scene.add(new Mesh());
 assert.deepEqual(registerFaceParts(plain),[]);assert.equal(plain.expressionManager.getExpression('happy').overrideBlink,'none');
 assert.equal(Object.keys(FACE_PARTS).every(name=>name.length<=64),true);
});

test('reportedExpressions leaves out names the backend refuses (over 64 characters) before taking 64 of them',()=>{
 const long='x'.repeat(65),names=[long,'happy','\u{1F600}'.repeat(64),...Array.from({length:70},(_,i)=>'e'+i)],reported=reportedExpressions(manager(...names));
 assert.equal(reported.length,MAX_EXPRESSIONS);assert.equal(reported.includes(long),false);assert.equal(reported[0],'happy');assert.equal(reported[1],'\u{1F600}'.repeat(64),'64 code points pass');
 assert.deepEqual(reportedExpressions(null),[]);
});

test('a surprise rises within 0.15 s and lingers after it ends; at full weight the base preset goes to 0',()=>{
 const mixer=new FaceMixer(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never}),joy={primary:'joy',intensity:.8};
 run(mixer,1,{emotion:joy});assert.ok(Math.abs(mixer.values.get('happy')-.8)<.01);
 let out=run(mixer,.15,{emotion:joy,intent:INTENTS.surprise});assert.ok(out.get('browUp')>=.9,`browUp ${out.get('browUp')}`);
 out=run(mixer,.85,{emotion:joy,intent:INTENTS.surprise});assert.ok(out.get('happy')<.01,`happy ${out.get('happy')}`);
 let lingering=0;for(let i=0;i<120;i++){out=mixer.step(1/60,{emotion:joy});if(out.get('browUp')>.1)lingering+=1/60;}
 assert.ok(lingering>=.4,`browUp lingered ${lingering.toFixed(2)} s`);assert.ok(Math.abs(mixer.values.get('happy')-.8)<.05,'the base face came back');
});

test('the resting smile shows only with no emotion or a neutral one, and intensity scales reactions',()=>{
 const m=manager(...VROID,...Object.keys(FACE_PARTS));
 assert.ok(Math.abs(run(new FaceMixer(m,{random:never}),2,{emotion:null,restingSmile:.2}).get('relaxed')-.2)<.01);
 assert.ok(Math.abs(run(new FaceMixer(m,{random:never}),2,{emotion:{primary:'neutral',intensity:.5},restingSmile:.2}).get('relaxed')-.2)<.01);
 assert.ok(!(run(new FaceMixer(m,{random:never}),2,{emotion:{primary:'sadness',intensity:.6},restingSmile:.2}).get('relaxed')>0));
 assert.ok(Math.abs(run(new FaceMixer(m,{random:never}),2,{intent:INTENTS.surprise,intensity:.5}).get('browUp')-.5)<.01);
 assert.ok(Math.abs(run(new FaceMixer(m,{random:never}),2,{intent:INTENTS.surprise,intensity:0,restingSmile:.15}).get('relaxed')-.15)<.01,'intensity 0 keeps the base face');
});

test('a model without the parts falls back to its presets; a hit rule replaces the intent at its own intensity',()=>{
 const mixer=new FaceMixer(manager(...VROID,'browUp'),{random:never});  // no eyeWide or mouthSurprised
 let out=run(mixer,1,{intent:INTENTS.surprise});assert.ok(out.get('Surprised')>.95);assert.ok(!(out.get('browUp')>0));
 const ruled=new FaceMixer(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never});
 out=run(ruled,1,{intent:INTENTS.worried,rule:{expression:'surprised',intensity:.5}});
 assert.ok(Math.abs(out.get('Surprised')-.5)<.01);assert.ok(!(out.get('browWorry')>0));assert.ok(!(out.get('relaxed')>.01),'the rule takes the whole face');
 out=run(ruled,1,{intent:INTENTS.worried,rule:{expression:'not-on-this-model',intensity:.5}});assert.ok(out.get('browWorry')>.5,'an unknown rule expression leaves the intent');
});

test('the eyelid is max(blink, intent lid) on blink only, and a blink closes over wide eyes',()=>{
 const mixer=new FaceMixer(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never});
 let out=run(mixer,.3,{intent:INTENTS.startle});assert.ok(Math.abs(out.get('blink')-.8)<.01);
 out=run(mixer,1,{intent:INTENTS.surprise});const wide=out.get('eyeWide');assert.ok(wide>.6);
 mixer.blinker.trigger('startle');out=run(mixer,.09,{intent:INTENTS.surprise});
 assert.ok(out.get('blink')>.95);assert.ok(out.get('eyeWide')<.05*wide,`eyeWide ${out.get('eyeWide')} under a closed lid`);
 assert.ok(mixer.values.get('eyeWide')>.6,'the blink does not change the stored surprise');
 out=run(mixer,.4,{intent:INTENTS.surprise});assert.ok(out.get('blink')<.01);assert.ok(out.get('eyeWide')>.6);
});

test('the worried mouth breathes with oh, and a reaction never pushes a channel past 1',()=>{
 const mixer=new FaceMixer(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never});run(mixer,2,{intent:INTENTS.worried});
 const ohs=[];for(let i=0;i<180;i++){const out=mixer.step(1/60,{intent:INTENTS.worried,emotion:{primary:'surprise',intensity:1}});ohs.push(out.get('oh')||0);for(const v of out.values())assert.ok(v<=1&&v>=0);}
 assert.ok(Math.max(...ohs)>.15&&Math.max(...ohs)<=.2+1e-9,`oh peak ${Math.max(...ohs)}`);assert.ok(Math.min(...ohs)<.03);
});

test('blinks: seeded intervals stay within 2-6 s, close by 0.1 s, open below 0.1 by 0.25 s, sometimes come in pairs',()=>{
 const blinker=new Blinker({random:seeded(3)}),starts=[];let previous=0,start=null,closedAt=null,checks=0;
 for(let i=0;i<60*600;i++){const t=(i+1)/60,lid=blinker.step(1/60);
  if(previous===0&&lid>0){starts.push(t);start=t;closedAt=null;}
  if(start!==null&&lid===1&&closedAt===null){closedAt=t;assert.ok(t-start<=.1+1e-9,`closed after ${t-start}`);}
  if(start!==null&&t-start>=.25-1e-9){assert.ok(lid<.1||t-start>.5,`lid ${lid} at ${(t-start).toFixed(3)} s`);if(t-start<.27)checks++;}
  previous=lid;}
 const gaps=starts.slice(1).map((t,i)=>t-starts[i]),doubles=gaps.filter(g=>g<1);
 assert.ok(checks>50);assert.ok(doubles.length>=3&&doubles.length<starts.length*.25,`${doubles.length} doubles in ${starts.length} blinks`);
 for(const gap of gaps)assert.ok(gap<1?gap>=.45&&gap<=.52:gap>=2+.29&&gap<=6+.32,`gap ${gap.toFixed(3)}`);
});

test('forced blinks start in the same step, suppress blocks only spontaneous ones, and moods change the interval',()=>{
 const blinker=new Blinker({random:()=>0});blinker.trigger('startle');assert.ok(blinker.step(1/60)>0);
 const quiet=new Blinker({random:()=>0});quiet.suppress(10);let blinked=false;for(let i=0;i<60*9.5;i++)blinked||=quiet.step(1/60)>0;assert.equal(blinked,false);
 quiet.trigger('press',.03);assert.equal(quiet.step(1/60),0);let lid=0;for(let i=0;i<4;i++)lid=Math.max(lid,quiet.step(1/60));assert.ok(lid>0,'a forced blink ignores suppress');
 const relief=new Blinker({random:never});relief.trigger('relief');const lids=[];for(let i=0;i<60;i++)lids.push(relief.step(1/60));
 assert.ok(lids.filter(v=>v===1).length>=14,'a relief blink stays closed about 0.25 s');
 const anxious=new Blinker({random:never,mood:'anxious'});let first=null;for(let i=0;i<60*8&&first===null;i++)if(anxious.step(1/60)>0)first=(i+1)/60;assert.ok(first<=3.5+1/60);
});

// The reaction steps the hold behaviour (avatar_hold.mjs) produces for a press, a grab, a release and a tug, as AvatarFace sequences.
test('a big gaze shift blinks only when no blink runs or waits, blinks are not suppressed (eyes held wide) and the last gaze blink is 2 s old',()=>{
 const face=new AvatarFace(manager(...VROID),{random:never});face.step(1/60);
 assert.equal(face.gazeShift(),true);assert.equal(face.blinker.busy,true);assert.equal(face.gazeShift(),false,'one is already queued');
 for(let i=0;i<30;i++)face.step(1/60);assert.equal(face.blinker.busy,false);assert.equal(face.gazeShift(),false,'0.5 s after the last gaze blink');
 for(let i=0;i<Math.round((GAZE_BLINK_GAP-.5)*60)+1;i++)face.step(1/60);face.blinker.suppress(.3);assert.equal(face.blinker.quiet,true);assert.equal(face.gazeShift(),false,'falling or surprised: her eyes stay wide');
 for(let i=0;i<20;i++)face.step(1/60);assert.equal(face.blinker.quiet,false);assert.equal(face.gazeShift(),true,'a suppressed shift never used up the 2 s');
 // a cursor sweeping past her face: a big shift every half second for 10 s blinks her at most every 2 s
 const sweep=new AvatarFace(manager(...VROID),{random:never}),at=[];for(let i=0;i<600;i++){sweep.step(1/60);if(i%30===0&&sweep.gazeShift())at.push(i/60);}
 assert.ok(at.length>=4&&at.length<=5&&at.every((t,i)=>!i||t-at[i-1]>=GAZE_BLINK_GAP-1e-9),`gaze blinks at ${at.join(', ')} s`);
});

const STEPS={press:[{intent:'poke',duration:.6,blink:'press',blinkDelay:.03}],grab:[{intent:'startle',duration:.12,suppress:.9},{intent:'surprise',duration:.78},{intent:'worried'}],release:[{intent:'relief',duration:1.5,blink:'relief',blinkDelay:.05}]};
test('AvatarFace sequences: a grab startles, then surprises, then worries while held; release gives relief and the resting smile returns',()=>{
 const face=new AvatarFace(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never});
 const step=seconds=>{let out=new Map();for(let t=0;t<seconds-1e-9;t+=1/60)out=face.step(1/60,{settings:{restingSmile:.15,intensity:1}});return out;};
 step(1);face.sequence(STEPS.press);let out=step(.08);assert.ok(out.get('blink')>0,'press blinks at once');assert.ok(out.get('browUp')>.1,'press pops the brows');
 face.sequence(STEPS.grab);out=step(.1);assert.equal(face.intent,'startle');assert.ok(out.get('browAngry')>.15);
 out=step(.4);assert.equal(face.intent,'surprise');assert.ok(out.get('browUp')>.9&&out.get('eyeWide')>.6&&out.get('mouthSurprised')>.4);
 out=step(2);assert.equal(face.intent,'worried');assert.ok(out.get('browWorry')>.45);assert.ok(out.get('browUp')<.05);
 face.sequence(STEPS.release);out=step(.6);assert.equal(face.intent,'relief');assert.ok(out.get('relaxed')>.35);assert.ok(out.get('browWorry')<.2);
 out=step(3);assert.equal(face.intent,null);assert.ok(Math.abs(out.get('relaxed')-.15)<.02);
 face.react('wince');out=step(.3);assert.ok(out.get('eyeSquint')>.4&&out.get('browAngry')>.3);face.react(null);out=step(2);assert.ok(!(out.get('eyeSquint')>.01));
 face.react('pout',{duration:.5,then:'annoyed'});step(.6);assert.equal(face.intent,'annoyed');face.react(null);assert.equal(face.intent,null);
});

test('AvatarFace keeps asset expression targets in the base face and never writes visemes or blinks from them',()=>{
 const face=new AvatarFace(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never});
 let out;for(let i=0;i<120;i++)out=face.step(1/60,{assets:new Map([['happy',.6],['surprised',.3],['aa',1],['blink',1]])});
 assert.ok(Math.abs(out.get('happy')-.6)<.01);assert.ok(Math.abs(out.get('Surprised')-.3)<.01);assert.ok(!(out.get('aa')>0));assert.ok(out.get('blink')<.01);
});

test('a blink closes fast and opens slowly: half open about 0.05 s after the hold, opening at about half the closing speed',()=>{
 const dt=1/1000,blinker=new Blinker({random:never});blinker.trigger('startle');blinker.step(1e-9);  // start a spontaneous-shaped startle at t=0
 const lids=[];for(let i=0;i<300;i++)lids.push(blinker.step(dt));
 const speeds=lids.slice(1).map((v,i)=>(v-lids[i])/dt),closing=Math.max(...speeds),opening=-Math.min(...speeds);
 const holdEnd=lids.lastIndexOf(1),half=lids.findIndex((v,i)=>i>holdEnd&&v<=.5);
 assert.ok(lids.indexOf(1)*dt<=.075,'closed within the close time');
 assert.ok((half-holdEnd)*dt>=.04&&(half-holdEnd)*dt<=.055,`half open ${((half-holdEnd)*dt).toFixed(3)} s after the hold`);
 assert.ok(opening/closing>.45&&opening/closing<.6,`opening peak ${opening.toFixed(1)}/s against closing ${closing.toFixed(1)}/s`);
});

test('a reaction ending never stacks a whole-face preset on its region part: the base comes back as the reaction leaves',()=>{
 for(const [primary,preset,intent,part] of [['surprise','Surprised','surprise','browUp'],['anger','angry','wince','browAngry'],['sadness','sad','worried','browWorry']]){
  const mixer=new FaceMixer(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never}),emotion={primary,intensity:1};
  run(mixer,2,{emotion});run(mixer,2,{emotion,intent:INTENTS[intent]});
  let peak=0;for(let i=0;i<180;i++){const out=mixer.step(1/60,{emotion});peak=Math.max(peak,(out.get(preset)||0)+(out.get(part)||0));}
  assert.ok(peak<=1.02,`${preset}+${part} peaked at ${peak.toFixed(3)} on release`);assert.ok(Math.abs(mixer.values.get(preset)-1)<.01,`${preset} came back`);
 }
 // a lighter reaction taking over (surprise, then a poke) never goes past where the poke settles
 const mixer=new FaceMixer(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never}),emotion={primary:'surprise',intensity:1};
 run(mixer,2,{emotion});run(mixer,2,{emotion,intent:INTENTS.surprise});let peak=0,out;
 for(let i=0;i<120;i++){out=mixer.step(1/60,{emotion,intent:INTENTS.poke});peak=Math.max(peak,out.get('Surprised')+out.get('browUp'));}
 const settled=out.get('Surprised')+out.get('browUp');assert.ok(peak<=settled+.02,`peak ${peak.toFixed(3)} against settled ${settled.toFixed(3)}`);
});

test('after warm-up a step never copies the expression map, and fresh spec objects are not kept',()=>{
 const m=manager(...VROID,...Object.keys(FACE_PARTS)),getter=Object.getOwnPropertyDescriptor(VRMExpressionManager.prototype,'expressionMap').get;let copies=0;
 Object.defineProperty(m,'expressionMap',{get(){copies++;return getter.call(this);}});
 const face=new AvatarFace(m,{random:never}),options={emotion:{primary:'joy',intensity:.6},assets:new Map([['surprised',.2]]),rule:null,settings:{}};
 for(let i=0;i<10;i++)face.step(1/60,options);face.sequence(STEPS.grab);for(let i=0;i<70;i++)face.step(1/60,options);
 copies=0;for(let i=0;i<120;i++)face.step(1/60,options);face.sequence(STEPS.release);for(let i=0;i<120;i++)face.step(1/60,options);
 assert.equal(copies,0);assert.ok(face.mixer.cache instanceof WeakMap,'ad-hoc specs are cached weakly');
 const mixer=new FaceMixer(m,{random:never});for(let i=0;i<5;i++)assert.ok(mixer.step(1/60,{intent:{...INTENTS.worried}}).get('browWorry')>0);
});

test('AvatarFace passes the body\'s breath phase to the breathing mouth',()=>{
 const ohAt=phase=>{const face=new AvatarFace(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never});face.react('worried');let out;for(let i=0;i<180;i++)out=face.step(1/60,{breathPhase:phase,settings:{}});return out.get('oh')||0;};
 assert.ok(Math.abs(ohAt(Math.PI)-.2)<.01,`oh ${ohAt(Math.PI)} at the top of the breath`);assert.ok(ohAt(0)<.001);
});

// One primitive whose morph 0 closes the eyelid and morph 1 is a sad face that also lowers the lid by 0.3 of a blink.
function lidRig({relaxedBlock=false}={}){
 const geometry=new BufferGeometry();geometry.setAttribute('position',new Float32BufferAttribute(new Array(12).fill(0),3));geometry.morphTargetsRelative=true;
 const close=[0,-.01,0, 0,-.008,0, 0,0,0, 0,0,0],sad=[0,-.003,0, 0,-.0024,0, .004,.002,0, 0,0,.003],fun=[0,-.001,0, 0,-.0008,0, 0,0,0, .002,0,0];
 geometry.morphAttributes.position=[close,sad,fun].map(values=>new Float32BufferAttribute(values,3));
 const mesh=new Mesh(geometry),twin=new Mesh(geometry);mesh.morphTargetInfluences=[0,0,0];twin.morphTargetInfluences=[0,0,0];const m=new VRMExpressionManager();  // two primitives, one vertex space
 for(const [name,index] of [['blink',0],['sad',1],['relaxed',2]]){const e=new VRMExpression(name);e.addBind(new VRMExpressionMorphTargetBind({primitives:[mesh,twin],index,weight:1}));m.registerExpression(e);}
 if(relaxedBlock)m.getExpression('relaxed').overrideBlink='block';
 return {m,mesh};
}
const drive=(face,m,seconds,options)=>{for(let t=0;t<seconds-1e-9;t+=1/60){for(const [name,value] of face.step(1/60,options))m.setValue(name,value);m.update();}};

test('measureLidClosure finds how much an expression already closes the lid, and the blink closes only the rest',()=>{
 const {m,mesh}=lidRig(),closures=measureLidClosure(m);
 assert.ok(Math.abs(closures.get('sad')-.3)<1e-3,`sad closes ${closures.get('sad')}`);assert.ok(Math.abs(closures.get('relaxed')-.1)<1e-3);assert.equal(closures.has('blink'),false);
 assert.equal(measureLidClosure(manager(...VROID)).size,0,'unbound expressions measure nothing');
 const face=new AvatarFace(m,{random:never}),options={emotion:{primary:'sadness',intensity:1},settings:{}};drive(face,m,1,options);
 face.blink('startle');let peak=0,total=0;for(let i=0;i<20;i++){for(const [name,value] of face.step(1/60,options))m.setValue(name,value);m.update();const closed=mesh.morphTargetInfluences[0]+.3*mesh.morphTargetInfluences[1];if(closed>total){total=closed;peak=mesh.morphTargetInfluences[0];}}
 assert.ok(Math.abs(peak-.7)<.02,`blink morph ${peak.toFixed(3)} under a full sad face`);assert.ok(Math.abs(total-1)<.02,`the lid closes to ${total.toFixed(3)} of a blink, not past it`);
 for(let i=0;i<60;i++)face.step(1/60,options);assert.ok(face.step(1/60,options).get('blink')<.001,'with the eyes open the blink stays at 0');
});

test('a relaxed expression authored to block blinks still lets the resting smile blink',()=>{
 const {m,mesh}=lidRig({relaxedBlock:true}),face=new AvatarFace(m,{random:never});
 assert.equal(m.getExpression('relaxed').overrideBlink,'blend');
 const options={settings:{restingSmile:.15}};drive(face,m,1,options);face.blink('startle');
 let peak=0;for(let i=0;i<20;i++){for(const [name,value] of face.step(1/60,options))m.setValue(name,value);m.update();peak=Math.max(peak,mesh.morphTargetInfluences[0]);}
 assert.ok(peak>.8,`the eyelid morph reached ${peak.toFixed(2)}`);
});

test('the face is settled only once every channel reached its goal, no blink runs and no timed reaction waits (the asleep pause)',()=>{
 const face=new AvatarFace(manager(...VROID,...Object.keys(FACE_PARTS)),{random:never});
 const step=(seconds,options={})=>{for(let t=0;t<seconds-1e-9;t+=1/60)face.step(1/60,{settings:{restingSmile:.15,intensity:1},...options});};
 step(2);assert.equal(face.settled,true,'the resting smile reached');
 const sad={emotion:{primary:'sadness',intensity:.8}};step(.1,sad);assert.equal(face.settled,false,'a fade to sadness is running');
 step(3,sad);assert.equal(face.settled,true,'and has finished');
 face.blink('press');assert.equal(face.settled,false,'a queued blink');step(.1,sad);assert.equal(face.settled,false,'a running blink');step(.5,sad);assert.equal(face.settled,true);
 face.react('pout',{duration:.5});assert.equal(face.settled,false,'a timed reaction waits');step(4,sad);assert.equal(face.settled,true,'it ended and faded out');
 face.react('worried');step(4,sad);assert.equal(face.settled,false,'the worried mouth keeps breathing');
});
