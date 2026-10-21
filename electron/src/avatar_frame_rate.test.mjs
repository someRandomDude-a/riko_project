import test from 'node:test';
import assert from 'node:assert/strict';
import {frameInterval,avatarActive,motionBusy,frameLoop,IDLE_FPS,SLEEP_SETTLE_MS,POINTER_ACTIVE_MS,MOTION_TAIL_MS} from './avatar_frame_rate.mjs';

test('adaptive pacing: about 30 fps standing still, every display frame while she moves, nothing hidden or once settled asleep',()=>{
 assert.equal(frameInterval({}),1000/IDLE_FPS);
 assert.equal(frameInterval({active:true}),0);
 assert.equal(frameInterval({hidden:true,active:true}),Infinity,'hidden draws nothing, even moving');
 assert.equal(frameInterval({asleep:true,quietFor:SLEEP_SETTLE_MS-1}),1000/IDLE_FPS,'asleep, she settles first');
 assert.equal(frameInterval({asleep:true,quietFor:SLEEP_SETTLE_MS}),Infinity);
 assert.equal(frameInterval({asleep:true,quietFor:1e9,settled:false}),1000/IDLE_FPS,'asleep, a blink or a face fade finishes first');
 assert.equal(frameInterval({asleep:true,active:true,quietFor:1e9}),0,'picked up while asleep, she moves at full rate');
 assert.equal(frameInterval({mode:'full'}),0,'always full');assert.equal(frameInterval({mode:'full',asleep:true,quietFor:1e9}),0);
 assert.equal(frameInterval({mode:'full',hidden:true}),Infinity);
});

test('she counts as moving while held, thrown, walking, animating or reacting, for 1.5 s after the pointer moves over her and for 1 s after she moved',()=>{
 assert.equal(avatarActive({now:10}),false);
 for(const key of ['dragging','released','walking','animating','reacting'])assert.equal(avatarActive({[key]:true,now:10}),true,key);
 assert.equal(avatarActive({pointerAt:1000,now:1000+POINTER_ACTIVE_MS-1}),true);
 assert.equal(avatarActive({pointerAt:1000,now:1000+POINTER_ACTIVE_MS}),false);
 assert.equal(avatarActive({movedAt:1000,now:1000+MOTION_TAIL_MS-1}),true,'her springs settle at full rate');
 assert.equal(avatarActive({movedAt:1000,now:1000+MOTION_TAIL_MS}),false);
});

test('motion is busy for moving modes, crossfades, one-shot actions and clips that do not loop, not for a looping idle clip',()=>{
 const base=asset=>({kind:'motion.base',status:'running',payload:{procedural:'idle',asset}});
 assert.equal(motionBusy([base({loop:true})],'idle',false),false);
 assert.equal(motionBusy([base({loop:false})],'idle',false),true);
 assert.equal(motionBusy([base(null)],'idle',true),true,'a crossfade');
 for(const mode of ['speaking','walking','hold','held','clicked','settling','recovering'])assert.equal(motionBusy([],mode),true,mode);
 for(const mode of ['idle','listening','thinking','tool','sleeping'])assert.equal(motionBusy([],mode),false,mode);
 for(const kind of ['gesture','motion.preview','wake_animation']){assert.equal(motionBusy([{kind,status:'running',payload:{}}]),true,kind);assert.equal(motionBusy([{kind,status:'completed',payload:{}}]),false,kind);}
});

function fakes(){
 const f={frames:new Map(),timers:new Map(),id:0,time:0};
 f.options={request:cb=>{const id=++f.id;f.frames.set(id,cb);return id;},cancel:id=>f.frames.delete(id),schedule:(cb,ms)=>{const id=++f.id;f.timers.set(id,{cb,ms,at:f.time+ms});return id;},unschedule:id=>f.timers.delete(id),now:()=>f.time,lead:16,slack:4};
 // a display frame every `period` ms: the pending animation-frame callbacks run at the next vsync
 f.vsync=(period=1000/120)=>{f.time=Math.ceil((f.time+1e-6)/period)*period;const pending=[...f.frames.values()];f.frames.clear();for(const cb of pending)cb();};
 f.fire=()=>{const pending=[...f.timers.values()];f.timers.clear();for(const {cb,at} of pending){f.time=Math.max(f.time,at);cb();}};
 return f;
}

test('the loop asks for display frames only while she moves, sleeps on a timer between idle frames, and pauses on Infinity',()=>{
 const f=fakes(),waits=[0,1000/30,Infinity],times=[];let steps=0;
 const loop=frameLoop(()=>{times.push(f.time);return waits[steps++];},f.options);loop.start();
 assert.equal(f.frames.size,1);f.vsync();assert.equal(steps,1);assert.equal(f.frames.size,1,'moving: the next display frame');
 f.vsync();assert.equal(steps,2);assert.equal(f.frames.size,0);assert.deepEqual([...f.timers.values()].map(t=>Math.round(t.ms*10)/10),[17.3],'idle: a timer ending a frame early, no display frames meanwhile');
 f.fire();assert.equal(f.frames.size,1,'then display frames');
 let early=0;while(steps===2){f.vsync();early++;}
 assert.ok(early>=1&&early<=3,'a few display frames until it is due');
 assert.ok(Math.abs(times[2]-times[1]-1000/30)<=4.2,'the idle frame lands on the vsync it is due on ('+(times[2]-times[1]).toFixed(1)+' ms)');
 assert.equal(f.frames.size+f.timers.size,0);assert.equal(loop.paused,true,'paused');
});

test('idle frames keep about 30 fps on 60 and 120 Hz displays',()=>{
 for(const hz of [60,120]){
  const f=fakes(),times=[];const loop=frameLoop(()=>{times.push(f.time);return 1000/30;},f.options);loop.start();
  for(let i=0;i<400&&times.length<31;i++){if(f.frames.size)f.vsync(1000/hz);else f.fire();}
  const fps=1000*(times.length-1)/(times.at(-1)-times[0]);assert.ok(fps>28&&fps<=30.5,hz+' Hz: '+fps.toFixed(1)+' fps');loop.stop();
 }
});

test('wake brings a waiting or paused frame to the next display frame once, and stop cancels everything',()=>{
 const f=fakes();let steps=0,wait=1000/30;
 const loop=frameLoop(()=>{steps++;return wait;},f.options);loop.start();f.vsync();
 assert.equal(f.timers.size,1);loop.wake();assert.equal(f.timers.size,0,'the timer is cancelled');assert.equal(f.frames.size,1);
 loop.wake();assert.equal(f.frames.size,1,'one display frame however often it is woken');
 wait=Infinity;f.vsync();assert.equal(steps,2,'woken, it draws at once');assert.equal(loop.paused,true);loop.wake();assert.equal(f.frames.size,1,'a paused loop wakes');
 wait=0;f.vsync();assert.equal(f.frames.size,1);loop.stop();assert.equal(f.frames.size,0);loop.wake();assert.equal(f.frames.size+f.timers.size,0,'stopped stays stopped');
 assert.equal(steps,3);
});

test('a step that throws still schedules the next frame, so one bad frame never stops the loop',()=>{
 const f=fakes();let steps=0;
 const loop=frameLoop(()=>{steps++;if(steps===2)throw new Error('bad frame');return 0;},f.options);loop.start();
 f.vsync();assert.throws(()=>f.vsync(),/bad frame/);assert.equal(steps,2);
 assert.equal(loop.paused,false,'not paused');assert.equal(f.timers.size,1,'the next frame waits at the idle rate');
 f.fire();for(let i=0;i<4&&steps===2;i++)f.vsync();assert.equal(steps,3,'and runs');assert.equal(f.frames.size,1,'the loop goes on');loop.stop();
});
