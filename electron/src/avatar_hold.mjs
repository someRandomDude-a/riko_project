/** The held behaviour: being picked up is a short scene, not a steady wobble. A press pops her brows and blinks; a grab startles
 * her into surprise; held still she dangles, breathes and glances up at whoever holds her; carried fast she looks alarmed; shaken she
 * gets dizzy; after protestSeconds still she pouts and kicks in bursts, after resignSeconds she sighs and goes limp; a hair tug makes
 * her wince. Let go, she either falls (thrown: arms flung out, eyes wide, looking at the floor; a wall bonks her) and lands with a squash, or
 * is set down in place with a smaller one; then she recovers: a sigh and a look back at the viewer, or a pout at the cursor after a
 * hard throw. Pressing her mid-fall catches her (a bigger surprise). Sources: VPet's raise loop (a move over 20 px restarts the
 * struggle), Shimeji's resist bursts (2.5 Hz with pauses) and Thrown/Fall, Mate Engine's 0.3 s drag lock and breathing mouth,
 * startle-then-emotion timing, squash and follow-through on contact.
 * Stepped once per animation frame with performance.now() in seconds; window velocity (px/s) and acceleration (px/s^2) come from
 * AvatarInertia. The flight itself is avatar_drop.mjs: the renderer reports release({fall}), bonk() and land(impact). Pure: no
 * three.js, no timers. */
export const HELD_STATES=new Set(['grabbed','dangling','carried','dizzy','protesting','resigned']);
/** After the release: the flight, the contact and squash (the first 0.26 s on the ground), and the recovery until recoverySeconds. */
export const DROP_STATES=new Set(['falling','landing','recovering']);
const POSITIVE=new Set(['joy','love','amusement','excitement']);
const T={startle:.12,grab:.35,surprise:.9,holder:.15,poke:.6,carryExit:.6,dizzyExit:1.2,annoyed:3,perk:.4,sigh:3,sighMouth:1,relief:1.8,annoyedLook:1.2,
 reach:.08,squash:.26,squeeze:.15,pout:1.9,viewerLook:1,floorLook:.26,bonk:.35};  // landing times run from the contact
const FAST=700,FAST_ACCEL=9000,SLOW=300,SHAKE=1200,SHAKE_WINDOW=1.5,SHAKES=4,MOVE=20,MIRROR=200,MIRROR_TIME=.4,MIN_PROTEST=3;  // px/s, px/s^2, px; s
/** Landing squash: a set-down is 0.4; a fall is its impact over 2500 px/s, clamped to 0.5-1.2; both times landingSquash. Knees bend for
 * 0.08 s first (the legs reach for the floor: 0.22 is 10 degrees), the squash builds by 0.26 s, then a second-order follower (2.5 Hz,
 * damping 0.7, initial response 1.4: about 11% overshoot into a small stretch) springs her back up. A throw, a hard landing or a big
 * squash (0.8+) shuts her eyes for 0.15 s. */
const SQUASH={setDown:.4,impact:2500,min:.5,max:1.2,reach:.22,big:.8,frequency:2.5,damping:.7,response:1.4};
const HARD=2500;  // px/s: a landing this fast is a pout even when she was not thrown
const BREATH={grabbed:[2.2,0],dangling:[2.2,1.5],carried:[1.6,1.5],protesting:[1.4,1.8],dizzy:[2,1.2],resigned:[2.8,1.5],falling:[1.2,1.5],landing:[2,1.5],recovering:[2,1.5]};  // period (s), chest depth (degrees)
const FACE={dangling:'worried',carried:'alarm',dizzy:'dizzy',protesting:'pout',resigned:'resigned',tugged:'wince',falling:'falling'};
const TAU=2*Math.PI;
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,Number.isFinite(v)?v:0));
const smooth=u=>{u=clamp(u);return u*u*(3-2*u);};
const wrap=phase=>phase%TAU;

export class HoldBehaviour{
 constructor({random=Math.random}={}){this.random=random;this.breathPhase=0;this.breathDepth=0;this.kickPhase=0;this.kickRamp=0;this.cancel();}
 /** Back to no reaction at once, with no events (the model unloads, or reactions were switched off). */
 cancel(){this.state=null;this.since=0;this.transition=.25;this.prior=null;this.variant='tuck';this.mirror=this.mirrorTarget=this.mirrorEase=1;this.kickRamp=0;this.squash=0;this.squashVelocity=0;this.squashGoal=0;this.landAt=0;this.fell=false;this.hard=false;this.squeeze=false;this.caught=false;this.bonkUntil=0;this.released=false;this.pending={events:[],blinks:[],suppress:0};}
 enter(state,now){
  const from=this.state;if(from===state)return;
  // landing to recovering keeps the landing's crossfade (the pose target is the same settle): a new time mid-blend would jump it
  this.transition=state==='landing'?(from==='falling'?.08:.35):state==='recovering'?this.transition:state==='falling'?.2:HELD_STATES.has(state)&&HELD_STATES.has(from)?.4:.25;
  this.state=state;this.since=now;this.pending.events.push({from,to:state,at:now});
  if(state==='protesting'){this.burstStart=now;this.burstEnd=now+.9+.6*this.random();this.nextBurst=this.burstEnd+2+2*this.random();}
  if(state==='resigned')this.sighUntil=now+T.sigh;
  if(state==='dangling'||state==='protesting'){this.glanceUntil=0;this.nextGlance=now+3+3*this.random();}
 }
 /** Pointer down on her body: a brow pop and a blink. */
 press(now){this.pressAt=now;this.enter('pressed',now);this.pending.blinks.push({kind:'press',delay:.03});}
 /** Released before the drag threshold: the brow raise holds 0.6 s (the renderer plays the 'clicked' pose). */
 click(now){this.enter('poked',now);}
 /** The drag threshold was crossed, or she was caught mid-fall (caught: a 1.2x surprise). kind 'spring' (hair or bust) is a tug, face
  * only; 'body' picks her up. */
 grab(now,{region=null,kind='body',emotion=null,caught=false}={}){
  this.grabAt=now;this.region=region;this.kind=kind;this.caught=!!caught;this.released=false;
  if(kind==='spring'){this.enter('tugged',now);return;}
  this.variant=POSITIVE.has(emotion?.primary)?'swing':this.random()<.7?'tuck':'reach';
  this.prior=null;this.mirror=this.mirrorTarget=this.mirrorEase=1;this.position={x:0,y:0};this.still={x:0,y:0};this.stillSince=now;this.reversals=[];this.shakeSign=0;this.lastShake=-Infinity;this.slowSince=null;this.annoyedUntil=0;this.perkUntil=0;
  this.enter('grabbed',now);this.pending.suppress=Math.max(this.pending.suppress,T.surprise);
 }
 /** Pointer up after a drag. fall: the renderer is throwing or dropping her (avatar_drop.mjs) and reports land() at contact; thrown:
  * it was a flick (she pouts after landing). defer: a set-down that glides onto the floor first; she stays as she was held (no more
  * tracking) until land() at the glide's end. Otherwise a held body is set down in place at once (a 0.4 squash). A tug or press just
  * ends. Returns the next state. */
 release(now,{fall=false,thrown=false,defer=false}={}){
  if(this.state==='pressed'||this.state==='tugged')this.enter(null,now);
  if(!HELD_STATES.has(this.state))return this.state;
  this.prior=this.state;this.thrown=!!thrown;
  if(fall){this.enter('falling',now);this.fell=true;return this.state;}
  this.fell=false;if(defer){this.released=true;return this.state;}
  this.land(now,null);
  return this.state;
 }
 /** Her feet touch the floor (a fall), or a set-down (impact null). impact: the downward speed at contact (px/s). Returns the state. */
 land(now,impact=null){
  if(this.state!=='falling'&&!HELD_STATES.has(this.state))return this.state;this.released=false;
  const setDown=impact===null||!this.fell,speed=Math.abs(Number(impact)||0);
  this.squashGoal=setDown?SQUASH.setDown:clamp(speed/SQUASH.impact,SQUASH.min,SQUASH.max);this.squash=0;this.squashVelocity=0;this.squashPrevious=0;
  this.hard=!setDown&&(this.thrown||speed>=HARD||this.prior==='protesting'||this.prior==='dizzy');
  this.landAt=now;this.enter('landing',now);
  this.squeeze=this.hard||this.squashGoal>=SQUASH.big;  // thrown, hard or a big squash: her eyes shut on impact; otherwise a relief blink
  if(!this.squeeze)this.pending.blinks.push({kind:'relief',delay:.05});
  return this.state;
 }
 /** A wall or the top of the screen stopped her mid-flight: a wince and a blink (one blink for a corner: two walls within 0.35 s). */
 bonk(now){if(this.state!=='falling')return;if(now>=this.bonkUntil)this.pending.blinks.push({kind:'startle',delay:0});this.bonkUntil=now+T.bonk;}
 /** Held: tracks moves, speed and shaking, and changes state. */
 track(now,h,{vx=0,vy=0,ax=0,ay=0,ragdoll=false},settings){
  const speed=Math.hypot(vx,vy),fast=speed>FAST||Math.hypot(ax,ay)>FAST_ACCEL;
  this.position.x+=vx*h;this.position.y+=vy*h;
  const moved=Math.hypot(this.position.x-this.still.x,this.position.y-this.still.y)>MOVE;
  if(moved){this.still={...this.position};this.stillSince=now;}
  if(Math.abs(vx)>MIRROR)this.mirrorTarget=Math.sign(vx);  // latched: nothing between -200 and 200 px/s flips it; step() eases toward it
  if(Math.abs(vx)>SHAKE){const sign=Math.sign(vx);if(this.shakeSign&&sign!==this.shakeSign)this.reversals.push(now);if(this.shakeSign&&sign!==this.shakeSign||this.state==='dizzy')this.lastShake=now;this.shakeSign=sign;}  // dizzy lasts while she is still moved hard
  while(this.reversals.length&&now-this.reversals[0]>SHAKE_WINDOW)this.reversals.shift();
  if(speed<SLOW)this.slowSince??=now;else this.slowSince=null;
  if(ragdoll)this.lastShake=now;
  const state=this.state,age=now-this.since,still=now-this.stillSince;
  if(state!=='dizzy'&&(ragdoll||this.reversals.length>=SHAKES)){this.enter('dizzy',now);return;}
  if(state==='grabbed'){if(age>=T.grab-1e-9)this.enter(fast?'carried':'dangling',now);}
  else if(state==='dizzy'){if(now-this.lastShake>=T.dizzyExit&&age>=T.dizzyExit){this.reversals=[];this.annoyedUntil=now+T.annoyed;this.enter('dangling',now);}}
  else if(state==='carried'){if(this.slowSince!==null&&now-Math.max(this.slowSince,this.since)>=T.carryExit-1e-9)this.enter('dangling',now);}
  else if(state==='dangling'){if(fast)this.enter('carried',now);else if(still>=settings.protestSeconds)this.enter('protesting',now);}
  else if(state==='protesting'){if(moved)this.enter(fast?'carried':'dangling',now);else if(still>=Math.max(settings.resignSeconds,settings.protestSeconds+MIN_PROTEST))this.enter('resigned',now);}  // she always protests a while first
  else if(state==='resigned'&&moved){this.perkUntil=now+T.perk;this.enter('carried',now);}
 }
 /** The squash: the reach and build-up over the first 0.26 s on the ground, then the follower back to 0 (it overshoots a little). */
 settle(now,h,scale){
  const age=now-this.landAt,goal=this.squashGoal*scale,reach=Math.min(SQUASH.reach,goal);
  if(age<T.squash){this.squash=age<T.reach?reach*smooth(age/T.reach):reach+(goal-reach)*smooth((age-T.reach)/(T.squash-T.reach));this.squashVelocity=0;this.squashPrevious=this.squash;return;}
  // y + k1 y' + k2 y'' = x + k3 x' toward x = 0; the step of x from the squash to 0 is the initial response (Szanto's second-order dynamics)
  const f=SQUASH.frequency,z=SQUASH.damping,k1=z/(Math.PI*f),k2=1/((TAU*f)**2),k3=SQUASH.response*z/(TAU*f);
  const x=0,xd=h>0?(x-this.squashPrevious)/h:0;this.squashPrevious=x;const k2s=Math.max(k2,h*h/2+h*k1/2,h*k1);
  this.squash+=h*this.squashVelocity;this.squashVelocity+=h*(x+k3*xd-this.squash-k1*this.squashVelocity)/k2s;
 }
 faceIntent(now){
  const state=this.state;
  if(state==='pressed')return now-this.pressAt<T.poke?'poke':null;
  if(state==='poked')return 'poke';
  if(state==='falling')return now<this.bonkUntil?'wince':'falling';
  if(state==='landing'||state==='recovering'){const age=now-this.landAt;
   if(this.squeeze&&age<T.squeeze)return 'squeeze';
   if(this.hard)return age<T.pout?'pout':null;
   if(!this.fell&&(this.prior==='protesting'||this.prior==='dizzy'))return age<T.annoyedLook?'annoyed':null;
   return age<T.relief?'relief':null;}
  if(HELD_STATES.has(state)){const g=now-this.grabAt;
   if(g<T.startle)return 'startle';if(g<T.surprise)return 'surprise';
   if(state==='carried'&&now<this.perkUntil)return 'perk';
   if(state==='resigned'&&now-this.since<T.sighMouth)return 'sigh';
   if(state==='dangling'&&now<this.annoyedUntil)return 'annoyed';
  }
  return FACE[state]||null;
 }
 /** Where she looks: 'viewer', 'holder', 'floor', 'cursor', 'ahead' (eyes at rest, not tracking: dizzy) or null (normal tracking: the
  * cursor when it is near her); avatar_gaze.mjs and the carry rig's head follow it. Grabbed: the holder from 0.15 s, the viewer after
  * the surprise, with a glance back at the holder every 3-6 s while dangling or protesting. Falling: the floor. Landed from a fall: the
  * floor through the squash, then the cursor (a hard landing's pout) or the viewer for a moment. Set down: the cursor after a protest
  * or dizziness. Then normal tracking. */
 look(now){
  const state=this.state,g=now-this.grabAt;
  if(HELD_STATES.has(state)&&state!=='dizzy'&&g<T.surprise)return g>=T.holder?'holder':'viewer';
  if(state==='dangling'||state==='protesting')return now<this.glanceUntil?'holder':'viewer';
  if(state==='landing'||state==='recovering'){const age=now-this.landAt;
   if(this.fell){if(age<T.floorLook)return 'floor';return this.hard?(age<T.floorLook+T.annoyedLook?'cursor':null):age<T.viewerLook?'viewer':null;}
   return (this.prior==='protesting'||this.prior==='dizzy')&&age<T.annoyedLook?'cursor':null;}
  return {carried:'viewer',resigned:'floor',tugged:'cursor',falling:'floor',dizzy:'ahead'}[state]||null;
 }
 /** steps: now (s), dt (s), window motion {vx, vy, ax, ay} from AvatarInertia, ragdoll (the pickup's opt-in limp trigger), settings
  * {enabled, protestSeconds, resignSeconds, recoverySeconds (from the contact), landingSquash}. Returns null when there is nothing to
  * show, else {state, age, transition (pose crossfade seconds), pose (null for face-only states), breath {rate, depth, phase}, face
  * {intent, intensity}, look, blinks [{kind, delay}], suppress (s of no spontaneous blinks), events [{from, to, at}]}. The frame a
  * reaction ends still reports it. */
 step(now,dt,motion={},settings={}){
  if(settings.enabled===false){if(this.state)this.cancel();return null;}
  const h=clamp(dt,0,.1),before=this.state;
  if(!this.state&&!this.pending.events.length)return null;
  if(this.state==='poked'&&now-this.since>=T.poke)this.enter(null,now);
  if(HELD_STATES.has(this.state)&&!this.released)this.track(now,h,motion,{protestSeconds:6,resignSeconds:18,...settings});
  if(this.state==='falling')this.pending.suppress=Math.max(this.pending.suppress,.3);  // eyes wide open on the way down
  if(this.state==='landing'||this.state==='recovering'){
   this.settle(now,h,clamp(settings.landingSquash??1,0,2));
   if(this.state==='landing'&&now-this.landAt>=T.squash-1e-9)this.enter('recovering',now);
   if(now-this.landAt>=(settings.recoverySeconds??2.5))this.enter(null,now);
  }
  const state=this.state;
  if(state==='protesting'&&now>=this.nextBurst){this.burstStart=this.nextBurst;this.burstEnd=this.burstStart+.9+.6*this.random();this.nextBurst=this.burstEnd+2+2*this.random();}
  const burst=state==='protesting'&&now>=this.burstStart&&now<this.burstEnd;
  this.kickRamp=clamp(this.kickRamp+(burst?h/.15:-h/.3));
  this.mirrorEase=clamp(this.mirrorEase+Math.sign(this.mirrorTarget-this.mirrorEase)*Math.min(Math.abs(this.mirrorTarget-this.mirrorEase),2*h/MIRROR_TIME),-1,1);this.mirror=2*smooth((this.mirrorEase+1)/2)-1;  // a change of drag direction crossfades the legs over 0.4 s
  const noise=.6*Math.sin(1.7*now)+.4*Math.sin(2.9*now+1);this.kickPhase=wrap(this.kickPhase+TAU*2.4*(1+.2*noise)*h);
  if((state==='dangling'||state==='protesting')&&now>=this.nextGlance){this.glanceUntil=now+.6+.4*this.random();this.nextGlance=this.glanceUntil+3+3*this.random();}
  const landed=state==='recovering'?now-this.landAt:0,sigh=state==='resigned'&&now<this.sighUntil||state==='recovering'&&this.fell&&!this.hard&&landed<T.squash+T.sigh;
  const [period,depth]=sigh?[3,BREATH.resigned[1]*2]:state==='recovering'&&this.hard?BREATH.protesting:BREATH[state]||[2.2,0];
  this.breathDepth+=(depth-this.breathDepth)*(1-Math.exp(-h/.3));this.breathPhase=wrap(this.breathPhase+TAU*h/period);
  const intensity=this.caught&&HELD_STATES.has(state)&&now-this.grabAt<T.surprise?1.2:1;
  const out={state,age:state?now-this.since:0,transition:this.transition,pose:this.pose(now),breath:{rate:1/period,depth:this.breathDepth,phase:this.breathPhase},
   face:{intent:state?this.faceIntent(now):null,intensity},look:state?this.look(now):null,blinks:this.pending.blinks,suppress:this.pending.suppress,events:this.pending.events};
  this.pending={events:[],blinks:[],suppress:0};
  return state||before||out.events.length?out:null;
 }
 /** The body's part: variant ('tuck', 'reach' or 'swing'), startle, alarm, limp, kick and kickPhase, sway, fall (the falling pose),
  * squash (landing and recovery), mirror (1 to -1, easing toward the sign of the drag so the more bent knee trails). null for the
  * face-only states. */
 pose(now){
  const state=this.state;if(!HELD_STATES.has(state)&&!DROP_STATES.has(state))return null;
  const g=now-this.grabAt,startle=HELD_STATES.has(state)&&g<T.grab?(g<T.startle?smooth(g/T.startle):smooth(1-(g-T.startle)/(T.grab-T.startle))):0;
  return {variant:this.variant,startle,alarm:state==='carried'?1:0,limp:state==='dizzy'||state==='resigned'?1:0,kick:smooth(this.kickRamp),kickPhase:this.kickPhase,
   sway:state==='grabbed'||state==='dangling'||state==='protesting'?1:0,fall:state==='falling'?1:0,squash:state==='landing'||state==='recovering'?this.squash:0,mirror:this.mirror};
 }
}

/** A stand-in for the held pose when reactions are off: the plain dangle, limp while the opt-in ragdoll is active. Shared, read-only. */
const PASSIVE=[false,true].map(ragdoll=>({state:ragdoll?'dizzy':'dangling',transition:.25,pose:{variant:'tuck',startle:0,alarm:0,limp:ragdoll?1:0,kick:0,kickPhase:0,sway:0,fall:0,squash:0,mirror:1},breath:{rate:0,depth:0,phase:0},face:{intent:null,intensity:1},look:null,blinks:[],suppress:0,events:[]}));
export const passiveHold=(ragdoll=false)=>PASSIVE[ragdoll?1:0];

/** Applies a step's face part to an AvatarFace: react on a change of intent, forced blinks, and suppressed spontaneous blinks. */
export class HoldFace{
 constructor(face){this.face=face;this.intent=null;this.intensity=1;}
 apply(out){
  const face=this.face;if(!face)return;
  const intent=out?.face?.intent||null,intensity=out?.face?.intensity??1;
  if(intent!==this.intent||intensity!==this.intensity){face.react(intent,{intensity});this.intent=intent;this.intensity=intensity;}
  for(const blink of out?.blinks||[])face.blink(blink.kind,blink.delay);
  if(out?.suppress>0)face.blinker.suppress(out.suppress);
 }
}
