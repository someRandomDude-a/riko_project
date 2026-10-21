import {VRMExpression,VRMExpressionMorphTargetBind} from '@pixiv/three-vrm';

/** The reaction face: robust expression names, face-part expressions from VRoid Fcl_* morphs, a base face crossfaded against a
 * reaction, and eased, event-driven blinks. AvatarFace is the API the hold behaviour (avatar_hold.mjs) drives: react(intent), sequence(steps), blink(kind). */
export const FACE_PARTS={browUp:'Fcl_BRW_Surprised',browAngry:'Fcl_BRW_Angry',browWorry:'Fcl_BRW_Sorrow',eyeWide:'Fcl_EYE_Surprised',eyeSquint:'Fcl_EYE_Angry',eyeSmile:'Fcl_EYE_Joy',eyeSad:'Fcl_EYE_Sorrow',mouthSurprised:'Fcl_MTH_Surprised',mouthSmall:'Fcl_MTH_Small',mouthAngry:'Fcl_MTH_Angry',mouthSad:'Fcl_MTH_Sorrow',mouthWide:'Fcl_MTH_Large',fang:'Fcl_MTH_SkinFung'};
export const MAX_EXPRESSIONS=64;  // the backend's report_capabilities limit
/** The names the backend accepts in a capabilities report: 1-64 characters each (counted as Python counts them), 64 names at most. */
export const reportedExpressions=manager=>Object.keys(manager?.expressionMap||{}).filter(name=>name.length&&[...name].length<=64).slice(0,MAX_EXPRESSIONS);
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,Number.isFinite(v)?v:0));
const ALIASES={surprised:['surprise'],happy:['joy'],sad:['sorrow'],relaxed:['fun'],angry:['anger']};
const BLINKS=new Set(['blink','blinkLeft','blinkRight']),VISEMES=new Set(['aa','ih','ou','ee','oh']);
const EYE_PARTS=['eyeWide','eyeSquint','eyeSad'];  // a blink closes over these, so they give way as the lid falls
const memo=new WeakMap();

/** The registered name for `name`: exact, then case-insensitive, then a VRM 0.x alias (Miku's surprise is 'Surprised'). */
export function resolveExpression(manager,name){
 if(typeof manager?.getExpression!=='function'||typeof name!=='string'||!name)return null;
 if(manager.getExpression(name)?.expressionName===name)return name;  // getExpression does not copy the map (expressionMap does)
 const names=Object.keys(manager.expressionMap);let entry=memo.get(manager);
 if(!entry||entry.count!==names.length){entry={count:names.length,lower:new Map()};for(const n of names)if(!entry.lower.has(n.toLowerCase()))entry.lower.set(n.toLowerCase(),n);memo.set(manager,entry);}
 for(const candidate of [name,...(Object.hasOwn(ALIASES,name.toLowerCase())?ALIASES[name.toLowerCase()]:[])]){const hit=entry.lower.get(candidate.toLowerCase());if(hit)return hit;}
 return null;
}

/** Registers each FACE_PARTS expression whose morph exists, never replacing a name and never past `limit` names in total. */
export function registerFaceParts(vrm,limit=MAX_EXPRESSIONS){
 const manager=vrm?.expressionManager;if(!manager||!vrm.scene)return [];
 const meshes=[];vrm.scene.traverse(o=>{if(o.isMesh&&o.morphTargetDictionary&&o.morphTargetInfluences)meshes.push(o);});
 const added=[];let vroid=false;
 for(const [name,morph] of Object.entries(FACE_PARTS)){
  const byIndex=new Map();for(const mesh of meshes){const index=mesh.morphTargetDictionary[morph];if(index===undefined)continue;if(!byIndex.has(index))byIndex.set(index,[]);byIndex.get(index).push(mesh);}
  if(!byIndex.size)continue;vroid=true;
  if(manager.getExpression(name)||Object.keys(manager.expressionMap).length>=limit)continue;
  const expression=new VRMExpression(name);
  for(const [index,primitives] of byIndex)expression.addBind(new VRMExpressionMorphTargetBind({primitives,index,weight:1}));
  if(name==='eyeSmile')expression.overrideBlink='blend';  // already-smiling eyes must not close a second time
  manager.registerExpression(expression);vrm.scene.add(expression);added.push(name);
 }
 // VRoid's Joy closes the eyes into a smile (about half a blink on Miku), so a full blink on top folds the lids through each other.
 // Fun (relaxed, the resting smile) closes them only about 12%, so it keeps its blinks.
 if(vroid){const happy=manager.getExpression('happy');if(happy&&happy.overrideBlink==='none')happy.overrideBlink='blend';}
 return added;
}

/** How far each expression closes the eyelids at full weight, as a share of a full blink: its morph displacement projected on the
 * blink's over the vertices the blink moves (Miku: Sorrow +0.23, Angry +0.15, Fun +0.12, Surprised -0.11). Map(name -> share). */
export function measureLidClosure(manager){
 const result=new Map(),blink=manager?.getExpression?.('blink');if(!blink?.binds)return result;
 // Each bind counts once per vertex space: the primitives of one glTF mesh share their position and morph attributes.
 const each=(expression,visit)=>{for(const bind of expression.binds||[]){if(!Array.isArray(bind.primitives)||!Number.isInteger(bind.index))continue;const seen=new Set();
  for(const mesh of bind.primitives){const g=mesh?.geometry,space=g?.attributes?.position,delta=g?.morphAttributes?.position?.[bind.index];if(!space||!delta||seen.has(space))continue;seen.add(space);visit(space,delta,g.morphTargetsRelative?null:space,Number.isFinite(bind.weight)?bind.weight:1);}}};
 const fields=new Map();
 each(blink,(space,delta,origin,weight)=>{let field=fields.get(space);if(!field)fields.set(space,field=new Float64Array(space.count*3));
  for(let k=0;k<Math.min(space.count,delta.count);k++){field[3*k]+=weight*(delta.getX(k)-(origin?origin.getX(k):0));field[3*k+1]+=weight*(delta.getY(k)-(origin?origin.getY(k):0));field[3*k+2]+=weight*(delta.getZ(k)-(origin?origin.getZ(k):0));}});
 const moved=new Map();let norm=0;
 for(const [space,field] of fields){const list=[];for(let k=0;k<space.count;k++){const l=field[3*k]**2+field[3*k+1]**2+field[3*k+2]**2;if(l>1e-12){list.push(k);norm+=l;}}moved.set(space,list);}
 if(!(norm>0))return result;
 for(const expression of manager.expressions||[]){const name=expression.expressionName;if(BLINKS.has(name))continue;let dot=0;
  each(expression,(space,delta,origin,weight)=>{const field=fields.get(space);if(!field)return;for(const k of moved.get(space))if(k<delta.count)dot+=weight*((delta.getX(k)-(origin?origin.getX(k):0))*field[3*k]+(delta.getY(k)-(origin?origin.getY(k):0))*field[3*k+1]+(delta.getZ(k)-(origin?origin.getZ(k):0))*field[3*k+2]);});
  if(Math.abs(dot/norm)>.005)result.set(name,dot/norm);}
 return result;
}

const BLINK_SHAPES={spontaneous:{close:.09,hold:.04,open:.17},press:{close:.07,hold:.04,open:.15},startle:{close:.07,hold:.04,open:.15},gaze:{close:.09,hold:.04,open:.17},relief:{close:.1,hold:.25,open:.25},sleepy:{close:.1,hold:.06,open:.25}};  // sleepy: resigned eyes open slowly
const INTERVALS={calm:[2,6],anxious:[1.5,3.5],resigned:[4,7]};

/** Eyelid 0..1: fast ease-in close, a short hold, slower ease-out open; occasional doubles; forced blinks on events. */
export class Blinker{
 constructor({random=Math.random,mood='calm'}={}){this.random=random;this.mood=INTERVALS[mood]?mood:'calm';this.time=0;this.lid=0;this.current=null;this.pending=[];this.quietUntil=0;this.double=false;this.next=this.interval();}
 interval(){const [a,b]=INTERVALS[this.mood];return a+(b-a)*this.random();}
 setMood(mood){if(!INTERVALS[mood]||mood===this.mood)return;this.mood=mood;this.next=Math.min(this.next,this.interval());}
 /** A forced blink ('startle', 'press', 'gaze' or 'relief') after `delay` seconds; suppress() never blocks it. */
 trigger(kind='startle',delay=0){this.pending.push({at:this.time+Math.max(0,delay),kind});}
 /** A blink is running or queued (a gaze-evoked blink then adds nothing). */
 get busy(){return !!this.current||this.pending.length>0;}
 /** No spontaneous blink for `seconds`. */
 suppress(seconds){this.quietUntil=Math.max(this.quietUntil,this.time+Math.max(0,seconds));}
 /** Spontaneous blinks are suppressed now (eyes held wide: a surprise, a fall). */
 get quiet(){return this.time<this.quietUntil;}
 step(dt){
  const h=clamp(dt,0,.1);this.time+=h;let advance=h;  // a blink that starts inside this step advances only from its start
  const due=this.pending.reduce((best,p,i)=>p.at<=this.time+1e-9&&(best<0||p.at<this.pending[best].at)?i:best,-1);
  if(due>=0){const p=this.pending.splice(due,1)[0];advance=clamp(this.time-p.at,0,h);this.current={kind:p.kind,age:0,from:this.lid,shape:BLINK_SHAPES[p.kind]||BLINK_SHAPES.startle};this.double=false;}
  else if(!this.current){this.next-=h;if(this.next<=0){
   if(this.time<this.quietUntil)this.next=this.quietUntil-this.time+this.random()*.3;
   else{advance=clamp(-this.next,0,h);this.current={kind:'spontaneous',age:0,from:this.lid,shape:this.mood==='resigned'?BLINK_SHAPES.sleepy:BLINK_SHAPES.spontaneous};this.double=!this.double&&this.random()<.12;}
  }}
  if(!this.current)return this.lid=0;
  const b=this.current,s=b.shape;b.age+=advance;
  if(b.age<s.close){const u=b.age/s.close;this.lid=b.from+(1-b.from)*u*u;}
  else if(b.age<s.close+s.hold)this.lid=1;
  else if(b.age<s.close+s.hold+s.open)this.lid=(1-(b.age-s.close-s.hold)/s.open)**2;  // a quadratic ease-out: half open after about 30% of the open, at about half the closing speed
  else{this.current=null;this.lid=0;this.next=b.kind==='spontaneous'&&this.double?.18:this.interval();if(b.kind!=='spontaneous'||!this.double)this.double=false;}
  return this.lid;
 }
}

/** Reaction intents: part weights (face parts or preset names), the eyelid, mouth extras ('oh' breathes, 'aa' holds), timing in
 * seconds (rising channels use attack, falling ones release), the reaction's share of the face (weight) and the blink mood.
 * A model without every part shows the fallback preset instead. */
export const INTENTS={
 poke:{parts:{browUp:.4},weight:.35,attack:.05,release:.35,fallback:{surprised:.25}},
 startle:{parts:{browAngry:.3},lid:.8,weight:.6,attack:.04,release:.15,fallback:{angry:.2}},
 surprise:{parts:{browUp:1,eyeWide:.7,mouthSurprised:.5},attack:.05,release:.35,fallback:{surprised:1}},
 worried:{parts:{browWorry:.55,mouthSmall:.3},oh:.2,attack:.3,release:.5,mood:'anxious',fallback:{sad:.5}},
 alarm:{parts:{browUp:.7,eyeWide:.45},aa:.3,attack:.08,release:.4,mood:'anxious',breath:1.6,fallback:{surprised:.7}},
 dizzy:{parts:{eyeSquint:.5,mouthSad:.5},lid:.6,attack:.2,release:.5,fallback:{sad:.6}},
 pout:{parts:{browAngry:.5,mouthSmall:.6,fang:.4},attack:.3,release:.5,breath:1.4,fallback:{angry:.6}},
 annoyed:{parts:{browAngry:.4,mouthAngry:.35},attack:.25,release:.5,fallback:{angry:.5}},
 resigned:{parts:{browWorry:.3},lid:.3,attack:.4,release:.6,mood:'resigned',fallback:{sad:.3}},
 relief:{parts:{relaxed:.45},oh:.2,attack:.25,release:.6},
 perk:{parts:{browUp:.5},attack:.05,release:.35,fallback:{surprised:.3}},
 sigh:{parts:{browWorry:.3},lid:.35,oh:.25,attack:.3,release:.6,mood:'resigned',fallback:{sad:.3}},
 wince:{parts:{eyeSquint:.5,browAngry:.4,mouthSmall:.4},attack:.06,release:.5,fallback:{angry:.4}},
 falling:{parts:{browUp:.9,eyeWide:.8,mouthSurprised:.5},attack:.06,release:.3,mood:'anxious',breath:1.2,fallback:{surprised:.9}},
 squeeze:{parts:{browAngry:.45,mouthSmall:.5},lid:1,attack:.03,release:.12,fallback:{angry:.3}},  // eyes shut on a hard landing
};
/** The base face per backend emotion (scaled by its intensity); neutral, calm and no emotion add the resting smile. */
const EMOTION_FACE={joy:{parts:{happy:1}},love:{parts:{happy:1}},affection:{parts:{happy:1}},amusement:{parts:{happy:1}},excitement:{parts:{happy:1}},sadness:{parts:{sad:1}},anger:{parts:{angry:1}},surprise:{parts:{surprised:1}},fear:{parts:{surprised:1}},confusion:{parts:{browWorry:1},fallback:{sad:.5}},embarrassment:{parts:{eyeSmile:.4,mouthSmall:.3},fallback:{happy:.4}},calm:{parts:{neutral:1},resting:true},neutral:{parts:{neutral:1},resting:true}};
const BASE={attack:.2,release:.3},RULE={weight:1,attack:.08,release:.3,lid:0,oh:0,aa:0};
const approach=(value,goal,tau,h)=>value+(goal-value)*(1-Math.exp(-h/Math.max(.01,tau)));
/** Moves every channel of `values` toward `goals` (time constant tau(name, rising, value)), dropping channels that reach 0. */
function follow(values,goals,h,tau){
 for(const name of goals.keys())if(!values.has(name))values.set(name,0);
 for(const [name,value] of values){const goal=goals.get(name)||0;let next=approach(value,goal,tau(name,goal>value,value),h);if(Math.abs(goal-next)<.002)next=goal;if(next>0)values.set(name,next);else values.delete(name);}
}

/** Mixes a base layer and a reaction layer into expression weights: out = base*(1-w) + reaction. Each layer is smoothed per channel
 * (attack up, release down). w is one smoothed crossfade weight: it rises at the reaction's attack and falls no faster than the
 * slowest falling reaction channel, so the base face comes back exactly as the reaction leaves and a whole-face preset never
 * stacks on the region parts. */
export class FaceMixer{
 constructor(manager,{random=Math.random}={}){
  this.manager=manager;this.blinker=new Blinker({random});
  this.base=new Map();this.reaction=new Map();this.owner=new Map();this.values=new Map();this.spare=new Map();this.out=new Map();this.baseGoal=new Map();this.reactionGoal=new Map();
  this.w=0;this.wGoal=0;this.lidGoal=0;this.wRelease=BASE.release;this.ohAmp=0;this.ohWritten=false;this.lidValue=0;this.lidRelease=BASE.release;this.breath=0;
  this.refresh();
 }
 /** Re-reads the model's expressions; call it after registering more. Names are resolved once and kept. */
 refresh(){
  this.names=new Map();this.cache=new WeakMap();this.closures=measureLidClosure(this.manager);
  // three-vrm's 'block' stops every blink while the weight is above 0, so an always-on resting smile would stop them all.
  // 'blend' is the same at full weight and lets a blink through in proportion below it.
  const relaxed=this.resolve('relaxed'),expression=relaxed&&this.manager.getExpression(relaxed);if(expression?.overrideBlink==='block')expression.overrideBlink='blend';
 }
 /** Every channel has reached its goal (follow snaps the last 0.002), the eyelid and crossfade too, the breathing mouth is still and no
  * blink runs or waits: the next step writes the same weights. The avatar's frame loop pauses asleep only on a settled face. */
 get settled(){
  if(this.blinker.busy||this.w!==this.wGoal||Math.abs(this.lidValue-this.lidGoal)>=.002||this.ohAmp>.002||this.ohWritten)return false;
  const reached=(values,goals)=>{for(const [name,value] of values)if(goals.get(name)!==value)return false;for(const [name,goal] of goals)if(goal>0&&values.get(name)!==goal)return false;return true;};
  return reached(this.base,this.baseGoal)&&reached(this.reaction,this.reactionGoal);
 }
 resolve(name){let hit=this.names.get(name);if(hit===undefined){hit=resolveExpression(this.manager,name);this.names.set(name,hit);}return hit;}
 /** Resolved weights of a {parts, fallback} spec: every part, else the fallback presets, else whatever parts exist. */
 targets(spec){
  let result=this.cache.get(spec);if(result)return result;
  const resolve=parts=>Object.entries(parts||{}).map(([name,weight])=>[this.resolve(name),weight]);
  const parts=resolve(spec.parts),fallback=resolve(spec.fallback).filter(([name])=>name);
  const chosen=parts.every(([name])=>name)||!fallback.length?parts.filter(([name])=>name):fallback;
  result=new Map();for(const [name,weight] of chosen)result.set(name,Math.max(result.get(name)||0,weight));
  this.cache.set(spec,result);return result;
 }
 baseTargets(emotion,restingSmile,assets){
  const goals=this.baseGoal,spec=emotion?.primary?EMOTION_FACE[emotion.primary]:null,intensity=clamp(emotion?.intensity??.65);goals.clear();
  if(spec)for(const [name,weight] of this.targets(spec))goals.set(name,weight*intensity);
  if(!spec||spec.resting){const relaxed=this.resolve('relaxed');if(relaxed)goals.set(relaxed,Math.max(goals.get(relaxed)||0,clamp(restingSmile,0,1)));}
  for(const [name,value] of assets||[]){if(BLINKS.has(name)||VISEMES.has(name))continue;const resolved=this.resolve(name);if(resolved)goals.set(resolved,clamp(value));}
  return goals;
 }
 /** The share of a blink that `name` already closes, unless three-vrm's own blink override handles it. */
 closure(name){const share=this.closures.get(name);return share&&this.manager.getExpression(name)?.overrideBlink==='none'?share:0;}
 /** intent: an INTENTS name or spec; rule: a hit rule {expression, intensity} that replaces it; breathPhase: the body's breath (radians)
  * for the breathing mouth. Returns Map(name -> weight), reused by the next step; a channel that drops out is written once as 0. */
 step(dt,{emotion=null,intent=null,rule=null,breathPhase,restingSmile=.15,intensity=1,assets=null}={}){
  const h=clamp(dt,0,.1),scale=clamp(intensity,0,1.5),spec=typeof intent==='string'?INTENTS[intent]:intent;
  let reaction=null;const ruled=rule?.expression?this.resolve(rule.expression):null,goals=this.reactionGoal;goals.clear();
  if(ruled){reaction=RULE;goals.set(ruled,clamp(rule.intensity??.7));}
  else if(spec){reaction={weight:clamp(spec.weight??1)*Math.min(1,scale),attack:spec.attack??.1,release:spec.release??.4,lid:clamp((spec.lid||0)*scale),oh:clamp((spec.oh||0)*scale),aa:clamp((spec.aa||0)*scale),mood:spec.mood,breath:spec.breath};
   for(const [name,value] of this.targets(spec))goals.set(name,clamp(value*scale));}
  const aa=this.resolve('aa');if(aa&&reaction?.aa)goals.set(aa,Math.max(goals.get(aa)||0,reaction.aa));
  follow(this.base,this.baseTargets(emotion,restingSmile,assets),h,(name,rising)=>rising?BASE.attack:BASE.release);
  // A reaction's channels rise at its attack and fall at its release; the channels of an earlier reaction give way at the new one's attack.
  let falling=0;
  follow(this.reaction,goals,h,(name,rising,value)=>{
   if(rising&&goals.has(name)){this.owner.set(name,reaction.release);return reaction.attack;}
   const tau=goals.has(name)?reaction.release:reaction?reaction.attack:this.owner.get(name)??BASE.release;if(value>.02)falling=Math.max(falling,tau);return tau;});
  for(const name of this.owner.keys())if(!this.reaction.has(name))this.owner.delete(name);
  const weight=this.wGoal=reaction?.weight||0;if(reaction)this.wRelease=reaction.release;
  this.w=approach(this.w,weight,weight>this.w?reaction.attack:falling||(reaction?reaction.attack:this.wRelease),h);if(Math.abs(this.w-weight)<.002)this.w=weight;
  const previous=this.values,values=this.spare,out=this.out;values.clear();out.clear();
  for(const [name,value] of this.base)values.set(name,value*(1-this.w));
  for(const [name,value] of this.reaction)values.set(name,clamp((values.get(name)||0)+value));
  for(const name of previous.keys())if(!values.has(name))out.set(name,0);
  for(const [name,value] of values)out.set(name,value);
  this.values=values;this.spare=previous;
  // The breathing mouth (Mate Engine's FACE_DRAG): 'oh' pulses 0..amplitude with the breath.
  const ohGoal=reaction?.oh||0;this.ohAmp=approach(this.ohAmp,ohGoal,ohGoal>this.ohAmp?reaction.attack:(reaction?.release??BASE.release),h);
  this.breath+=h*2*Math.PI/(reaction?.breath||2.2);const oh=this.resolve('oh');
  if(oh&&(this.ohAmp>.002||this.ohWritten)){const phase=Number.isFinite(breathPhase)?breathPhase:this.breath,value=this.ohAmp>.002?this.ohAmp*(.5-.5*Math.cos(phase)):0;out.set(oh,Math.max(out.get(oh)||0,value));this.ohWritten=value>0;}
  // The eyelid is written only to 'blink': max(blink, the intent's lid), so Fcl_EYE_Close never applies twice.
  const lidGoal=this.lidGoal=reaction?.lid||0;if(lidGoal>this.lidValue)this.lidRelease=reaction.release;
  this.lidValue=approach(this.lidValue,lidGoal,lidGoal>this.lidValue?reaction.attack:this.lidRelease,h);if(this.lidValue<.002&&!lidGoal)this.lidValue=0;
  this.blinker.setMood(reaction?.mood||'calm');const blinkLid=this.blinker.step(h);
  for(const part of EYE_PARTS){const name=this.resolve(part);if(name&&out.has(name))out.set(name,out.get(name)*(1-blinkLid));}
  // Expressions that already lower the lid (sadness, anger) would carry a full blink past closed; scale the blink by what is left.
  const blink=this.resolve('blink'),lid=Math.max(blinkLid,this.lidValue);
  if(blink){let closed=0;if(lid>0&&this.closures.size)for(const [name,value] of out)closed+=this.closure(name)*value;out.set(blink,clamp(lid*(1-closed)));}
  return out;
 }
}

/** The least time (s) between two gaze-evoked blinks. */
export const GAZE_BLINK_GAP=2;
/** One face per model: a queue of reaction steps over the mixer. */
export class AvatarFace{
 constructor(manager,{random=Math.random}={}){this.mixer=new FaceMixer(manager,{random});this.queue=[];this.age=0;this.gazeBlinkAt=-Infinity;}
 get blinker(){return this.mixer.blinker;}
 get intent(){return this.queue[0]?.intent||null;}
 /** The face stands still (FaceMixer.settled) and no timed reaction step is waiting to give way to the next. */
 get settled(){const step=this.queue[0];return this.mixer.settled&&!(step&&Number.isFinite(step.duration??Infinity));}
 /** Holds `intent` (an INTENTS name, a spec or null) for `duration` seconds, then `then` (or the base face). */
 react(intent,{duration=Infinity,intensity=1,then=null}={}){return this.sequence([{intent,duration,intensity},...(then?[{intent:then}]:[])]);}
 /** Steps: {intent, duration (s, default until replaced), intensity, blink (a Blinker kind), blinkDelay, suppress (s)}. */
 sequence(steps){this.queue=steps.map(step=>({...step}));this.age=0;this.begin();return this;}
 blink(kind='startle',delay=0){this.mixer.blinker.trigger(kind,delay);}
 /** A big gaze shift (AvatarEyeGaze.bigShift): a gaze-evoked blink, unless a blink is already running or queued, blinks are
  * suppressed (her eyes held wide in a surprise or a fall), or the last gaze blink was under 2 s ago (a cursor sweeping past her face
  * jumps her eyes past 25 degrees every half second; a blink with each read as twitching). Returns whether it blinked. */
 gazeShift(){const b=this.mixer.blinker;if(b.busy||b.quiet||b.time-this.gazeBlinkAt<GAZE_BLINK_GAP)return false;this.gazeBlinkAt=b.time;this.blink('gaze');return true;}
 begin(){const step=this.queue[0];if(!step)return;if(step.blink)this.blink(step.blink,step.blinkDelay||0);if(step.suppress)this.mixer.blinker.suppress(step.suppress);}
 /** Options: emotion, rule (a hit rule), assets (MotionEngine's expression targets), breathPhase (radians, so the breathing mouth
  * follows the body's breath), settings ({restingSmile, intensity}). */
 step(dt,{emotion=null,rule=null,assets=null,breathPhase,settings={}}={}){
  this.age+=clamp(dt,0,.1);
  while(this.queue.length&&this.age>=(this.queue[0].duration??Infinity)){this.age-=this.queue[0].duration;this.queue.shift();this.begin();}
  const step=this.queue[0],spec=typeof step?.intent==='string'?INTENTS[step.intent]:step?.intent||null;
  return this.mixer.step(dt,{emotion,rule,assets,breathPhase,intent:spec,restingSmile:settings.restingSmile??.15,intensity:(settings.intensity??1)*(step?.intensity??1)});
 }
}
