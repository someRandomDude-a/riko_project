/** Letting go of her. A flick at or above the throw speed throws her with the cursor's velocity: she flies under gravity with air
 * drag, bounces off the work area's sides and top, and lands on its floor (the taskbar or dock line). A gentle release sets her down
 * where she is, except that her feet never end below the floor, and a set-down just above it lands on it. Sources: Shimeji's
 * Thrown/Fall (initial velocity from the cursor, per-tick drag, gravity), VPet's fall-and-land. Everything here is in overlay px (y
 * down) for the avatar canvas's top-left corner; pure, no three.js, no timers. Release is the lifecycle the renderer runs (and the
 * real-model smoke test runs the same one): it steps a Drop once per frame, places the canvas and sends the one final place once she
 * stands. */
const G=9.8,SUBSTEP=1/240;
export const THROW={drag:1.2,restitution:.35,maxSpeed:6000};  // 1/s of horizontal air drag, the share of speed kept off a wall, px/s
/** A set-down whose soles end this share of the canvas height (or less) above the floor lands on it (Miku at 720 px: 43 px). */
export const FLOOR_SNAP=.06;
/** A throw that starts on the floor (she stood on the taskbar: the canvas is at it, just above it, or below it after the scruff slide)
 * first hops this far above it: speed x 0.04, 40-160 px, so a sideways flick off the taskbar flies in an arc instead of being set down
 * or landing at once and stopping dead. */
const HOP={share:.04,min:40,max:160};
const finite=(value,fallback)=>typeof value==='number'&&Number.isFinite(value)?value:fallback;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const smoother=u=>{u=clamp(u,0,1);return u*u*u*(u*(6*u-15)+10);};

/** The work area of the overlay's display in overlay px. The overlay covers the display's full bounds, taskbar or dock included, so
 * this is window.screen's available rectangle (global DIP, like window.screenX/Y) relative to the overlay's own position, cut to the
 * viewport. screen: {availLeft, availTop, availWidth, availHeight}; at: {x, y}, the overlay's screen position; viewport: {width, height}.
 * Without a usable available rectangle (or one that leaves under 50 px) it is the whole viewport. Returns {left, top, right, bottom}. */
export function workArea(screen,at,viewport){
 const view={left:0,top:0,right:viewport.width,bottom:viewport.height},x=finite(at?.x,0),y=finite(at?.y,0);
 const width=finite(screen?.availWidth,NaN),height=finite(screen?.availHeight,NaN);if(!(width>0&&height>0))return view;
 const left=finite(screen.availLeft,x)-x,top=finite(screen.availTop,y)-y;
 const area={left:Math.max(0,left),top:Math.max(0,top),right:Math.min(viewport.width,left+width),bottom:Math.min(viewport.height,top+height)};
 return area.right-area.left>=50&&area.bottom-area.top>=50?area:view;
}
/** The canvas y at which her soles touch the work area's floor: footLine is the soles' distance below the canvas top (px). */
export const floorTop=(area,footLine)=>area.bottom-footLine;
/** The canvas position limits that keep her body inside the work area's sides and top. box: her body's extent inside the canvas
 * {left, right, top} (px from the canvas's left and top edges). */
export const flightWalls=(area,box)=>({left:area.left-box.left,right:area.right-box.right,top:area.top-box.top});
/** Gravity in px/s^2 at this scale: 9.8 m/s^2 times the setting (1 is real gravity; the default 0.7 reads better at desktop size). */
export const fallGravity=(worldPerPixel,factor=.7)=>(worldPerPixel>0?G/worldPerPixel:2500/.7)*clamp(finite(factor,.7),0,10);
/** What a release does: 'fall' or 'place'. velocity {x, y} in px/s; settings {release: 'throw' | 'place' | 'fall', throwSpeed};
 * grounded: a set-down would put her on the floor (setDownTarget), where a slow 'fall' is only a set-down. A flick at or above the
 * throw speed flies wherever it starts (from the floor it hops first). */
export function chooseRelease(velocity,{release='throw',throwSpeed=900}={},{grounded=false}={}){
 if(release==='place')return 'place';
 const fast=Math.hypot(finite(velocity?.x,0),finite(velocity?.y,0))>=finite(throwSpeed,900);
 if(release==='fall')return grounded&&!fast?'place':'fall';
 return fast?'fall':'place';
}
/** Where a set-down at canvas y ends: on the floor when her feet would be below it or within snap px above it, measured as she is or
 * without the scruff slide (slid: its y in px, so one picked up off the taskbar by the head and let go lands back on it), else where
 * she is. */
export const setDownTarget=(y,floor,snap=0,slid=0)=>y>floor||floor-y<=snap||floor-(y-finite(slid,0))<=snap?floor:y;
/** How far (px) a throw from at or below the floor rises above it first. */
export const hopHeight=speed=>clamp(finite(speed,0)*HOP.share,HOP.min,HOP.max);
const glideSeconds=distance=>clamp(.15+Math.abs(distance)/1500,.15,.45);

/** One release. A fall flies until she lands; a set-down glides (smootherstep, 0.15-0.45 s) onto the floor if it must, or stands at
 * once. The floor is measured every step as the work area's bottom (where her soles stand) minus foot (her soles below the canvas top
 * with the current camera), and once standing she keeps her soles on their line: on the floor, or where she was set down in mid-air.
 * So the camera's safety net easing back after a release (it moves her soles in the canvas) never leaves her floating, sunk or
 * drifting. */
export class Drop{
 /** start {x, y, vx, vy}: px and px/s. options: fall; gravity (px/s^2); walls {left, right, top} (flightWalls; a wall she starts beyond
  * stays out at her until she is back inside, so a release never pushes her); bottom (the work area's floor, overlay px); foot (px);
  * snap, slid (setDownTarget); hop (px, hopHeight: a fall from where a set-down would put her on the floor, at, below or just above
  * it, first rises to this far above it). */
 constructor({x,y,vx=0,vy=0},{fall=false,gravity=2500,walls={},bottom=Infinity,foot=0,snap=0,slid=0,hop=0,drag=THROW.drag,restitution=THROW.restitution,maxSpeed=THROW.maxSpeed}={}){
  this.x=x;this.y=y;this.vx=finite(vx,0);this.vy=finite(vy,0);this.gravity=gravity;this.drag=drag;this.restitution=restitution;this.maxSpeed=maxSpeed;
  this.bottom=finite(bottom,Infinity);this.foot=finite(foot,0);const floor=this.bottom-this.foot;
  this.raw=walls||{};this.slack={left:Math.max(0,finite(this.raw.left,-Infinity)-x),right:Math.max(0,x-finite(this.raw.right,Infinity)),top:Math.max(0,finite(this.raw.top,-Infinity)-y)};this.walls=this.within(this.raw);
  this.time=0;this.onFloor=false;this.impact=0;this.line=y+this.foot;
  if(fall){
   // From the floor she first rises above it (a hop), or she would land the moment she started and stop dead.
   if(setDownTarget(y,floor,snap,slid)===floor){const need=Math.sqrt(2*Math.max(0,gravity)*Math.max(0,y-floor+finite(hop,0)));if(this.vy>-need)this.vy=-need;}
   this.phase='flying';this.cap();return;
  }
  const target=setDownTarget(y,floor,snap,slid);this.vx=this.vy=0;
  if(target===y){this.phase='standing';this.onFloor=y===floor;return;}
  this.phase='gliding';this.onFloor=true;this.from=y;this.duration=glideSeconds(target-y);
 }
 get flying(){return this.phase==='flying';}
 get gliding(){return this.phase==='gliding';}
 get standing(){return this.phase==='standing';}
 /** Seconds until her feet touch: a flight's time to the floor (ignoring walls), the rest of a glide, 0 standing. */
 get seconds(){
  if(this.phase==='gliding')return Math.max(0,this.duration-this.time);
  if(this.phase!=='flying')return 0;
  const g=this.gravity,d=this.bottom-this.foot-this.y;return g>0?Math.max(0,(-this.vy+Math.sqrt(Math.max(0,this.vy*this.vy+2*g*d)))/g):Infinity;
 }
 within(walls){return {left:finite(walls?.left,-Infinity)-this.slack.left,right:finite(walls?.right,Infinity)+this.slack.right,top:finite(walls?.top,-Infinity)-this.slack.top};}
 cap(){const speed=Math.hypot(this.vx,this.vy);if(speed>this.maxSpeed){this.vx*=this.maxSpeed/speed;this.vy*=this.maxSpeed/speed;}}
 /** Advances dt seconds. ground: {bottom, foot} (re-measured every frame; omitted, the last ones) and walls (flightWalls of her body as
 * it is now: her arms fling out on the way, so the walls move in with them). Returns {x, y, landed (her feet touched this step),
 * impact (px/s, the downward speed at a flight's contact; null when a glide ends), hits (walls struck this step: 'left', 'right',
 * 'top')}. */
 step(dt,{bottom,foot,walls}={}){
  const h=clamp(finite(dt,0),0,.1),hits=[];let landed=false,impact=0;this.time+=h;
  this.bottom=finite(bottom,this.bottom);this.foot=finite(foot,this.foot);const floor=this.bottom-this.foot;
  if(walls){this.raw=walls;this.walls=this.within(walls);}
  if(this.phase==='flying'){
   const g=this.gravity,k=this.drag;let left=h;
   while(left>1e-12&&this.phase==='flying'){
    let t=Math.min(SUBSTEP,left);left-=t;
    // Exact constant-gravity motion, so free fall covers g t^2 / 2 at any frame rate; contact at the exact crossing.
    const y1=this.y+this.vy*t+.5*g*t*t,v1=this.vy+g*t;
    if(v1>=0&&y1>=floor){const d=floor-this.y;t=d<=0?0:(-this.vy+Math.sqrt(Math.max(0,this.vy*this.vy+2*g*d)))/g;impact=this.impact=this.vy+g*t;this.y=floor;this.vy=0;this.phase='standing';this.onFloor=true;this.line=this.bottom;landed=true;}
    else{this.y=y1;this.vy=v1;}
    const e=Math.exp(-k*t);this.x+=k>0?this.vx*(1-e)/k:this.vx*t;this.vx*=e;  // exact linear drag
    if(this.x<this.walls.left){this.x=this.walls.left;if(this.vx<0){this.vx=-this.vx*this.restitution;hits.push('left');}}
    else if(this.x>this.walls.right){this.x=this.walls.right;if(this.vx>0){this.vx=-this.vx*this.restitution;hits.push('right');}}
    if(this.phase==='flying'&&this.y<this.walls.top){this.y=this.walls.top;if(this.vy<0){this.vy=-this.vy*this.restitution;hits.push('top');}}
    if(landed)this.vx=0;else this.cap();
   }
   // Back inside a wall she was released beyond: from now on it holds.
   const raw=this.raw;for(const [side,gap] of [['left',finite(raw.left,-Infinity)-this.x],['right',this.x-finite(raw.right,Infinity)],['top',finite(raw.top,-Infinity)-this.y]])this.slack[side]=Math.min(this.slack[side],Math.max(0,gap));
   this.walls=this.within(raw);
  }
  else if(this.phase==='gliding'){
   const u=smoother(this.time/this.duration);this.y=this.from+(floor-this.from)*u;
   if(this.time>=this.duration){this.phase='standing';this.y=floor;this.line=this.bottom;landed=true;impact=null;}
  }
  else this.y=(this.onFloor?this.bottom:this.line)-this.foot;
  if(this.onFloor&&this.phase==='standing')this.line=this.bottom;
  return {x:this.x,y:this.y,landed,impact:landed?impact:0,hits};
 }
}

/** The release lifecycle: letGo() at pointerup decides and starts a Drop, step() moves it every frame, grab() is a press on her while
 * it runs (the catch), dispose() ends it when the renderer goes away. Hooks:
 * - place(geometry, final): the overlay position (the renderer's positionRef); final=true exactly once per release, when she stands
 *   and the camera's safety net is back at rest (a press or dispose() sends it sooner);
 * - dragging(bool): false once that final place has gone out;
 * - interact(kind): 'recover' when she touches down (always after a flight; after a set-down when the hold lands), unless the
 *   ragdoll's release already posted it;
 * - hold (HoldBehaviour: release, land, bonk) and pickup (AvatarPickup: release, airborne, land, ragdoll, cameraActive);
 * - measure: area() (workArea), foot(rest) (her soles below the canvas top, px, with the current camera or, given true, the one at
 *   rest), box() (her body in the canvas: flightWalls), worldPerPixel(), height() (the canvas, px).
 * A set-down that must glide onto the floor lands when the glide ends (the squash then meets the floor); until then the hold stays
 * as it was and the carry rig treats the glide as a short flight. */
export class Release{
 constructor({place=()=>{},dragging=()=>{},interact=()=>{},hold=null,pickup=null,measure}={}){
  this.hooks={place,dragging,interact};this.hold=hold;this.pickup=pickup;this.measure=measure;this.drop=null;this.geometry=null;this.recovered=false;
 }
 get active(){return !!this.drop;}
 get flying(){return !!this.drop?.flying;}
 /** now (s); final: the canvas geometry where the drag ended {x, y, width, height}; velocity {x, y} (px/s, the cursor's); slid: the
  * scruff slide in px; settings: the pickup settings {release, throwSpeed, fallGravity}. Returns 'fall' or 'place'. */
 letGo(now,final,velocity,{slid=null,settings={}}={}){
  const m=this.measure,area=m.area(),foot=m.foot(),floor=area.bottom-foot,snap=FLOOR_SNAP*m.height(),sy=finite(slid?.y,0);
  const vx=finite(velocity?.x,0),vy=finite(velocity?.y,0),speed=Math.hypot(vx,vy),hold=this.hold,pickup=this.pickup;
  const decision=chooseRelease({x:vx,y:vy},settings,{grounded:setDownTarget(final.y,floor,snap,sy)===floor});
  this.recovered=!!pickup?.ragdoll;  // the ragdoll's release posts 'recover' itself
  if(decision==='fall'){
   const drop=this.drop=new Drop({x:final.x,y:final.y,vx,vy},{fall:true,gravity:fallGravity(m.worldPerPixel(),settings.fallGravity),walls:flightWalls(area,m.box()),bottom:area.bottom,foot,snap,slid:sy,hop:hopHeight(speed)});
   hold?.release(now,{fall:true,thrown:speed>=finite(settings.throwSpeed,900)});pickup?.release({flight:true,seconds:drop.seconds});
  }else{
   const drop=this.drop=new Drop(final,{bottom:area.bottom,foot,snap,slid:sy});
   if(drop.gliding){hold?.release(now,{defer:true});pickup?.release({flight:true,seconds:drop.seconds});}
   else{if(hold?.release(now)==='landing'&&!this.recovered)this.hooks.interact('recover');pickup?.release();}
  }
  this.geometry={...final,x:this.drop.x,y:this.drop.y};this.settle();
  return decision;
 }
 /** One frame. Returns the Drop's step, or null when nothing runs. */
 step(now,dt){
  const drop=this.drop;if(!drop)return null;
  const m=this.measure,area=m.area(),s=drop.step(dt,{bottom:area.bottom,foot:m.foot(),walls:drop.flying?flightWalls(area,m.box()):null});
  if(s.hits.length){this.hold?.bonk(now);if(drop.flying)this.pickup?.airborne(drop.seconds);}
  if(s.landed){const state=this.hold?.land(now,s.impact);this.pickup?.land();if((s.impact!==null||state==='landing')&&!this.recovered)this.hooks.interact('recover');}
  this.geometry={...this.geometry,x:s.x,y:s.y};this.settle();
  return s;
 }
 settle(){
  const done=this.drop.standing&&!this.pickup?.cameraActive;this.hooks.place(this.geometry,done);
  if(done){this.drop=null;this.hooks.dragging(false);}
 }
 /** A press on her. Mid-flight or mid-glide she is caught where she is (grabbed; caught: it was a flight); standing (waiting for the
  * camera) her place is sent now. Returns {caught, grabbed}. */
 grab(){
  const drop=this.drop;if(!drop)return {caught:false,grabbed:false};this.drop=null;
  if(drop.standing){this.hooks.place(this.geometry,true);return {caught:false,grabbed:false};}
  return {caught:drop.flying,grabbed:true};
 }
 /** The renderer goes away mid-release: the final place is where she would have ended (the floor for a flight or a glide, her line
  * standing), measured with the camera at rest, so she is not left drawn mid-air at an unsaved position. */
 dispose(){
  const drop=this.drop;if(!drop)return;this.drop=null;let y=drop.y;
  try{const area=this.measure.area(),foot=this.measure.foot(true);y=(drop.standing&&!drop.onFloor?drop.line:area.bottom)-foot;}catch{}
  this.hooks.place({...this.geometry,x:drop.x,y:Number.isFinite(y)?y:drop.y},true);this.hooks.dragging(false);
 }
}
