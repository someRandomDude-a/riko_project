/** The avatar's frame pacing (avatar_renderer.jsx). Standing still she draws about 30 frames a second; moving she draws every display
 * frame; asleep (once she has settled) or hidden she draws nothing until something wakes her. avatarGraphics.frameRate 'full' keeps
 * every display frame while she is visible. */
export const IDLE_FPS=30,SLEEP_SETTLE_MS=2000,POINTER_ACTIVE_MS=1500,MOTION_TAIL_MS=1000;
export const FRAME_RATES=[['adaptive','Adaptive: about 30 fps standing still, full rate while she moves'],['full','Always the display’s full rate']];
// Motion modes that move her visibly every frame (motion_engine.mjs): talking and walking, and the hold, click and settle reactions.
const MOVING=new Set(['speaking','walking','hold','held','clicked','settling','recovering']);
// Actions that play something once: a gesture, an animation preview, a wake animation.
const ONE_SHOT=new Set(['gesture','motion.preview','wake_animation']);

/** Whether MotionEngine is animating more than an idle stance: a moving mode, a crossfade to a new pose, a one-shot action, or a
 * base clip that does not loop. A looping idle clip is standing still. */
export function motionBusy(actions=[],mode='idle',blending=false){
 return !!blending||MOVING.has(mode)||actions.some(a=>a?.status==='running'&&(ONE_SHOT.has(a.kind)||a.kind==='motion.base'&&a.payload?.asset&&a.payload.asset.loop===false));
}
/** Whether she moves this frame: held or dragged, thrown, falling or being set down (the camera easing back too), walking, animating
 * (motionBusy), in a hold reaction, the pointer moved over or near her within the last 1.5 s (her eyes, head and hair follow it), or
 * she moved within the last second (movedAt: her springs and the carry rig settle after a walk, a drag or a set-down). */
export function avatarActive({dragging=false,released=false,walking=false,animating=false,reacting=false,pointerAt=-Infinity,movedAt=-Infinity,now=0}){
 return !!(dragging||released||walking||animating||reacting)||now-pointerAt<POINTER_ACTIVE_MS||now-movedAt<MOTION_TAIL_MS;
}
/** Milliseconds until the next frame: 0 is the next display frame, Infinity is paused until woken. quietFor is how long she has
 * been inactive and settled whether her face has reached its expression with no blink running (AvatarFace.settled): asleep, she
 * pauses only once her pose, springs and face have settled. */
export function frameInterval({mode='adaptive',active=false,hidden=false,asleep=false,quietFor=0,settled=true}={}){
 if(hidden)return Infinity;
 if(mode==='full'||active)return 0;
 if(asleep&&settled&&quietFor>=SLEEP_SETTLE_MS)return Infinity;
 return 1000/IDLE_FPS;
}
/** Runs step() once per frame, where step returns the next interval (frameInterval). 0 asks for the next display frame. A finite wait
 * sleeps on a timer that ends about one display frame (lead ms) before the frame is due, then follows display frames until it is due
 * (within slack ms), so the frames in between cost nothing: no animation-frame callback, no begin-frame. Infinity schedules nothing.
 * A step that throws still schedules the next frame (at the idle rate) before its error propagates, so one bad frame never stops the
 * loop. wake() brings a waiting or paused frame forward to the next display frame. */
export function frameLoop(step,{request=globalThis.requestAnimationFrame,cancel=globalThis.cancelAnimationFrame,schedule=setTimeout,unschedule=clearTimeout,now=()=>performance.now(),lead=16,slack=4}={}){
 let frame=null,timer=null,stopped=false,due=0;
 const ask=()=>{if(frame===null&&!stopped)frame=request(run);};
 function run(){
  frame=null;if(stopped)return;
  const start=now();
  if(start<due-slack){ask();return;}  // a display frame before the one it is due on
  let wait=1000/IDLE_FPS;
  try{wait=step();}
  finally{
   if(!stopped){
    if(!(wait>0)){due=0;ask();}
    else if(Number.isFinite(wait)){due=start+wait;timer=schedule(()=>{timer=null;ask();},Math.max(0,wait-lead));}
   }
  }
 }
 return {
  start(){ask();},
  wake(){if(stopped)return;due=0;if(timer!==null){unschedule(timer);timer=null;}ask();},
  stop(){stopped=true;if(frame!==null)cancel(frame);if(timer!==null)unschedule(timer);frame=timer=null;},
  get paused(){return frame===null&&timer===null&&!stopped;},
 };
}
