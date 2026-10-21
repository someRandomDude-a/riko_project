export function pointerSampler(interval=80){
 let pending=null,last=-Infinity;
 return {
  queue(event){pending={x:event.clientX,y:event.clientY};},
  take(now,dragging=false){if(!pending||!dragging&&now-last<interval)return null;const point=pending;pending=null;last=now;return point;},
  clear(){pending=null;}
 };
}
/** When the renderer tells the backend where the pointer is (POST /api/animation/interaction, kind 'pointer'): when it comes near her
 * or goes (the near flag flips), and while it stays near and keeps moving once per keepAlive ms, inside the 2 s after which the
 * backend forgets it (animation/runtime.py). A pointer moving anywhere else on the screen costs no request. Gaze is local. */
export function pointerReporter(keepAlive=1000){
 let near=false,last=-Infinity;
 return {
  due(pointer,now){const next=!!pointer?.near;if(next===near&&!(next&&now-last>=keepAlive))return false;near=next;last=now;return true;},
  reset(){near=false;last=-Infinity;},
 };
}
