export function pointerSampler(interval=80){
 let pending=null,last=-Infinity;
 return {
  queue(event){pending={x:event.clientX,y:event.clientY};},
  take(now,dragging=false){if(!pending||!dragging&&now-last<interval)return null;const point=pending;pending=null;last=now;return point;},
  clear(){pending=null;}
 };
}
