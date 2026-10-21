// Keep click-through stable during captured gestures; hover alone is not a lock.
function overlayInput(apply){
 const hover=new Set(),drag=new Set();let current=false,buttons=null;
 function sync(){const next=!!(hover.size||drag.size);if(next!==current){current=next;apply(next);}}
 return {hover(name,on){if(on)hover.add(name);else hover.delete(name);sync();},
  drag(name,on){if(on){drag.add(name);buttons=1;}else drag.delete(name);sync();},
  press(){buttons=1;},release(){buttons=0;},
  get interactive(){return current;},get buttons(){return buttons;},get dragging(){return drag.size>0;}};
}
// Linux has no click-through forwarding (setIgnoreMouseEvents' forward option is macOS/Windows only), so a window
// that ignores the mouse never sees the moves that would make it interactive. While the cursor moves, hand its
// position to each click-through window under it; the renderer replays it to its usual hit tests (preload.cjs).
function pointerForwarder({cursor,targets,bounds,send,interval=33,schedule=setTimeout,cancel=clearTimeout}){
 let timer=null,last=null;
 function tick(){
  timer=schedule(tick,interval);timer?.unref?.(); // a poll never keeps the process alive (Node timers; fakes have no unref)
  const point=cursor();if(last&&point.x===last.x&&point.y===last.y)return;last=point;
  for(const target of targets()){const b=bounds(target);if(point.x>=b.x&&point.y>=b.y&&point.x<b.x+b.width&&point.y<b.y+b.height)send(target,{x:point.x-b.x,y:point.y-b.y});}
 }
 return {start(){if(timer===null)tick();},stop(){if(timer!==null)cancel(timer);timer=null;last=null;}};
}
// The windows pointerForwarder serves. As Electron does, only windows ignoring the mouse get forwarded moves (a stale buttons=0
// move would end the drags of one taking it): the overlay while click-through and not under another window (Electron forwards
// only to the topmost one; `covering` are the chat, whiteboard, setup and training-data windows), and the dock or mini chat
// while click-through and no window gesture runs.
function forwardTargets({point,overlay,overlayInteractive,control,controlIgnoring,gesture,covering=[]}){
 const shown=w=>w&&!w.isDestroyed()&&w.isVisible();
 const covered=[!controlIgnoring&&control,...covering].some(w=>{if(!shown(w))return false;const b=w.getBounds();return point.x>=b.x&&point.y>=b.y&&point.x<b.x+b.width&&point.y<b.y+b.height;});
 return [!overlayInteractive&&!covered&&overlay,controlIgnoring&&!gesture&&control].filter(shown);
}
module.exports={overlayInput,pointerForwarder,forwardTargets};
