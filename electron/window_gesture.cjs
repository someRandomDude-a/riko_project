const {clampBounds,compactSize}=require('./window_layout.cjs');
function gestureBounds(g,cursor,area){
 const dx=cursor.x-g.cursor.x,dy=cursor.y-g.cursor.y,b=g.bounds;
 if(g.kind==='move')return clampBounds({...b,x:b.x+dx,y:b.y+dy},area);
 const delta=g.mode==='collapsed'&&Math.abs(dy*88/100)>Math.abs(dx)?-dy*88/100:dx;
 const {width,height}=compactSize(g.mode,area,b.width+delta,b.height-dy);
 return clampBounds({x:b.x,y:b.y+b.height-height,width,height},area);
}
// One drag or resize of a frameless window at a time, driven from main: every 16 ms it follows the native cursor from where the
// gesture began (so missed frames never accumulate) until the renderer ends it, the window goes away or a minute passes.
// onBounds(next,gesture) hears each size or position actually applied.
function gestureLoop({screen,setTimer=setTimeout,clearTimer=clearTimeout,now=Date.now,interval=16,limit=60000}){
 let active=null;
 function end(){if(active)clearTimer(active.timer);active=null;}
 function begin({id,window,kind,mode,onBounds}){
  end();
  const g={id,window,kind,mode,bounds:window.getBounds(),cursor:screen.getCursorScreenPoint(),began:now(),timer:null};active=g;
  (function frame(){
   if(active!==g)return;
   if(window.isDestroyed()||now()-g.began>limit){end();return;}
   const cursor=screen.getCursorScreenPoint(),area=screen.getDisplayMatching(kind==='move'?{...g.bounds,x:g.bounds.x+cursor.x-g.cursor.x,y:g.bounds.y+cursor.y-g.cursor.y}:g.bounds).workArea;
   const next=gestureBounds(g,cursor,area),current=window.getBounds();
   if(!Object.keys(next).every(k=>next[k]===current[k])){window.setBounds(next);onBounds?.(next,g);}
   g.timer=setTimer(frame,interval);
  })();
  return g;
 }
 return {begin,end,endIf(id,window){if(active?.id===id&&active.window===window)end();},get active(){return active;}};
}
module.exports={gestureBounds,gestureLoop};
