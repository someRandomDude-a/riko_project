import React from 'react';
import useWindowGesture from './use_window_gesture.jsx';
export default function CornerResize(){
 const gesture=useWindowGesture('resize');
 return <button className="corner-resize" aria-label="Resize window: drag top-right corner" title="Drag to resize" {...gesture}
  onKeyDown={e=>{if(!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();window.compactBridge?.scale({width:window.innerWidth+(e.key==='ArrowRight'?10:e.key==='ArrowLeft'?-10:0),height:window.innerHeight+(e.key==='ArrowUp'?10:e.key==='ArrowDown'?-10:0)}).catch(()=>{});}}/>;
}
