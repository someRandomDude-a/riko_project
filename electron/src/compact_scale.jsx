import React from 'react';
import useWindowGesture from './use_window_gesture.jsx';
export default function CompactScale(){
 const gesture=useWindowGesture('resize',()=>window.windowBridge?.mode('full'));
 return <button className="scale-grip" aria-label="Click for full interface; drag to resize mini chat" {...gesture}><span/></button>;
}
