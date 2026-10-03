import React from 'react';
import useWindowGesture from './use_window_gesture.jsx';
export default function DashBar({onClick,label='Drag to move; click to collapse',className=''}){
 const gesture=useWindowGesture('move',onClick);
 return <button type="button" className={'dash-bar '+className} aria-label={label} title={label} {...gesture}><span/></button>;
}
