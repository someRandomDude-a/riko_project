import {clampBubble} from './feedback_model.mjs';
export function beginPopupDrag(point,position){return {x:point.x,y:point.y,left:position.left,top:position.top,position:null};}
export function movePopupDrag(drag,point,size,viewport){drag.position=clampBubble({left:drag.left+point.x-drag.x,top:drag.top+point.y-drag.y},size,viewport);return drag.position;}
