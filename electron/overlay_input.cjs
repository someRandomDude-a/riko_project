// Keep click-through stable during captured gestures; hover alone is not a lock.
function overlayInput(apply){
 const hover=new Set(),drag=new Set();let current=false,buttons=null;
 function sync(){const next=!!(hover.size||drag.size);if(next!==current){current=next;apply(next);}}
 return {hover(name,on){if(on)hover.add(name);else hover.delete(name);sync();},
  drag(name,on){if(on){drag.add(name);buttons=1;}else drag.delete(name);sync();},
  press(){buttons=1;},release(){buttons=0;},
  get interactive(){return current;},get buttons(){return buttons;},get dragging(){return drag.size>0;}};
}
module.exports={overlayInput};
