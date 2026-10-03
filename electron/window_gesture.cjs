const {clampBounds}=require('./window_layout.cjs');
function gestureBounds(g,cursor,area){
 const dx=cursor.x-g.cursor.x,dy=cursor.y-g.cursor.y,b=g.bounds;
 if(g.kind==='move')return clampBounds({...b,x:b.x+dx,y:b.y+dy},area);
 const min=g.mode==='collapsed'?88:Math.min(area.height,area.width/6);
 const delta=g.mode==='collapsed'&&Math.abs(dy*88/100)>Math.abs(dx)?-dy*88/100:dx;
 const width=Math.round(Math.max(min,Math.min(g.mode==='collapsed'?260:Math.min(area.width,area.height),b.width+delta)));
 const height=g.mode==='collapsed'?width:Math.round(Math.max(width,Math.min(area.height,b.height-dy)));
 return clampBounds({x:b.x,y:b.y+b.height-height,width,height},area);
}
module.exports={gestureBounds};
