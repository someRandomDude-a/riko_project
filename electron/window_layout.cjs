function clampBounds(bounds,area){const width=Math.min(area.width,Math.round(bounds.width)),height=Math.min(area.height,Math.round(bounds.height));return {x:Math.round(Math.max(area.x,Math.min(area.x+area.width-width,bounds.x))),y:Math.round(Math.max(area.y,Math.min(area.y+area.height-height,bounds.y))),width,height};}
// The one size rule for the small chat surfaces, used by the mode switch (chatBounds), the resize gesture (gestureBounds) and
// the renderer's compact-scale: the dock is a square 88-260 px wide; the mini chat is at least a sixth of the work area wide
// (never more than its height), at most its shorter side, and never shorter than it is wide. A missing height is 1.35x the width.
function compactSize(mode,area,width,height){
 if(mode==='collapsed'){const side=Math.round(Math.max(88,Math.min(260,width)));return {width:side,height:side};}
 const w=Math.round(Math.max(Math.min(area.height,area.width/6),Math.min(area.width,area.height,width)));
 return {width:w,height:Math.round(Math.max(w,Math.min(area.height,height??w*1.35)))};
}
function chatBounds(mode,current,area,anchor,saved){
 const point=anchor||{x:current.x+current.width/2,y:current.y+current.height-58};
 const {width,height}=mode==='full'?{width:saved?.width||1050,height:saved?.height||800}:compactSize(mode,area,saved?.width||(mode==='collapsed'?88:area.width/4),saved?.height||null);
 return clampBounds({x:point.x-width/2,y:point.y+(mode==='collapsed'?height*.6:58)-height,width,height},area);
}
module.exports={clampBounds,compactSize,chatBounds};
