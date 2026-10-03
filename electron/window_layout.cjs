function clampBounds(bounds,area){const width=Math.min(area.width,Math.round(bounds.width)),height=Math.min(area.height,Math.round(bounds.height));return {x:Math.round(Math.max(area.x,Math.min(area.x+area.width-width,bounds.x))),y:Math.round(Math.max(area.y,Math.min(area.y+area.height-height,bounds.y))),width,height};}
function chatBounds(mode,current,area,anchor,saved){
 const point=anchor||{x:current.x+current.width/2,y:current.y+current.height-58};
 const width=mode==='collapsed'?saved?.width||88:mode==='compact'?Math.min(area.height,Math.max(area.width/6,Math.min(area.width,saved?.width||area.width/4))):saved?.width||1050;
 const height=mode==='collapsed'?width:mode==='compact'?Math.max(width,Math.min(area.height,saved?.height||width*1.35)):saved?.height||800;
 return clampBounds({x:point.x-width/2,y:point.y+(mode==='collapsed'?height*.6:58)-height,width,height},area);
}
module.exports={clampBounds,chatBounds};
