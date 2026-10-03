function syncWindowContent(window){
 if(window.isDestroyed())return;
 const view=window.contentView?.children.find(child=>child.webContents===window.webContents);
 if(!view)return;
 const {width,height}=window.getContentBounds();
 const target={x:0,y:0,width,height},current=view.getBounds();
 // Child-view coordinates are local, not the desktop coordinates of the window.
 if(Object.keys(target).some(key=>current[key]!==target[key]))view.setBounds(target);
}
module.exports={syncWindowContent};
