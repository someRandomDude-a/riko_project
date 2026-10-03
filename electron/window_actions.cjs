function windowAction(window,action){
  if(!['minimize','maximize','close'].includes(action))throw new Error('Unsupported window action');
  if(action==='minimize')window.minimize();
  if(action==='maximize'){if(window.isMaximized())window.unmaximize();else window.maximize();}
  if(action==='close')window.close(); // Keep the existing hide/unsaved-change handling.
  return {maximized:window.isDestroyed()?false:window.isMaximized()};
}
module.exports={windowAction};
