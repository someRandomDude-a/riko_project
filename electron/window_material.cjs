function applyWindowMaterial(window,mode,enabled,platform=process.platform){
 const blurred=mode==='full'&&enabled;
 let supported=false;
 window.setBackgroundColor('#00000000');
 if(platform==='win32'&&typeof window.setBackgroundMaterial==='function'){
  try{window.setBackgroundMaterial(blurred?'acrylic':'none');supported=true;}catch{/* Retain transparent fallback. */}
 }
 // Clearing material may reset the native fill: restore alpha afterwards.
 if(!blurred||!supported)window.setBackgroundColor('#00000000');
 return {supported,blurred:blurred&&supported};
}
module.exports={applyWindowMaterial};
