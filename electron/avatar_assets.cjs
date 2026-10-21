const fs=require('fs');
const path=require('path');
// Development only (installed application resources are read-only): copy the repository's VRM models into the renderer's
// public/ and dist/ model folders whenever the source is newer.
function prepareAvatarAssets(source,targets,names=['Mita.vrm','tiny_gremlin.vrm'],files=fs){
 targets.forEach(dir=>files.mkdirSync(dir,{recursive:true}));
 for(const name of names){
  const from=path.join(source,name);
  for(const dir of targets){
   const target=path.join(dir,name);
   if(files.existsSync(from)&&(!files.existsSync(target)||files.statSync(from).mtimeMs>files.statSync(target).mtimeMs))files.copyFileSync(from,target);
  }
 }
}
module.exports={prepareAvatarAssets};
