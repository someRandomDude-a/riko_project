// What main tells the backend about the desktop, each only when it changed: Electron's own processes (the resource panel
// counts them) and the displays (avatar and whiteboard screen pickers). `request` is backend_supervisor's.
const KINDS=['Browser','GPU','Renderer','Utility','Zygote','Sandbox helper'];
function processTelemetry({app,request}){
 let signature='',updates=Promise.resolve();
 // force resends an unchanged list (a renderer asked, the backend now listens); empty reports none as main quits.
 function publish(force=false,empty=false){
  if(!app.isReady())return;
  const body=JSON.stringify({processes:empty?[]:app.getAppMetrics().map(item=>({pid:item.pid,kind:KINDS.includes(item.type)?item.type:'Utility'}))});
  if(!force&&body===signature)return;signature=body;
  updates=updates.then(()=>request('/api/resources/electron',{method:'POST',headers:{'Content-Type':'application/json'},body})).catch(()=>{signature='';});
 }
 function watch(){
  app.whenReady().then(()=>publish());
  app.on('web-contents-created',(_event,contents)=>{for(const event of ['did-finish-load','destroyed','render-process-gone'])contents.on(event,()=>publish());});
  for(const event of ['gpu-info-update','child-process-gone'])app.on(event,()=>publish());
 }
 return {publish,watch};
}
function displayTelemetry({screen,request}){
 let signature='',publishing=false,again=false;
 const list=()=>screen.getAllDisplays().map((display,index)=>({index,id:display.id,
  label:display.label||`Screen ${index+1}`,primary:display.id===screen.getPrimaryDisplay().id,
  bounds:display.bounds,scaleFactor:display.scaleFactor}));
 async function publish(){
  // A change while a post is in flight (macOS sends display-added, then display-metrics-changed) is checked again once it ends.
  if(publishing){again=true;return;}
  const body=JSON.stringify(list());
  if(body===signature)return;
  publishing=true;
  try{const response=await request('/api/displays',{method:'POST',headers:{'Content-Type':'application/json'},body});if(response.ok)signature=body;}
  catch{/* Retry on the next backend snapshot/reconnection. */}
  finally{publishing=false;}
  if(again){again=false;await publish();}
 }
 // The backend lost them (a new backend, or the overlay saw none) or the screens changed: send them again.
 return {list,publish,reset(){signature='';}};
}
module.exports={processTelemetry,displayTelemetry};
