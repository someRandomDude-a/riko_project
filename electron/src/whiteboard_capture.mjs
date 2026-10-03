/** One snapshot scheduled by a visual mutation, never a capture/polling loop. */
export function scheduleBoardCapture({revision,rect,capture,send,ready=Promise.resolve(),schedule=setTimeout,unschedule=clearTimeout}){
  let stopped=false;
  const timer=schedule(async()=>{
    try{
      await ready;if(stopped)return;
      const png=await capture(rect());if(stopped)return;
      await send(revision,png);
    }catch{/* Backend fallback rendering remains available if Electron capture fails. */}
  },250);
  return()=>{stopped=true;unschedule(timer);};
}
