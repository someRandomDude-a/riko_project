// Every window loads the same preload, so any renderer can invoke or send on any channel. Main therefore registers its IPC
// only through this table: each route names the windows allowed to use it ({channel, from, handler}, plus `refuse` and
// `invoke` for ipcMain.handle), and the router answers a request only when event.sender is one of those windows' live
// webContents. Anyone else's invoke rejects with the route's refusal; anyone else's send is dropped. The handler receives
// {name, window} for the sender, then the renderer's arguments.
function senderWindow(windows,names,event){
 for(const name of names){const window=windows.get(name);if(window&&!window.isDestroyed()&&window.webContents.id===event?.sender?.id)return {name,window};}
 return null;
}
const handle=(channel,from,refuse,handler)=>({channel,from,refuse,handler,invoke:true});
const on=(channel,from,handler)=>({channel,from,handler,invoke:false});
function registerRoutes(ipcMain,windows,routes){
 const channels=new Set();
 for(const {channel,from,refuse,handler,invoke} of routes){
  if(channels.has(channel))throw new Error('IPC channel registered twice: '+channel);
  if(!Array.isArray(from)||!from.length||typeof handler!=='function'||invoke&&!refuse)throw new Error('IPC route needs its windows, a handler and (for invoke) a refusal: '+channel);
  channels.add(channel);
  if(invoke)ipcMain.handle(channel,(event,...args)=>{const sender=senderWindow(windows,from,event);if(!sender)throw new Error(refuse);return handler(sender,...args);});
  else ipcMain.on(channel,(event,...args)=>{const sender=senderWindow(windows,from,event);if(sender)handler(sender,...args);});
 }
 return channels;
}
module.exports={senderWindow,registerRoutes,handle,on};
