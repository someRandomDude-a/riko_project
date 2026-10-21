const { app, BrowserWindow, globalShortcut, ipcMain, dialog, screen, session, shell, Tray, Menu, nativeImage, systemPreferences } = require('electron');
// Wiring only: each concern lives in its own module, built here with Electron's objects (src/main_wiring.test.mjs loads this
// file with fakes). Hardware acceleration is Electron's default; keep WebGL and GPU rasterization enabled.
app.commandLine.appendSwitch('enable-gpu-rasterization');
const {windowPlatform,x11Relaunch}=require('./window_platform.cjs');
const relaunch=x11Relaunch({hasSwitch:name=>app.commandLine.hasSwitch(name)});
if(relaunch){app.relaunch(relaunch);app.exit(0);return;}
// A second launch only wakes the running instance. app.quit() before ready still runs whenReady callbacks, which
// would spawn a second backend (loading the model again) and GPT-SoVITS, so stop evaluating this file here.
if(app.isPackaged&&!app.requestSingleInstanceLock()){app.quit();return;}
const path = require('path');
const fs = require('fs');
const release=require('./release.cjs');
const {launchSettings}=require('./launch_config.cjs');
const {resolveBackend,rendererArguments}=require('./backend_origin.cjs');
const {backendSupervisor,microphoneNotice}=require('./backend_supervisor.cjs');
const {windowManager}=require('./window_manager.cjs');
const {registerRoutes}=require('./ipc_router.cjs');
const {desktopRoutes}=require('./ipc_routes.cjs');
const {processTelemetry,displayTelemetry}=require('./desktop_telemetry.cjs');
const {createTray,registerShortcuts}=require('./tray_shortcuts.cjs');
const {prepareAvatarAssets}=require('./avatar_assets.cjs');
// Packaged: bundled resources until whenReady reads the data folder the setup page chose; development: the repository.
let root=app.isPackaged?process.resourcesPath:path.resolve(__dirname,'..');
const brandIcon=path.join(root,'assets','tray.png'),platform=windowPlatform(process.platform),settings={dirty:false};
let tray;  // held, or the tray icon goes with the garbage collector
// The backend's origin, resolved once in whenReady (backend_origin.cjs): RIKO_BACKEND_URL in development, the free port given
// to the backend it spawns when packaged. Windows get it through preload (rikoConfig.backend); none keeps its own copy.
const backend=backendSupervisor({release,dialog,quitting:()=>app.isQuitting});
const patchBoard=body=>backend.request('/api/surfaces/whiteboard',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).catch(()=>{});
const post=route=>backend.request(route,{method:'POST'}).catch(()=>{});
const dev=process.env.RIKO_DEV==='1';
const page=name=>dev?`http://localhost:5173/#/${name}`:`file://${path.join(__dirname,'dist','index.html')}#/${name}`;
// Every window: the preload, isolated from Node, and the origin preload hands the page before its first request.
const rendererPreferences=(extra={})=>({preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false,additionalArguments:rendererArguments(backend.origin),...extra});
const windows=windowManager({BrowserWindow,screen,dialog,app,page,preferences:rendererPreferences,icon:brandIcon,openExternal:url=>shell.openExternal(url),platform,patchBoard,settings});
const processes=processTelemetry({app,request:backend.request}),displays=displayTelemetry({screen,request:backend.request});
const actions={showControls:view=>windows.showControls(view),whiteboard:()=>windows.showWhiteboard(),toggleWhiteboard:()=>windows.toggleWhiteboard(),quit:()=>app.quit(),post,request:backend.request,dialog};
app.on('second-instance',()=>windows.focus());
registerRoutes(ipcMain,windows,desktopRoutes({app,dialog,screen,release,windows,backend,processes,displays,settings,platform,resources:process.resourcesPath}));
app.whenReady().then(async () => {
  try {
    if(!app.isPackaged)backend.connect(resolveBackend(process.env));
    if(app.isPackaged){
      const locator=path.join(app.getPath('userData'),'data-location.json');
      if(!fs.existsSync(locator)){await windows.openSetup();return;}
      root=JSON.parse(fs.readFileSync(locator,'utf8')).directory;
      process.env.RIKO_CONFIG=path.join(root,'character_config.yaml');
      // macOS attributes the backend's microphone use to this app, so it asks now; the answer does not hold up the model load.
      release.askMicrophone(systemPreferences).then(state=>microphoneNotice(state==='blocked',{marker:path.join(app.getPath('userData'),'microphone-blocked-notice'),dialog,shell})).catch(()=>{});
      await backend.spawn(root,process.resourcesPath,()=>processes.publish(true));
    }
    // Resolve the config as run_server does (relative to RIKO_DATA_DIR in development), so main reads the
    // secrets the backend wrote beside it, and scope the token to the origin before any window loads.
    const configPath = path.resolve(!app.isPackaged && process.env.RIKO_DATA_DIR ? process.env.RIKO_DATA_DIR : root, process.env.RIKO_CONFIG || 'character_config.yaml');
    backend.watchSecrets(path.join(path.dirname(configPath),'persistent_memories'));
    backend.injectToken(session.defaultSession.webRequest);
    // What main reads from the YAML, parsed as the backend's PyYAML does (launch_config.cjs: YAML 1.1).
    const {debug,shortcuts,name,sovits}=launchSettings(fs.readFileSync(configPath,'utf8'));
    if(app.isPackaged)backend.startSovits(sovits);
    if(!app.isPackaged)prepareAvatarAssets(path.join(__dirname,'..','character_files'),[path.join(__dirname,'public','models'),path.join(__dirname,'dist','models')]);
    windows.createWindows(name,{debug});
    tray=createTray({Tray,Menu,nativeImage,icon:brandIcon,name},actions);
    for(const event of ['display-added','display-removed','display-metrics-changed'])screen.on(event,()=>{displays.reset();displays.publish();});
    if(debug)windows.get('control').show();
    registerShortcuts(globalShortcut,shortcuts,actions);
  } catch (error) {
    dialog.showErrorBox('Desktop startup failed', error.stack || error.message);
    app.quit();
  }
});
processes.watch();
app.on('will-quit',()=>{globalShortcut.unregisterAll();processes.publish(true,true);});
// Ask before anything stops: once quitting, the backend is stopped first and unsaved Settings are discarded.
app.on('before-quit',event=>{
  const control=windows.get('control');
  if(!app.isQuitting&&settings.dirty&&control&&!control.isDestroyed()){windows.setControlMode('full',undefined,true);if(dialog.showMessageBoxSync(control,{type:'question',buttons:['Quit and discard','Keep editing'],defaultId:1,cancelId:1,message:'Quit and discard unsaved settings changes?'})!==0){event.preventDefault();return;}}
  app.isQuitting=true;
  if(backend.shutdown(()=>app.quit()))event.preventDefault();
  windows.dispose();
});
