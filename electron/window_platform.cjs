const {applyWindowMaterial}=require('./window_material.cjs');
// What each OS adds to Electron's own window behaviour; main and window_manager.cjs ask this instead of process.platform.
// Windows: the click-through overlay hears the left button through native messages, so drags it captured end even outside
// it, and a released button ends a dock or mini-chat drag; the full chat may use acrylic (window_material.cjs).
// Linux: no click-through forwarding (setIgnoreMouseEvents' forward is macOS/Windows only), so main forwards the cursor
// itself (pointerForwarder in overlay_input.cjs). macOS: Electron's forwarding and transparency need nothing extra.
const WM_LBUTTONDOWN=0x0201,WM_LBUTTONUP=0x0202;
function windowPlatform(platform=process.platform){
 const win32=platform==='win32';
 return Object.freeze({platform,
  nativeButtons:win32,  // avatar-cursor reports the button state only where native messages track it
  forwardsPointer:platform==='linux',
  hookOverlay(overlay,{press,release}){if(win32){overlay.hookWindowMessage(WM_LBUTTONDOWN,press);overlay.hookWindowMessage(WM_LBUTTONUP,release);}},
  hookChat(control,release){if(win32)control.hookWindowMessage(WM_LBUTTONUP,release);},
  material:(window,mode,enabled)=>applyWindowMaterial(window,mode,enabled,platform)});
}
// Linux Wayland: the overlay needs X11 (placement, always-on-top, click-through) and Ozone is chosen before main.cjs runs, so a
// launch without a platform (a double-clicked AppImage, a terminal) relaunches once under XWayland. The app.relaunch options, or null.
function x11Relaunch({platform=process.platform,env=process.env,argv=process.argv,hasSwitch}){
 if(platform!=='linux'||!env.WAYLAND_DISPLAY||hasSwitch('ozone-platform'))return null;
 return {...(env.APPIMAGE?{execPath:env.APPIMAGE}:{}),args:[...argv.slice(1),'--ozone-platform=x11']};
}
module.exports={windowPlatform,x11Relaunch,WM_LBUTTONDOWN,WM_LBUTTONUP};
