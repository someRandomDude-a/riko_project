// A global shortcut takes its chord away from every other application, so only the summon shortcut is bound by
// default: Cmd/Ctrl+Shift+Q is macOS Log Out and Chrome's quit, and Cmd/Ctrl+Shift+W closes windows in browsers,
// Finder and editors. Quit, Settings and the whiteboard stay in the tray menu. desktop.shortcuts.<name> binds an
// action; null, '' or false unbinds it, including the default.
const DEFAULT_SHORTCUTS=Object.freeze({popup:'CommandOrControl+Shift+Space'});
function shortcutBindings(configured,names){
 const shortcuts=configured&&typeof configured==='object'&&!Array.isArray(configured)?configured:{};
 return names.flatMap(name=>{const value=Object.hasOwn(shortcuts,name)?shortcuts[name]:DEFAULT_SHORTCUTS[name];return typeof value==='string'&&value.trim()?[[name,value.trim()]]:[];});
}
module.exports={DEFAULT_SHORTCUTS,shortcutBindings};
