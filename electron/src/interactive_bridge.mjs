/** A yes/no the renderer reports to main on pointer moves (popup-interactive, compact-interactive): sent only when it changes, so a
 * moving cursor costs no IPC. reset() forgets the last value, for when main changed that state itself (a chat mode switch). */
export function changeReporter(send){
 let last;
 const report=value=>{value=!!value;if(value===last)return false;last=value;send(value);return true;};
 report.reset=()=>{last=undefined;};
 return report;
}
// The overlay's speech popups share one: the hit test, a popup's enter and leave and the end of a drag all report through it.
export const popupInteractive=changeReporter(value=>globalThis.window?.popupBridge?.interactive(value));
