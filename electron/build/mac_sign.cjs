// electron-builder's mac.sign hook (electron-builder.yml). electron-builder 26.0.12 signs only with a keychain identity
// (CSC_LINK, or a Developer ID it discovers) and otherwise skips signing: the app would keep Electron's stock signature,
// broken by the renamed executable and Info.plist, with no hardened runtime and none of entitlements.mac.plist. This
// signs with the identity electron-builder found, or ad hoc ('-') without one, through the same @electron/osx-sign call and
// per-file options (hardened runtime, entitlements) electron-builder would use. Notarization still needs a real identity.
const path=require('path');
function signOptions(options){return {...options,identity:options.identity||'-',identityValidation:false};}
// osx-sign comes with electron-builder (app-builder-lib's pinned dependency), so resolve it from there.
function osxSign(){return require(require.resolve('@electron/osx-sign',{paths:[path.dirname(require.resolve('app-builder-lib/package.json'))]}));}
// electron-builder calls sign(options, packager); the packager is not used. Like electron-builder's own signing, retry: with
// a real identity every Mach-O gets a timestamp from Apple's server, which sometimes drops a request.
const retry={attempts:3,delay:5000};
async function sign(options){
  for(let attempt=1;;attempt++){
    try{return await osxSign().signAsync(signOptions(options));}
    catch(error){if(attempt>=retry.attempts)throw error;await new Promise(resolve=>setTimeout(resolve,retry.delay));}
  }
}
module.exports=sign;
module.exports.retry=retry;
module.exports.signOptions=signOptions;
module.exports.osxSign=osxSign;
