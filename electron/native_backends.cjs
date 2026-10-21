// The native-backend manifest (native_backends.json) names the bundled llama.cpp builds each OS ships, their library
// file names and build flags; the Python backend (Code/process/app_core/configuration/native_backends.py) and
// tools/release/build.py read the same file. Every function takes the platform, so tests cover each OS on any host.
const fs=require('fs');
const path=require('path');
const MANIFEST=require('./native_backends.json');
function libraryName(platform=process.platform,manifest=MANIFEST){if(!Object.hasOwn(manifest.platforms,platform))throw new Error('No native library is defined for '+platform);return manifest.platforms[platform].library;}
// Shipped for the platform, in preference order.
function backendsFor(platform=process.platform,manifest=MANIFEST){return manifest.backends.filter(backend=>backend.platforms.includes(platform));}
function nativeLibrary(resources,backend,platform=process.platform,manifest=MANIFEST){
 if(!backendsFor(platform,manifest).some(item=>item.id===backend))throw new Error('No '+backend+' native backend ships for '+platform);
 return path.join(resources,'native',backend,libraryName(platform,manifest));
}
// The bundles this installation holds: every default one, and an optional one (hip) only when a release staged it.
function installedBackends(resources,platform=process.platform,manifest=MANIFEST,exists=fs.existsSync){return backendsFor(platform,manifest).filter(backend=>!backend.optional||!!resources&&exists(nativeLibrary(resources,backend.id,platform,manifest)));}
// detected: {backend id: detection output}. The first backend (of those offered; default: shipped) that needs no detection or whose detection found hardware.
function preferredBackend(platform=process.platform,detected={},manifest=MANIFEST,backends=backendsFor(platform,manifest)){return backends.find(backend=>!backend.detect?.required||detected[backend.id])?.id||null;}
module.exports={MANIFEST,libraryName,backendsFor,installedBackends,nativeLibrary,preferredBackend};
