import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {MANIFEST,libraryName,backendsFor,installedBackends,nativeLibrary,preferredBackend}=require('../native_backends.cjs');
// Written out, not derived: each OS is checked on every host (tests/test_release_config.py holds the same tables).
const LIBRARIES={win32:'riko-native.dll',linux:'libriko-native.so',darwin:'libriko-native.dylib'};
const SHIPPED={win32:['cuda','vulkan'],linux:['cuda','hip','vulkan'],darwin:['metal']};  // what bundled:<backend> resolves
const RELEASED={win32:['cuda','vulkan'],linux:['cuda','vulkan'],darwin:['metal']};  // in every release; hip only when a release adds it

test('every OS names its native library and ships its own backends',()=>{
  assert.deepEqual(Object.keys(MANIFEST.platforms).sort(),Object.keys(LIBRARIES).sort());
  for(const [platform,library] of Object.entries(LIBRARIES)){
    assert.equal(libraryName(platform),library);
    assert.deepEqual(backendsFor(platform).map(backend=>backend.id),SHIPPED[platform]);
    for(const backend of SHIPPED[platform])assert.equal(nativeLibrary('/app/resources',backend,platform),path.join('/app/resources','native',backend,library));
    assert.deepEqual(backendsFor(platform).filter(backend=>!backend.optional).map(backend=>backend.id),RELEASED[platform]);
    for(const backend of ['cuda','hip','vulkan','metal'].filter(id=>!SHIPPED[platform].includes(id)))assert.throws(()=>nativeLibrary('/app/resources',backend,platform),new RegExp('No '+backend+' native backend ships for '+platform));
    for(const backend of ['../../other','','CUDA'])assert.throws(()=>nativeLibrary('/app/resources',backend,platform),/native backend ships/);
  }
  for(const platform of ['freebsd','constructor','__proto__'])assert.throws(()=>libraryName(platform),/No native library is defined/);
});

test('setup prefers CUDA only when nvidia-smi finds a GPU, and Metal on macOS',()=>{
  assert.equal(preferredBackend('linux',{cuda:'NVIDIA GeForce RTX 4090, 24564, 550.54.15',vulkan:''}),'cuda');
  assert.equal(preferredBackend('win32',{cuda:'',vulkan:'GPU0: AMD Radeon'}),'vulkan');
  assert.equal(preferredBackend('linux',{}),'vulkan');
  assert.equal(preferredBackend('darwin',{cuda:'NVIDIA'}),'metal');
  assert.equal(preferredBackend('freebsd',{}),null);
});

test('the opt-in hip bundle counts only where installed, and is preferred over Vulkan only when ROCm finds an AMD GPU',()=>{
  const hip=path.join('/app/resources','native','hip','libriko-native.so'),seen=[];
  const exists=file=>{seen.push(file);return file===hip;};
  assert.deepEqual(installedBackends('/app/resources','linux',MANIFEST,()=>false).map(backend=>backend.id),['cuda','vulkan']);
  assert.deepEqual(installedBackends('/app/resources','linux',MANIFEST,exists).map(backend=>backend.id),['cuda','hip','vulkan']);
  assert.deepEqual(seen,[hip],'only optional bundles are looked for');
  assert.deepEqual(installedBackends(null,'linux',MANIFEST,()=>true).map(backend=>backend.id),['cuda','vulkan'],'no resources: development');
  assert.deepEqual(installedBackends('/app/resources','win32',MANIFEST,()=>true).map(backend=>backend.id),['cuda','vulkan']);
  const installed=installedBackends('/app/resources','linux',MANIFEST,exists);
  assert.equal(preferredBackend('linux',{hip:'Name: gfx1100'},MANIFEST,installed),'hip');
  assert.equal(preferredBackend('linux',{hip:'Name: gfx1100',cuda:'NVIDIA'},MANIFEST,installed),'cuda');
  assert.equal(preferredBackend('linux',{hip:'Name: gfx1100'},MANIFEST,installedBackends('/app/resources','linux',MANIFEST,()=>false)),'vulkan');
  assert.equal(preferredBackend('linux',{},MANIFEST,installed),'vulkan');
});

test('a new bundle is one manifest entry: CPU never preselected over a GPU backend',()=>{
  const manifest={platforms:MANIFEST.platforms,backends:[...MANIFEST.backends,{id:'cpu',label:'CPU',description:'no GPU',platforms:['win32','linux','darwin']}]};
  assert.deepEqual(backendsFor('linux',manifest).map(backend=>backend.id),['cuda','hip','vulkan','cpu']);
  assert.equal(preferredBackend('linux',{},manifest),'vulkan');
  assert.equal(preferredBackend('darwin',{},manifest),'metal');
  assert.equal(nativeLibrary('/r','cpu','darwin',manifest),path.join('/r','native','cpu','libriko-native.dylib'));
});
