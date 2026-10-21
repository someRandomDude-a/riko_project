import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {tempDir} from './electron_fakes.mjs';
const {prepareAvatarAssets}=createRequire(import.meta.url)('../avatar_assets.cjs');
test('development copies the bundled models into public/ and dist/ only when the source is newer',()=>{
 const root=tempDir('riko-assets-'),source=path.join(root,'character_files'),targets=[path.join(root,'public','models'),path.join(root,'dist','models')];
 try{
  fs.mkdirSync(source);fs.writeFileSync(path.join(source,'Mita.vrm'),'v1');
  prepareAvatarAssets(source,targets);
  for(const dir of targets){assert.equal(fs.readFileSync(path.join(dir,'Mita.vrm'),'utf8'),'v1');assert.ok(!fs.existsSync(path.join(dir,'tiny_gremlin.vrm')),'a missing model is skipped');}
  fs.writeFileSync(path.join(targets[0],'Mita.vrm'),'edited');fs.utimesSync(path.join(targets[0],'Mita.vrm'),new Date(),new Date(Date.now()+60000));
  // Newer than the plain copy, older than the edited one, whatever the filesystem's timestamp resolution.
  fs.writeFileSync(path.join(source,'Mita.vrm'),'v2');fs.utimesSync(path.join(source,'Mita.vrm'),new Date(),new Date(Date.now()+30000));prepareAvatarAssets(source,targets);
  assert.equal(fs.readFileSync(path.join(targets[0],'Mita.vrm'),'utf8'),'edited','a newer copy is kept');assert.equal(fs.readFileSync(path.join(targets[1],'Mita.vrm'),'utf8'),'v2');
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
