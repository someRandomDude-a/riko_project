import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const {syncWindowContent}=createRequire(import.meta.url)('../window_content.cjs');

test('renderer origin stays local across collapsed, mini and full content bounds',()=>{
 const webContents={},calls=[];
 let child={x:-600,y:-400,width:1050,height:800},content;
 const view={webContents,getBounds:()=>child,setBounds:bounds=>{child=bounds;calls.push(bounds);}};
 const window={webContents,contentView:{children:[view]},isDestroyed:()=>false,getContentBounds:()=>content};
 for(const [width,height] of [[88,88],[480,648],[1050,800]]){
  content={x:1200,y:600,width,height};
  syncWindowContent(window);
  assert.deepEqual(child,{x:0,y:0,width,height});
 }
 assert.equal(calls.length,3);
 syncWindowContent(window);
 assert.equal(calls.length,3);
});

test('content alignment leaves unrelated child views and destroyed windows alone',()=>{
 const webContents={},view={webContents:{},getBounds:()=>{throw Error('unrelated child');}};
 syncWindowContent({webContents,contentView:{children:[view]},isDestroyed:()=>false});
 syncWindowContent({isDestroyed:()=>true});
});
