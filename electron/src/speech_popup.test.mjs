import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {normalizeFeedback,dockPopupPlacement} from './feedback_model.mjs';
const viewport={width:1920,height:1080},screen={x:0,y:0},size={width:400,height:100};
test('popup defaults follow the dock and saved role positions stay independent',()=>{
 const p=normalizeFeedback({transcriptAnchor:'screen',transcriptX:27,transcriptY:70});
 assert.equal(p.transcriptAnchor,'screen');assert.equal(p.transcriptX,27);
 assert.equal(p.replyAnchor,'dock');assert.equal(p.replyX,50);assert.equal(p.popupTheme,true);
 assert.equal(normalizeFeedback({popupBlur:200,popupColor:'invalid'}).popupBlur,48);
});
test('transcripts prefer below the dock and replies prefer above',()=>{
 const b={x:900,y:500,width:88,height:100};
 assert.deepEqual(dockPopupPlacement('transcript',b,screen,size,viewport),{left:944,top:712});
 assert.deepEqual(dockPopupPlacement('reply',b,screen,size,viewport),{left:944,top:488});
});
test('popups use adjacent free space instead of covering a bottom-edge dock',()=>{
 const b={x:900,y:980,width:88,height:100};
 assert.deepEqual(dockPopupPlacement('transcript',b,screen,size,viewport),{left:1200,top:1080});
});
test('transparent chat clears document layers and leaves button material alone',()=>{
 const css=fs.readFileSync(new URL('./ui/morph.css',import.meta.url),'utf8');
 assert.match(css,/html\.transparent-chat body/);
 assert.match(css,/\.app-shell,\.app-body,\.primary-view,\.chat-host,\.stream-chat/);
 const host=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8');
 assert.match(host,/thickFrame:false, hasShadow:false/);
 assert.match(host,/windowGesture!==g/);
 assert.match(host,/!windowGesture\)control.setIgnoreMouseEvents/);
});
