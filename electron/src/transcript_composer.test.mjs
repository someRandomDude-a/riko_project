import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {initialTranscriptComposer,reduceTranscriptComposer,transcriptMatches} from './transcript_composer.mjs';
const event=(type,text='',final=false,id='a')=>({type,timestamp:100,payload:{utterance_id:id,text,final}});
test('live composer replaces provisional text and sends the final text exactly once',()=>{
 let state=reduceTranscriptComposer(initialTranscriptComposer,event('voice.started'));
 state=reduceTranscriptComposer(state,event('voice.transcript','hello'));
 state=reduceTranscriptComposer(state,event('voice.transcript','hello there'));
 assert.equal(state.text,'hello there');assert.equal(state.phase,'live');
 state=reduceTranscriptComposer(state,event('voice.transcript','hello there!',true));
 assert.equal(state.phase,'sending');
 assert.equal(reduceTranscriptComposer(state,event('voice.transcript','old partial')),state);
 state=reduceTranscriptComposer(state,{type:'transcript.dismiss',id:'a'});
 assert.equal(state.text,'');assert.equal(state.phase,'idle');
 assert.equal(reduceTranscriptComposer(state,event('voice.transcript','hello there!',true)),state);
});
test('old utterances and dismissal timers cannot replace new speech',()=>{
 const state=reduceTranscriptComposer(initialTranscriptComposer,event('voice.started','','','b'));
 assert.equal(reduceTranscriptComposer(state,event('voice.transcript','stale',true)),state);
 assert.equal(reduceTranscriptComposer(state,{type:'transcript.dismiss',id:'a'}),state);
});
test('snapshots do not revive old recognized speech and disconnect clears the composer',()=>{
 assert.equal(reduceTranscriptComposer(initialTranscriptComposer,{type:'state.snapshot',payload:{runtime:{latest_transcript:{text:'stale',final:true}}}}),initialTranscriptComposer);
 const state=reduceTranscriptComposer(initialTranscriptComposer,event('voice.transcript','current'));
 for(const type of ['voice.stopped','voice.error','connection.closed'])assert.equal(reduceTranscriptComposer(state,{type}).phase,'idle');
});
test('pending speech matches wake-prefix-stripped chat input without matching unrelated text',()=>{
 assert.equal(transcriptMatches('Hey name,  hello there','hello there'),true);
 assert.equal(transcriptMatches('hello','different message'),false);
 assert.equal(transcriptMatches('',''),false);
});
test('chat uses only the transcript composer popup, with no recognized-speech duplicate',()=>{
 const source=fs.readFileSync(new URL('./stream_chat.jsx',import.meta.url),'utf8');
 assert.match(source,/<SpeechPopup inline sending=.*prefix="transcript"/);
 assert.doesNotMatch(source,/popupReply|Recognized speech|Live transcript — provisional/);
 assert.match(source,/visibleMessages/);
 const overlay=fs.readFileSync(new URL('./overlay_feedback.jsx',import.meta.url),'utf8');
 assert.match(overlay,/visible=\{!mini&&replyVisible/);
});
test('size changes resize the native window once; rendering stays GPU-enabled',()=>{
 const source=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8');
 const transition=source.split('function tweenBounds(')[1].split('function setControlMode(')[0];
 assert.equal((transition.match(/\.setBounds\(/g)||[]).length,1);
 assert.doesNotMatch(transition,/Math\.round|function frame/);
 assert.doesNotMatch(source,/disableHardwareAcceleration|disable-gpu/);
  assert.match(source,/applyWindowMaterial\(control,mode,fullControlBlur\)/);
  assert.match(source,/w===control&&controlMode==='full'\)fullControlBlur=enabled/);
 assert.match(source,/appendSwitch\('enable-gpu-rasterization'\)/);
 assert.match(source,/!windowGesture\)control.setIgnoreMouseEvents/);
});
