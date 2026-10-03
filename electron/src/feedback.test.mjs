import test from 'node:test';
import assert from 'node:assert/strict';
import {initialVoice,reduceVoice,voiceLabel,microphoneAction} from './voice_state.mjs';
import {feedbackDefaults,normalizeFeedback,borderActive,bubblePlacement,clampBubble} from './feedback_model.mjs';
import {sliderSpec} from './settings_model.mjs';
import fs from 'node:fs';
import {initialReply,reduceOverlayReply} from './overlay_reply.mjs';
test('single mic control starts, wakes or stops according to live state',()=>{
 assert.equal(microphoneAction(initialVoice).path,'/api/voice/start');
 assert.equal(microphoneAction({...initialVoice,enabled:true,phase:'waiting',wake:{active:false}}).path,'/api/voice/activate');
 for(const phase of ['awake','capturing','follow_up'])assert.equal(microphoneAction({...initialVoice,enabled:true,phase}).path,'/api/voice/stop');
 assert.equal(microphoneAction({...initialVoice,enabled:true,wake:{active:true}}).path,'/api/voice/stop');
});

test('mic feedback follows activation immediately without waiting for wake snapshots',()=>{
  let state=reduceVoice(initialVoice,{type:'voice.starting'});
  state=reduceVoice(state,{type:'voice.ready'});assert.equal(state.enabled,true);
  state=reduceVoice(state,{type:'voice.activated'});assert.equal(voiceLabel(state),'Awake — speak now');
  state=reduceVoice(state,{type:'voice.started',payload:{utterance_id:'new'}});assert.equal(voiceLabel(state),'Listening to your utterance');
  state=reduceVoice(state,{type:'voice.transcript',payload:{utterance_id:'old',text:'stale'}});assert.equal(state.transcript.text,'');
  state=reduceVoice(state,{type:'voice.utterance_ended',payload:{utterance_id:'new'}});assert.equal(voiceLabel(state),'Transcribing…');
  state=reduceVoice(state,{type:'voice.stopped'});assert.equal(voiceLabel(state),'Microphone off');assert.equal(state.level,0);
});

test('feedback controls are bounded, disableable and support screen/avatar anchoring',()=>{
  const p=normalizeFeedback({replyOpacity:100,transcriptX:-100,listenBorderStyle:'bad',replyBubble:'off'});
  assert.equal(p.replyOpacity,1);assert.equal(p.transcriptX,0);assert.equal(p.listenBorderStyle,'rainbow');assert.equal(p.replyBubble,'off');
  assert.equal(borderActive({enabled:true,phase:'capturing'},feedbackDefaults),true);
  assert.equal(borderActive({enabled:false,phase:'capturing'},feedbackDefaults),false);
  assert.deepEqual(bubblePlacement({...feedbackDefaults,replyAnchor:'avatar'},'reply',{x:10,y:20,width:200,height:400}),{left:110,top:60});
  assert.deepEqual(bubblePlacement(feedbackDefaults,'transcript',{}),{left:'50vw',top:'92vh'});
  assert.deepEqual(clampBubble({left:'0vw',top:'0vh'},{width:400,height:100},{width:1000,height:700}),{left:212,top:112});
});

test('numeric sliders preserve exact typed numbers outside their typical display range',()=>{
  assert.equal(sliderSpec({kind:'number',path:'runtime.n_ctx',integer:true,min:1,max:1048576},131072).max,131072);
  assert.equal(sliderSpec({kind:'number',path:'runtime.seed',integer:true},-1),null);
});

test('settings edits no longer mirror budgets, temperatures or output into other controls',()=>{
  const page=fs.readFileSync(new URL('./settings_page.jsx',import.meta.url),'utf8');
  const change=page.slice(page.indexOf('function change('),page.indexOf('async function save()'));
  assert.doesNotMatch(change,/syncPool|next\[budget\]|next\[outputPath\]|next\[temperaturePath\]|hf_revision/);
});

test('transparent interaction does not enable focus or throttle feedback when hidden',()=>{
  const host=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8');
  assert.match(host,/overlay\.setFocusable\(false\)/);assert.match(host,/overlay\.showInactive\(\)/);
  assert.doesNotMatch(host,/overlay\.(?:focus|show)\(/);assert.match(host,/backgroundThrottling:false/);
});

test('reply speech clouds stream the current turn and ignore stale snapshots/deltas',()=>{
  let state=reduceOverlayReply(initialReply,{type:'model.started',turn_id:'turn'});
  state=reduceOverlayReply(state,{type:'chat.delta',turn_id:'turn',payload:{text:'New reply'}});
  state=reduceOverlayReply(state,{type:'chat.delta',turn_id:'old',payload:{text:'stale'}});
  state=reduceOverlayReply(state,{type:'chat.completed',turn_id:'turn',payload:{text:'New reply'}});
  state=reduceOverlayReply(state,{type:'state.snapshot',payload:{speech:'old reply',runtime:{generating:false}}});
  assert.equal(state.text,'New reply');assert.equal(state.generating,false);
});
