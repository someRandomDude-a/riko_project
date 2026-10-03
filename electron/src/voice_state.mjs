export const initialVoice={enabled:false,status:'stopped',phase:'stopped',wake:null,level:0,transcript:null,error:''};
export function microphoneAction(voice){
  if(!voice.enabled)return {path:'/api/voice/start',label:'Turn on microphone',icon:'on'};
  const awake=voice.wake?.active||['awake','capturing','transcribing','follow_up'].includes(voice.phase);
  if(awake)return {path:'/api/voice/stop',label:'Turn off microphone',icon:'off'};
  return {path:'/api/voice/activate',label:'Speak now',icon:'speak'};
}
export function reduceVoice(state,event){
  const p=event.payload||{};
  const status=event.type==='resource.snapshot'?p.voice:event.type==='resource.voice'?p:null;
  if(status)return {...state,enabled:!!status.capture_running||!!status.listening,status:status.microphone_status||'ready',phase:status.phase||state.phase,wake:status.wake,transcript:status.latest_transcript??state.transcript};
  if(event.type==='state.snapshot')return {...state,enabled:!!p.runtime?.listening||p.runtime?.microphone_status==='starting',status:p.runtime?.microphone_status||state.status,phase:p.runtime?.voice_phase||state.phase,wake:p.runtime?.wake||state.wake,transcript:p.runtime?.latest_transcript??state.transcript};
  if(event.type==='voice.starting')return {...state,enabled:true,status:'starting',phase:'starting',error:''};
  if(event.type==='voice.ready')return {...state,enabled:true,status:'ready',phase:state.phase==='starting'?(state.wake?.active?'awake':'waiting'):state.phase,error:''};
  if(event.type==='voice.stopped')return {...state,enabled:false,status:'stopped',phase:'stopped',level:0};
  if(event.type==='voice.level')return {...state,level:state.enabled?Math.min(1,(p.rms||0)*8):0};
  if(event.type==='voice.error')return {...state,error:p.error||'Microphone error'};
  if(event.type==='voice.wake_status')return {...state,wake:p};
  if(event.type==='voice.wake_score')return {...state,wake:state.wake?{...state.wake,last_score:p.score}:null};
  if(event.type==='voice.activated'||event.type==='voice.follow_up')return {...state,phase:event.type==='voice.activated'?'awake':'follow_up',wake:{...state.wake,active:true}};
  if(event.type==='voice.waiting'&&!['capturing','transcribing'].includes(state.phase))return {...state,phase:'waiting'};
  if(event.type==='voice.started')return {...state,phase:'capturing',transcript:{utterance_id:p.utterance_id,text:'',final:false}};
  if(event.type==='voice.resumed'&&state.transcript?.utterance_id===p.utterance_id)return {...state,phase:'capturing'};
  if(['voice.utterance_ended','voice.transcribing','voice.transcript'].includes(event.type)){
    if(state.transcript?.utterance_id&&p.utterance_id!==state.transcript.utterance_id)return state;
    if(event.type==='voice.transcript')return {...state,transcript:p,phase:p.final?(state.wake?.active?'awake':'waiting'):state.phase};
    return {...state,phase:'transcribing'};
  }
  return state;
}
export function voiceLabel(state){
  if(!state.enabled)return 'Microphone off';
  if(state.wake?.testing)return 'Detector testing';
  if(state.wake?.calibrating)return 'Wake-name setup';
  return {starting:'Starting microphone…',awake:'Awake — speak now',follow_up:'Accepting follow-ups',capturing:'Listening to your utterance',transcribing:'Transcribing…'}[state.phase]||
    (state.wake?.mode==='continuous'?'Listening continuously':state.wake?.mode==='manual'?'Ready — use Speak now':state.wake?.enrolled?`Waiting for “${state.wake.wake_word}”`:'Set up your wake name');
}
