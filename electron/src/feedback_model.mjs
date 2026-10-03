export const feedbackDefaults={listenBorder:true,listenBorderStyle:'rainbow',listenBorderAnchor:'avatar',listenBorderWhen:'awake',listenBorderColor:'#80ed99',listenBorderWidth:3,
  chatTranscript:true,overlayTranscript:true,transcriptAnchor:'dock',transcriptX:50,transcriptY:92,transcriptOpacity:.85,transcriptWidth:520,transcriptSeconds:12,
  replyBubble:'hidden',replyAnchor:'dock',replyX:50,replyY:10,replyOpacity:.9,replyWidth:480,replySeconds:20,
  popupTheme:true,popupColor:'#182130',popupBlur:20,wakeFlash:true,transcribingBorder:true,
  transcribingBorderStyle:'rainbow',transcribingBorderAnchor:'avatar',transcribingBorderColor:'#b699e6',transcribingBorderWidth:3};
export function normalizeFeedback(raw){
  const value={...feedbackDefaults,...raw};
  for(const key of ['listenBorder','chatTranscript','overlayTranscript','popupTheme','wakeFlash','transcribingBorder'])if(typeof value[key]!=='boolean')value[key]=feedbackDefaults[key];
  for(const [key,options] of Object.entries({listenBorderStyle:['rainbow','solid'],listenBorderAnchor:['avatar','screen'],listenBorderWhen:['mic','awake','utterance'],transcriptAnchor:['avatar','screen','dock'],replyAnchor:['avatar','screen','dock'],replyBubble:['hidden','always','off']}))if(!options.includes(value[key]))value[key]=feedbackDefaults[key];
  if(!/^#[a-f0-9]{6}$/i.test(value.popupColor))value.popupColor=feedbackDefaults.popupColor;
  value.popupBlur=Number.isFinite(value.popupBlur)?Math.max(0,Math.min(48,value.popupBlur)):20;
  if(!/^#[a-f0-9]{6}$/i.test(value.listenBorderColor))value.listenBorderColor=feedbackDefaults.listenBorderColor;
  for(const [key,options] of Object.entries({transcribingBorderStyle:['rainbow','solid'],transcribingBorderAnchor:['avatar','screen']}))if(!options.includes(value[key]))value[key]=feedbackDefaults[key];
  if(!/^#[a-f0-9]{6}$/i.test(value.transcribingBorderColor))value.transcribingBorderColor=feedbackDefaults.transcribingBorderColor;
  value.transcribingBorderWidth=Number.isFinite(value.transcribingBorderWidth)?Math.max(1,Math.min(12,value.transcribingBorderWidth)):feedbackDefaults.transcribingBorderWidth;
  for(const [key,min,max] of [['listenBorderWidth',1,12],['transcriptX',0,100],['transcriptY',0,100],['replyX',0,100],['replyY',0,100],['transcriptOpacity',.1,1],['replyOpacity',.1,1],['transcriptWidth',200,1000],['replyWidth',200,1000],['transcriptSeconds',1,120],['replySeconds',1,120]])value[key]=Number.isFinite(value[key])?Math.min(max,Math.max(min,value[key])):feedbackDefaults[key];
  return value;
}
export function borderActive(voice,preferences){
  if(voice.enabled&&voice.phase==='transcribing')return preferences.transcribingBorder!==false;
  if(!preferences.listenBorder||!voice.enabled)return false;
  return preferences.listenBorderWhen==='mic'||(preferences.listenBorderWhen==='utterance'?voice.phase==='capturing':
    !voice.wake?.calibrating&&!voice.wake?.testing&&(!!voice.wake?.active||['awake','capturing','follow_up'].includes(voice.phase)));
}
export function bubblePlacement(preferences,prefix,geometry){
  const x=preferences[prefix+'X'],y=preferences[prefix+'Y'];
  if(preferences[prefix+'Anchor']==='avatar')return {left:geometry.x+geometry.width*x/100,top:geometry.y+geometry.height*y/100};
  return {left:x+'vw',top:y+'vh'};
}
export function dockPopupPlacement(prefix,bounds,screen,size,viewport){
 const x=bounds.x-screen.x+bounds.width/2,top=bounds.y-screen.y,bottom=top+bounds.height;
 if(prefix==='reply'&&top>=size.height+24)return {left:x,top:top-12};
 if(prefix==='transcript'&&bottom+size.height+24<=viewport.height)return {left:x,top:bottom+size.height+12};
 const right=bounds.x-screen.x+bounds.width+size.width/2+12;
 const left=bounds.x-screen.x-size.width/2-12;
 if(right+size.width/2+12<=viewport.width)return {left:right,top:bottom};
 if(left-size.width/2>=12)return {left,top:bottom};
 return {left:x,top:top>=size.height+24?top-12:bottom+size.height+12};
}

export function clampBubble(position,size,viewport){
  const x=typeof position.left==='string'?parseFloat(position.left)*viewport.width/100:position.left;
  const y=typeof position.top==='string'?parseFloat(position.top)*viewport.height/100:position.top;
  const half=Math.min(size.width/2,Math.max(0,viewport.width/2-12));
  return {left:Math.max(half+12,Math.min(viewport.width-half-12,x)),top:Math.max(size.height+12,Math.min(viewport.height-12,y))};
}
