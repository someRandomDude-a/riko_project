import React from 'react';
import Explanation from './explanation.jsx';
import BorderSettings from './border_settings.jsx';
export default function FeedbackSettings({preferences:p,update}){
  const toggle=(key,label)=><label className="switch-row" key={key}>{label}<input type="checkbox" role="switch" checked={p[key]} onChange={e=>update({[key]:e.target.checked})}/></label>;
  const choice=(key,label,options)=><label className="setting-field" key={key}><span>{label}</span><select value={p[key]} onChange={e=>update({[key]:e.target.value})}>{options.map(([value,label])=><option key={value} value={value}>{label}</option>)}</select></label>;
  const slider=(key,label,min,max,step=1)=><label className="setting-field" key={key}><span>{label}: {p[key]}</span><input type="range" min={min} max={max} step={step} value={p[key]} onChange={e=>update({[key]:Number(e.target.value)})}/></label>;
  const anchors=[['dock','Near chat dock'],['screen','Screen space'],['avatar','Near avatar']];
   return <section className="panel feedback-settings"><h3>Listening & speech</h3><Explanation simple="Choose how you see listening and replies on your desktop. Changes apply right away."><p className="caption">Positions are percentages of the screen or avatar area. Speech bubbles do not take keyboard focus.</p></Explanation>
    <BorderSettings preferences={p} update={update}/>
    {toggle('chatTranscript','Show live transcript in conversation')}{toggle('overlayTranscript','Show transcript speech cloud on desktop')}
    {toggle('popupTheme','Use system theme for speech popups')}
    <div className="settings-grid"><label className="setting-field"><span>Custom popup color</span><input type="color" disabled={p.popupTheme} value={p.popupColor} onChange={e=>update({popupColor:e.target.value})}/></label>{slider('popupBlur','Popup blur',0,48)}</div>
    <p className="caption">Drag a desktop popup to remember its position. Your transcripts and assistant replies remember separate positions. Choose Near chat dock to restore automatic placement.</p>
    <div className="settings-grid">{choice('transcriptAnchor','Transcript anchoring',anchors)}{slider('transcriptX','Transcript horizontal position (%)',0,100)}{slider('transcriptY','Transcript vertical position (%)',0,100)}{slider('transcriptOpacity','Transcript opacity',.1,1,.05)}{slider('transcriptWidth','Transcript width',200,1000,10)}{slider('transcriptSeconds','Final transcript display seconds',1,120)}</div>
    <div className="settings-grid">{choice('replyBubble','Assistant speech bubble',[['hidden','When chat is hidden / minimized'],['always','Always'],['off','Disabled']])}{choice('replyAnchor','Reply anchoring',anchors)}{slider('replyX','Reply horizontal position (%)',0,100)}{slider('replyY','Reply vertical position (%)',0,100)}{slider('replyOpacity','Reply opacity',.1,1,.05)}{slider('replyWidth','Reply width',200,1000,10)}{slider('replySeconds','Final reply display seconds',1,120)}</div>
  </section>;
}
