import React, {useEffect, useRef, useState} from 'react';
import AvatarRenderer from './avatar_renderer.jsx';
import Whiteboard from './whiteboard.jsx';
import {VideoEffect} from './surfaces.jsx';
import FormattedText from './formatted_text.jsx';
import useRuntime from './use_runtime.jsx';
import {request} from './api.mjs';
import {usePreferences} from './ui_preferences.mjs';
import {appearanceStyle} from './ui/skins.mjs';
import {ApprovalBubble} from './tool_approvals.jsx';
import OverlayFeedback from './overlay_feedback.jsx';
import WindowChrome from './window_chrome.jsx';
import {ExplanationContext} from './explanation.jsx';

export function Overlay() {
  const state = useRuntime();
  const [preferences,updatePreferences]=usePreferences();
  const [geometry, setGeometry] = useState({x: 0, y: 0, width: 480, height: 720});
  const [error, setError] = useState('');
  const dragging=useRef(false);
  useEffect(() => {
    document.documentElement.style.background = document.body.style.background = 'transparent';
  }, []);
  useEffect(() => {
    if (state.avatar_geometry&&!dragging.current) setGeometry(state.avatar_geometry);
  }, [JSON.stringify(state.avatar_geometry)]);
  useEffect(() => {
    window.riko?.syncSurfaces({whiteboard_visible: state.whiteboard_visible,
      whiteboard_geometry: state.whiteboard_geometry, effect: !!state.effect,
      avatar_screen: state.avatar_geometry?.screen, has_displays: !!state.displays?.length});
  }, [state.whiteboard_visible, JSON.stringify(state.whiteboard_geometry), !!state.effect, state.avatar_geometry?.screen, JSON.stringify(state.displays)]);
  function position(x, y, final, dimensions) {
    setGeometry(old => ({...old, ...dimensions, x, y}));
    if (final) request('/api/surfaces/avatar', {method: 'PATCH', body: {x: Math.round(x), y: Math.round(y),
      ...(dimensions ? {width:dimensions.width,height:dimensions.height} : {})}})
      .then(() => setError('')).catch(error => setError(error.message));
  }
  return <>{state.avatar?.enabled!==false&&<AvatarRenderer emotion={state.emotion} modelPath={state.avatar?.model} modelFormat={state.avatar?.format||'auto'} fov={state.avatar?.camera?.fov||30}
    actions={state.actions} animation={state.animation} preferences={preferences} geometry={geometry} onDragging={value=>{dragging.current=value;}} onPosition={position} onError={setError}/>}
    <div className={'theme-scope skin-'+preferences.skin+(preferences.reduceMotion?' reduce-motion':'')} style={appearanceStyle(preferences)}><OverlayFeedback preferences={preferences} update={updatePreferences} geometry={geometry}/>
    {error&&<div className="bubble"><FormattedText text={error}/></div>}<ApprovalBubble overlay/></div></>;
}

export function BoardWindow() {
  const state = useRuntime();
  const [preferences]=usePreferences();
  const [mode,setMode]=useState('full');
  useEffect(()=>{let alive=true;window.windowBridge?.state().then(v=>{if(alive)setMode(v.mode);}).catch(()=>{});const off=window.windowBridge?.subscribe(v=>{if(v.mode)setMode(v.mode);});return()=>{alive=false;off?.();};},[]);
  useEffect(()=>{document.title=[state.character_name,'Whiteboard'].filter(Boolean).join(' — ');},[state.character_name]);
  return <ExplanationContext.Provider value={preferences.advancedExplanations}><div className={'app-shell board-window board-mode-'+mode+' skin-'+preferences.skin+(preferences.reduceMotion?' reduce-motion':'')} style={appearanceStyle(preferences)}><WindowChrome title={state.character_name||''} blur={false}/><Whiteboard collapsed={mode==='collapsed'} commands={state.whiteboard} pages={state.whiteboard_pages}
    modelPage={state.whiteboard_page} clearCommand={state.whiteboard_clear} loaded={!!state.whiteboard_pages} persistenceError={state.board_persistence_error} revision={state.whiteboard_revision} acknowledge/></div></ExplanationContext.Provider>;
}

export function EffectsWindow() {
  const state = useRuntime();
  useEffect(() => {
    document.documentElement.style.background = document.body.style.background = 'transparent';
  }, []);
  return <VideoEffect effect={state.effect}/>;
}
