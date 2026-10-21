import React, {useEffect} from 'react';
import {changeReporter} from './interactive_bridge.mjs';

// The card a crashed window shows. The overlay and the dock are click-through windows: the card takes the pointer only while the
// cursor is over it (popup-interactive and compact-interactive; main ignores the one that is not its window's).
function CrashCard({error}) {
  useEffect(() => {
    const popup = changeReporter(value => window.popupBridge?.interactive(value)), compact = changeReporter(value => window.compactBridge?.interactive(value));
    const hit = event => {const over = !!event.target.closest?.('.crash-card');popup(over);compact(over);};
    document.addEventListener('pointermove', hit, true);document.addEventListener('mousemove', hit, true);
    return () => {document.removeEventListener('pointermove', hit, true);document.removeEventListener('mousemove', hit, true);popup(false);compact(false);};
  }, []);
  return <div className="notice error crash-card" role="alert" style={{position:'fixed', left:8, bottom:8, zIndex:1000, margin:0, maxWidth:'min(360px, calc(100vw - 16px))', maxHeight:'calc(100vh - 16px)', overflow:'auto', flexWrap:'wrap', background:'#2a1418f0', pointerEvents:'auto'}}>
    <div><strong>This window stopped working</strong><p>{String(error?.message || error || 'Unknown error').slice(0, 300)}</p></div>
    <button onClick={() => location.reload()}>Reload</button>
  </div>;
}

/** One per route (main.jsx): a render error shows a Reload card instead of unmounting the whole window, and is logged with its route. */
export default class ErrorBoundary extends React.Component {
  constructor(props) {super(props);this.state = {error: null};}
  static getDerivedStateFromError(error) {return {error};}
  componentDidCatch(error, info) {console.error(`Riko ${this.props.route} window failed`, error, info?.componentStack);}
  render() {return this.state.error ? <CrashCard error={this.state.error}/> : this.props.children;}
}
