import {useEffect, useState} from 'react';
import {connectEvents} from './event_connection.mjs';

export default function useRuntime() {
  const [state, setState] = useState({});
  useEffect(() => connectEvents(event => {
    if (event.type === 'state.snapshot') setState(event.payload);
  }), []);
  return state;
}
