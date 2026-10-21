import {useRef, useSyncExternalStore} from 'react';
import {connectEvents} from './event_connection.mjs';
import {createRuntimeStore, shallowEqual} from './runtime_store.mjs';

// One store per window over the shared socket (event_connection.mjs). Its snapshots are structurally shared (runtime_store.mjs).
export const runtimeStore = createRuntimeStore(connectEvents);
const whole = state => state;

/** The latest state.snapshot, or what select takes from it: the component renders again only when that changed (shallowEqual on the
 * shared snapshot), never for a snapshot that changed only fields it does not read. Without select it is the whole snapshot. */
export default function useRuntime(select = whole) {
  const cache = useRef(null);
  const read = () => {
    const state = runtimeStore.get(), last = cache.current;
    if (last && last.state === state && last.select === select) return last.value;
    const value = select(state), kept = last && shallowEqual(last.value, value) ? last.value : value;
    cache.current = {state, select, value: kept};
    return kept;
  };
  return useSyncExternalStore(runtimeStore.subscribe, read, read);
}
