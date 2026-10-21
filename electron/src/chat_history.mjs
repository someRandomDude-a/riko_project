import {share} from './runtime_store.mjs';
export function mergeHistory(current, incoming) {
  const byID = new Map(current.map(message => [message.id, message]));
  for (const message of incoming) {
    const old = byID.get(message.id);
    const merged = old && (old.event_sequence || 0) > (message.event_sequence || 0)
      ? {...message, ...old, sequence:message.sequence} : {...old, ...message};
    // A message the page repeats unchanged keeps its object, so its memoized bubble does not render again after every turn.
    byID.set(message.id, old ? share(old, merged) : merged);
  }
  return [...byID.values()].sort((a, b) => (a.sequence ?? Infinity) - (b.sequence ?? Infinity));
}
