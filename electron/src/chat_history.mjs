export function mergeHistory(current, incoming) {
  const byID = new Map(current.map(message => [message.id, message]));
  for (const message of incoming) {
    const old = byID.get(message.id);
    byID.set(message.id, old && (old.event_sequence || 0) > (message.event_sequence || 0)
      ? {...message, ...old, sequence:message.sequence} : {...old, ...message});
  }
  return [...byID.values()].sort((a, b) => (a.sequence ?? Infinity) - (b.sequence ?? Infinity));
}
