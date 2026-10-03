"""Token-budgeted context packing without editing durable history."""
from copy import deepcopy
import json


def pack_context(messages, count, context_limit, output_tokens, *, cancelled=lambda: False):
    if type(context_limit) is not int or type(output_tokens) is not int or not 0 < output_tokens < context_limit:
        raise ValueError('Context limit must exceed reserved output tokens')
    original = deepcopy(list(messages))
    payloads, actions = {}, []
    # Keep the latest user input and its entire tool-call/result chain intact.
    users = [i for i, m in enumerate(original) if m.role == 'user']
    if len(users) > 1:
        for start, end in zip(users[:-1], users[1:]):
            actions.append(('remove', list(range(start, end))))
    for i, message in enumerate(original):
        if message.context_kind in {'initiative', 'reflection'}:
            payloads[i] = json.loads(message.content)
            data = payloads[i]
            if message.context_kind == 'initiative':
                actions.extend(('recent', i) for _ in data.get('recent_messages', []))
            else:
                evidence = data.get('evidence', [])
                actions.extend(('related', i) for _ in evidence[1:])
                formation = evidence[0].get('formation_context', {}) if evidence else {}
                actions.extend(('formation_history', i) for _ in formation.get('history', []))
                actions.extend(('formation_field', i, key) for key in formation if key not in {'history', 'captured_at', 'current_input', 'user_name', 'phase'})
        elif message.context_kind == 'optional': actions.append(('remove', [i]))

    def candidate(dropped):
        result, data, removed = deepcopy(original), deepcopy(payloads), set()
        for action in actions[:dropped]:
            kind, index = action[:2]
            if kind == 'remove': removed.update(index)
            elif kind == 'recent': data[index]['recent_messages'].pop(0)
            elif kind == 'related': data[index]['evidence'].pop()
            elif kind == 'formation_history': data[index]['evidence'][0]['formation_context']['history'].pop(0)
            elif kind == 'formation_field': data[index]['evidence'][0]['formation_context'].pop(action[2], None)
        for index, value in data.items(): result[index].content = json.dumps(value, ensure_ascii=False)
        return [m for i, m in enumerate(result) if i not in removed]

    def measure(value):
        if cancelled(): raise RuntimeError('Context packing cancelled')
        tokens = count(value)
        if cancelled(): raise RuntimeError('Context packing cancelled')
        return tokens + output_tokens

    required = measure(original)
    if required <= context_limit: return original
    if not actions:
        raise ValueError(f'Required prompt and output require {required} tokens, exceeding context {context_limit}; inference not started')
    minimum = candidate(len(actions))
    minimum_tokens = measure(minimum)
    if minimum_tokens > context_limit:
        raise ValueError(f'Required prompt and output require {minimum_tokens} tokens, exceeding context {context_limit}; inference not started')
    low, high = 1, len(actions)
    while low < high:
        middle = (low + high) // 2
        if measure(candidate(middle)) <= context_limit: high = middle
        else: low = middle + 1
    packed = candidate(high)
    if measure(packed) > context_limit: packed = minimum
    from ..events.bus import event_bus
    event_bus.publish('context.trimmed', context_limit=context_limit, output_tokens=output_tokens,
        removed_units=high, original_tokens=required)
    return packed


def estimate_tokens(messages, tools=None):
    # Remote providers do not universally expose tokenizers. This conservative
    # UTF-8 byte estimate is explicitly not an exact token measurement.
    return len(json.dumps({'messages': [m.as_dict() for m in messages], 'tools': tools or []}, ensure_ascii=False).encode('utf-8')) + 64
