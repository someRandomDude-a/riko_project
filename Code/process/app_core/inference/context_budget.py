"""Token-budgeted context packing without editing durable history."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import re

TOKEN_CHARS = 32  # Tokens average 3-5 characters; tool output beyond 32 per token of context is cut before measuring.
TRUNCATED = '\n[tool output truncated to fit the context: {} characters omitted]\n'


def pack_context(messages, count, context_limit, output_tokens, *, cancelled=lambda: False, state=None):
    """Fit messages into context_limit, reserving output_tokens.

    Reductions, in order: whole past user turns (oldest first), initiative/reflection payload items,
    optional system context, then, last, the current turn's tool output (head and tail kept, marked).
    History is measured outward from the newest turn, so the cost follows the context size, not the
    history length. A caller-owned `state` dict keeps the retained start between calls: the prompt
    prefix (and the server's KV cache) stays byte-identical until the prompt overflows; that trim
    leaves history three quarters of its room, and the window re-expands only below half of it.
    """
    if type(context_limit) is not int or type(output_tokens) is not int or not 0 < output_tokens < context_limit:
        raise ValueError('Context limit must exceed reserved output tokens')
    original = list(messages)
    payloads, actions = {}, []
    # Keep the latest user input and its entire tool-call/result chain intact; optional context
    # inside a past turn has its own removal action rather than leaving with the turn.
    users = [i for i, m in enumerate(original) if m.role == 'user']
    actions.extend(('remove', [j for j in range(start, end) if original[j].context_kind != 'optional']) for start, end in zip(users[:-1], users[1:]))
    turns = len(actions)
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
    tools = [i for i in range(users[-1] if users else 0, len(original)) if original[i].role == 'tool']
    longest = max((len(original[i].content) for i in tools), default=0)
    ceiling = TOKEN_CHARS * context_limit
    initial = ceiling if longest > ceiling else None

    def candidate(dropped, cap):
        data, removed = deepcopy(payloads) if dropped > turns else payloads, set()
        for action in actions[:dropped]:
            kind, index = action[:2]
            if kind == 'remove': removed.update(index)
            elif kind == 'recent': data[index]['recent_messages'].pop(0)
            elif kind == 'related': data[index]['evidence'].pop()
            elif kind == 'formation_history': data[index]['evidence'][0]['formation_context']['history'].pop(0)
            elif kind == 'formation_field': data[index]['evidence'][0]['formation_context'].pop(action[2], None)
        result = {i: replace(m) for i, m in enumerate(original) if i not in removed}
        for index, value in data.items():
            if index in result: result[index].content = json.dumps(value, ensure_ascii=False)
        for index in tools if cap is not None else ():
            text = result[index].content
            if len(text) > cap: result[index].content = text[:cap - cap // 4] + TRUNCATED.format(len(text) - cap) + text[len(text) - cap // 4:]
        return list(result.values())

    measured = {}
    def measure(dropped, cap=initial):
        if (dropped, cap) not in measured:
            if cancelled(): raise RuntimeError('Context packing cancelled')
            measured[dropped, cap] = count(candidate(dropped, cap)) + output_tokens
            if cancelled(): raise RuntimeError('Context packing cancelled')
        return measured[dropped, cap]

    def least(low, high, limit):
        # Fewest reductions in [low, high] that fit; `high` itself must fit the context.
        while low < high:
            middle = (low + high) // 2
            if measure(middle) <= limit: high = middle
            else: low = middle + 1
        return high

    def gallop(top, limit):
        # Fewest dropped turns in [0, top] that fit `limit`, probing outward from the newest turns.
        good, step = top, 1
        while good > 0:
            probe = max(good - step, 0)
            if measure(probe) > limit: return least(probe + 1, good, limit)
            good, step = probe, step * 2
        return 0

    def key(turn):
        message = original[users[turn]]
        return message.role, message.content, message.timestamp

    anchor = None
    if state is not None and users and state.get('key'):
        saved = state.get('dropped')
        anchor = saved if type(saved) is int and 0 <= saved <= turns and key(saved) == state['key'] else next(
            (turn for turn in range(turns, -1, -1) if key(turn) == state['key']), None)

    def history():
        if anchor is not None:
            tokens = measure(anchor)
            if tokens <= context_limit and (anchor == 0 or tokens >= context_limit - (context_limit - state.get('floor', 0)) // 2): return anchor
        floor = measure(turns)
        if floor > context_limit: return None
        if state is None: return gallop(turns, context_limit)
        state['floor'] = floor
        if anchor is None and gallop(turns, context_limit) == 0: return 0  # a cold start keeps a history that already fits
        target = context_limit - (context_limit - floor) // 4
        if anchor is not None and measure(anchor) > context_limit: return least(anchor + 1, turns, target)
        return gallop(turns if anchor is None else anchor, target)

    dropped, cap = history(), initial
    if dropped is None:
        if measure(len(actions)) <= context_limit: dropped = least(turns + 1, len(actions), context_limit)
        elif not tools or measure(len(actions), 0) > context_limit:
            raise ValueError(f'Required prompt and output require {measure(len(actions), 0 if tools else initial)} tokens, '
                f'exceeding context {context_limit}; inference not started')
        else:
            # Last resort: one character cap for every current tool result, largest that fits.
            dropped, low, high = len(actions), 0, min(longest, ceiling)
            while high - low > 64:
                middle = (low + high + 1) // 2
                if measure(dropped, middle) <= context_limit: low = middle
                else: high = middle - 1
            cap = low
    if state is not None:
        if users: state.update(dropped=min(dropped, turns), key=key(min(dropped, turns)))
        else: state.clear()
    if dropped != (anchor or 0) or cap is not None:
        from ..events.bus import event_bus
        event_bus.publish('context.trimmed', context_limit=context_limit, output_tokens=output_tokens,
            removed_units=dropped, required_tokens=measure(dropped, cap), tool_output_chars=cap)
    return deepcopy(candidate(dropped, cap))


# Remote providers expose no tokenizer, so text is split roughly the way BPE tokenizers pre-split it.
# Calibrated on Riko's own prompts against the Qwen3/3.5, Llama 3, Gemma 3, Mistral Nemo and DeepSeek V3
# tokenizers: within 10% of the largest of their counts, or above it, for English, JSON, code, ids, digits,
# timestamps, CJK, Hangul, Cyrillic and emoji, and 1.15-1.5x a templated live prompt (UTF-8 bytes were 3-4x).
# Older 32k-vocabulary tokenizers (Llama 2, Phi-3) split CJK and Hangul into up to twice as many tokens.
_PIECES = re.compile(r' ?[0-9]|[A-Z\u00c0-\u00de]*[a-z\u00df-\u07ff]+|[A-Z\u00c0-\u00de]+|\s+|[!-/:-@\[-`{-~]+|.', re.S)


# High-entropy runs (base64, JWTs, API keys) and random-looking letters split into far smaller pieces than words do.
_RUNS = re.compile(r'[A-Za-z0-9+/=_-]{16,}')
_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def _kind(char): return 'l' if char.islower() else 'u' if char.isupper() else 'd' if char.isdigit() else 'o'


def _letters(piece):
    if not piece.isascii(): return math.ceil(len(piece) / 2.5)
    caps = len(piece) - len(piece.lstrip(_UPPER))
    if caps == len(piece): return 1 if caps == 1 else math.ceil(caps / 3)  # all-caps words split more than lowercase ones
    rest, cost = piece[caps:], (math.ceil(caps / 3) if caps > 1 else 0)
    vowels = sum(char in 'aeiouy' for char in rest)
    random_like = len(rest) >= 6 and (vowels < .25 * len(rest) or re.search(r'[^aeiouy]{4}', rest))
    return cost + math.ceil(len(rest) / (1.8 if random_like else 6))


def _pieces(text):
    total = 0
    for piece in _PIECES.findall(text):
        first = piece[0]
        if piece[-1] in '0123456789': total += len(piece)  # digits split one by one, apart from a space before them
        elif first.isspace(): total += 0 if piece == ' ' else max(1, len(piece) // 8)  # a single space joins the next word
        elif first.isalpha() and ord(first) < 0x800: total += _letters(piece)
        elif first.isascii(): total += math.ceil(len(piece) / 2)  # punctuation runs
        else: total += 4 if ord(first) >= 0x10000 else 1  # each CJK/kana/Hangul character or symbol; emoji are byte-split
    return total


def estimate_text_tokens(text):
    # A run that keeps switching between upper case, lower case and digits is ~1.3 characters per token.
    return _pieces(text) + sum(max(0, math.ceil(len(run) / 1.3) - _pieces(run)) for run in _RUNS.findall(text)
        if sum(_kind(a) != _kind(b) for a, b in zip(run, run[1:])) > .3 * len(run))


def estimate_tokens(messages, tools=None):
    """Conservative prompt estimate: content, chat-template framing and tool schemas (rendered up to ~1.4x compact)."""
    total = 64
    for message in messages:
        total += 8 + estimate_text_tokens(message.content) + (estimate_text_tokens(message.tool_call_id) if message.tool_call_id else 0)
        for call in message.tool_calls:
            total += 24 + estimate_text_tokens(call.id + call.name + json.dumps(call.arguments, ensure_ascii=False))
    if tools: total += 64 + math.ceil(1.25 * estimate_text_tokens(json.dumps(tools, ensure_ascii=False)))
    return total
