"""Application-owned, punctuation-prioritized soft word limits for speech."""
import re


DEFAULTS = {'max_words': 40, 'split_window_words': 15,
            'split_priority': ['.!?', ';:', ',', '\n']}
WORDS = re.compile(r"\b\w+(?:['’]\w+)*\b")


def validate_settings(settings):
    values = {**DEFAULTS, **settings}
    for key, low, high in (('max_words', 1, 1000), ('split_window_words', 0, 1000)):
        value = values[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'speech.{key} must be an integer between {low} and {high}')
    if values['split_window_words'] > values['max_words']:
        raise ValueError('speech.split_window_words cannot exceed speech.max_words')
    groups = values['split_priority']
    if not isinstance(groups, list) or not groups or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('speech.split_priority must be a nonempty list of punctuation groups')
    characters = ''.join(groups)
    if any(c not in '.!?;:,\n。！？；：，' for c in characters) or len(set(characters)) != len(characters):
        raise ValueError('speech.split_priority contains unsupported or repeated punctuation')
    return values


class SpeechChunks:
    def __init__(self, settings=None):
        self.settings = validate_settings(settings or {})
        self.pending = ''

    def feed(self, delta, final=False):
        self.pending += delta
        result = []
        while True:
            words = list(WORDS.finditer(self.pending))
            limit = self.settings['max_words']
            if len(words) <= limit:
                break
            lower = max(1, limit - self.settings['split_window_words'])
            # Prefer punctuation within the look-back window ending at the
            # word threshold. If absent, wait for the first future boundary.
            candidates = []
            count = 0
            for position, character in enumerate(self.pending):
                while count < len(words) and words[count].end() <= position:
                    count += 1
                if count < lower: continue
                for rank, group in enumerate(self.settings['split_priority']):
                    if character in group:
                        # Ignore decimal points and intra-word apostrophe-like
                        # punctuation: a boundary needs whitespace/end/quotes.
                        end = position + 1
                        while end < len(self.pending) and self.pending[end] in '.!?。！？"”’\')]}':
                            end += 1
                        if end < len(self.pending) and not self.pending[end].isspace():
                            break
                        candidates.append((count, rank, end))
                        break
            nearby = [c for c in candidates if c[0] <= limit]
            if nearby:
                boundary = min(nearby, key=lambda c: (c[1], limit-c[0], -c[2]))[2]
            elif candidates:
                boundary = min(candidates, key=lambda c: c[2])[2]
            else:
                break
            text = self.pending[:boundary].strip()
            self.pending = self.pending[boundary:].lstrip()
            if text: result.append(text)
        if final:
            tail = self.pending.strip()
            if tail: result.append(tail)
            self.pending = ''
        return result
