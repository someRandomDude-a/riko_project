"""Convert legacy content-block chat history to the current ChatMessage format.

Save the supplied JSON as a private file outside the repository, then run:
  python tools/migrate_legacy_chat_history.py OLD_JSON --output NEW_JSON
Add --write after reviewing the dry-run. Existing files are never overwritten.
Stop the app before installing output as memory.history_file/history_file.
This is conversation history, not a set of verified factual memories. It does
not promote assistant claims into facts or run inference/reflection.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

from migrate_legacy_memories import read_json


def normalize_speaker(text, role):
    # The first name before ':' is envelope metadata, not message content.
    match = re.match(r'^([^:\n]+):\s?', text)
    if not match:
        raise ValueError('Expected the fixed timestamp + speaker name + colon format')
    if role == 'assistant':
        return text[match.end():]
    speaker = match[1]
    remainder = text[match.end():]
    while True:
        duplicate = re.match(r'^([^:\n]+):\s?', remainder)
        if not duplicate or duplicate[1] != speaker:
            break
        remainder = remainder[duplicate.end():]
    return speaker + ': ' + remainder


def convert_history(raw):
    if not isinstance(raw, list):
        raise ValueError('Expected a JSON list of messages')
    result = []
    for index, item in enumerate(raw):
        label = f'message {index + 1}'
        if not isinstance(item, dict) or item.get('role') not in {'user', 'assistant'}:
            raise ValueError(f'{label}: expected a user or assistant message')
        content = item.get('content')
        if not isinstance(content, list) or not content:
            raise ValueError(f'{label}: expected nonempty content blocks')
        texts = []
        for block in content:
            if not isinstance(block, dict) or block.get('type') not in {'input_text', 'output_text'} or not isinstance(block.get('text'), str):
                raise ValueError(f'{label}: unsupported block; refusing to silently discard content')
            texts.append(block['text'])
        text = '\n'.join(texts)
        match = re.match(r'^\[([^\]\n]+)\]\s?', text)
        stamp = item.get('timestamp')
        if match:
            try:
                datetime.fromisoformat(match[1])
            except ValueError:
                pass  # A non-date bracketed prefix is actual content.
            else:
                if stamp is not None and stamp != match[1]:
                    raise ValueError(f'{label}: conflicting embedded and metadata timestamps')
                stamp = match[1]
                text = text[match.end():]
        if not isinstance(stamp, str):
            raise ValueError(f'{label}: missing ISO timestamp; refusing to invent one')
        datetime.fromisoformat(stamp)
        text = normalize_speaker(text, item['role'])
        # Leave naive timestamps naive: the legacy export supplies no timezone.
        result.append({'role': item['role'], 'content': text, 'timestamp': stamp})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    try:
        source, output = args.source.resolve(), args.output.resolve()
        if source == output or output.exists():
            raise ValueError('Output must be a new file, distinct from the source')
        if not output.parent.is_dir():
            raise ValueError('Output directory must already exist')
        _, raw = read_json(source)
        records = convert_history(raw)
        payload = (json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
        if args.write:
            with output.open('xb') as stream:
                try:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException:
                    stream.close()
                    output.unlink()
                    raise
        print(json.dumps({'mode': 'written' if args.write else 'dry-run', 'messages': len(records), 'output': str(output), 'source_modified': False}, indent=2))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f'Migration failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
