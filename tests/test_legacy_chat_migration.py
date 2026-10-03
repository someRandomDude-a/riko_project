import importlib.util
from pathlib import Path
import sys

import pytest

tools = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('legacy_chat_migration', tools / 'migrate_legacy_chat_history.py')
migration = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(tools))
try:
    spec.loader.exec_module(migration)
finally:
    sys.path.remove(str(tools))


def row(role, text):
    return {'role': role, 'content': [{'type': 'input_text' if role == 'user' else 'output_text', 'text': '[2026-05-08T12:17] ' + text}], 'tokens': 99}


def test_normalizes_duplicate_user_and_assistant_labels():
    result = migration.convert_history([row('user', 'Saucy Jack: Saucy Jack: hello'), row('assistant', 'Riko: Hello!')])
    assert result[0]['content'] == 'Saucy Jack: hello'
    assert result[1]['content'] == 'Hello!'
    assert result[0]['timestamp'] == '2026-05-08T12:17'
    assert 'tokens' not in result[0]


def test_only_envelope_name_is_the_speaker():
    result = migration.convert_history([row('user', 'Saucy_Jack: Senpai: Good morning.')])
    assert result[0]['content'] == 'Saucy_Jack: Senpai: Good morning.'


def test_preserves_colons_and_labels_inside_message_bodies():
    result = migration.convert_history([row('user', 'Saucy Jack: Reminder: do this\nSaucy Jack: quoted text'), row('assistant', 'Note: do this\nRiko: quoted text')])
    assert result[0]['content'] == 'Saucy Jack: Reminder: do this\nSaucy Jack: quoted text'
    assert result[1]['content'] == 'do this\nRiko: quoted text'


def test_custom_character_name():
    assert migration.convert_history([row('assistant', 'Other: hello')])[0]['content'] == 'hello'


def test_refuses_unknown_blocks_instead_of_losing_content():
    with pytest.raises(ValueError):
        migration.convert_history([{'role': 'user', 'content': [{'type': 'image', 'text': 'data'}]}])
