import pytest

from process.app_core.audio.speech_chunks import SpeechChunks, validate_settings


def test_short_reply_waits_for_generation_end():
    splitter = SpeechChunks({'max_words': 8, 'split_window_words': 4})
    assert splitter.feed('Hello there. How are you? ') == []
    assert splitter.feed('', final=True) == ['Hello there. How are you?']


def test_sentence_ending_has_priority_over_nearer_comma():
    splitter = SpeechChunks({'max_words': 8, 'split_window_words': 4})
    assert splitter.feed('one two three four. five six seven, eight nine ten') == ['one two three four.']
    assert splitter.feed('', final=True) == ['five six seven, eight nine ten']


def test_waits_past_limit_until_future_punctuation():
    splitter = SpeechChunks({'max_words': 4, 'split_window_words': 2})
    assert splitter.feed('one two three four five six') == []
    assert splitter.feed(' seven, eight') == ['one two three four five six seven,']
    assert splitter.feed('', final=True) == ['eight']


def test_final_tail_without_boundary_is_not_forced_apart():
    splitter = SpeechChunks({'max_words': 4, 'split_window_words': 2})
    assert splitter.feed('one two three four five six', final=True) == ['one two three four five six']


def test_old_boundary_outside_window_is_not_used():
    splitter = SpeechChunks({'max_words': 8, 'split_window_words': 2})
    assert splitter.feed('one. two three four five six seven eight nine') == []


def test_comma_can_be_configured_above_period():
    splitter = SpeechChunks({'max_words': 8, 'split_window_words': 4, 'split_priority':[',', '.!?']})
    assert splitter.feed('one two three four. five six seven, eight nine') == ['one two three four. five six seven,']


def test_no_text_is_lost_at_any_stream_boundary():
    source = 'one two three four. five six, seven eight nine ten. End.'
    for size in range(1, len(source)):
        splitter = SpeechChunks({'max_words': 8, 'split_window_words': 4})
        pieces = []
        for offset in range(0, len(source), size):
            pieces.extend(splitter.feed(source[offset:offset+size]))
        pieces.extend(splitter.feed('', final=True))
        assert ''.join(''.join(pieces).split()) == ''.join(source.split())


@pytest.mark.parametrize('settings', [{'max_words':True}, {'max_words':0}, {'split_window_words':41},
    {'split_priority':[]}, {'split_priority':['.', '.']}, {'split_priority':['abc']}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError): validate_settings(settings)
