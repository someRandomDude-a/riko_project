import numpy as np

from process.app_core.audio.wake_capture import WakeCapture, prepare_audio


def test_vad_finishes_sample_without_fixed_duration():
    capture = WakeCapture()
    frame = b'\x10\x00' * 512
    for _ in range(10): assert capture.feed(frame, False) == (None, None)
    for _ in range(12): assert capture.feed(frame, True) == (None, None)
    result = None
    for _ in range(11):
        outcome, pcm = capture.feed(frame, False)
        if outcome:
            result = outcome, pcm
            break
    assert result[0] == 'complete'
    assert capture.elapsed < 2


def test_five_second_limit_discards_even_when_still_speaking():
    capture = WakeCapture()
    frame = bytes(1024)
    for _ in range(156): assert capture.feed(frame, True) == (None, None)
    assert capture.feed(frame, True) == ('timeout', None)
    assert capture.frames == []


def test_silence_only_timeout():
    capture = WakeCapture()
    for _ in range(156): assert capture.feed(bytes(1024), False) == (None, None)
    assert capture.feed(bytes(1024), False) == ('timeout', None)


def test_reference_and_live_padding_prepare_identical_vectors():
    word = (np.sin(np.arange(8000) * .13) * 10000).astype('<i2').tobytes()
    reference = bytes(6400) + word + bytes(12800)
    live = bytes(19200) + word + bytes(3200)
    np.testing.assert_array_equal(prepare_audio(reference, 24000), prepare_audio(live, 24000))
