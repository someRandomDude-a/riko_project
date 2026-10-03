from process.app_core.audio.voice_segments import VoiceSegments


def test_preroll_partial_asr_and_single_final_utterance():
    segments, activities = [], []
    anchor = ('turn', 20)
    recorder = VoiceSegments(segments.append, lambda: anchor,
                            lambda seconds, anchor: activities.append((seconds, anchor)))
    now = 0.0
    def frames(count, speaking, marker):
        nonlocal now
        for _ in range(count):
            now += 0.032
            recorder.feed(marker, speaking, now)
    frames(40, False, b'p')
    frames(10, True, b'a')
    frames(10, False, b's')
    assert len(segments) == 1
    assert not segments[0].final
    assert segments[0].pcm.startswith(b'p' * 32)
    first_id = segments[0].utterance_id
    frames(20, True, b'b')
    frames(32, False, b's')
    assert segments[-1].final
    assert all(segment.utterance_id == first_id for segment in segments)
    assert all(segment.anchor == anchor for segment in segments)
    assert abs(segments[-1].speech_seconds - 0.96) < 0.001
    assert segments[-1].pcm == b''
    assert recorder.utterance_id is None


def test_activity_survives_natural_gaps_until_endpoint():
    calls, parts = [], []
    recorder = VoiceSegments(parts.append, lambda: ('a', 4), lambda seconds, anchor: calls.append(seconds))
    time = 0
    for speech, count in ((True, 30), (False, 12), (True, 20), (False, 32)):
        for _ in range(count):
            time += .032
            recorder.feed(b'x', speech, time)
    assert calls[-1] >= 1.5
    assert sum(part.final for part in parts) == 1
    assert len({part.utterance_id for part in parts}) == 1


def test_wake_activation_discards_preroll_queued_audio_and_keyword_tail():
    segments, starts = [], []
    recorder = VoiceSegments(segments.append, lambda: starts.append(True), lambda *_: None)
    recorder.pre_roll.append(b'old-keyword')
    recorder.activate((10.0, True))
    # Delayed capture packets and rolling detector matches cannot leak wake audio.
    recorder.feed(b'queued-keyword', True, 9.9)
    recorder.feed(b'straddling-keyword', True, 10.01)
    recorder.feed(b'keyword-tail', True, 10.1)
    assert not segments and not starts and not recorder.pre_roll
    # A one-frame VAD dropout inside the keyword is not its endpoint.
    recorder.feed(b'keyword-gap', False, 10.2)
    recorder.feed(b'keyword-tail', True, 10.3)
    for index in range(10): recorder.feed(b'activation-end', False, 10.4 + index * .032)
    recorder.feed(b'request', True, 10.8)
    for index in range(32): recorder.feed(b'silence', False, 10.9 + index * .032)
    assert starts == [True]
    assert b''.join(part.pcm for part in segments) == b'request' + b'silence' * 10
    assert sum(part.final for part in segments) == 1


def test_manual_activation_does_not_wait_for_a_wake_utterance():
    segments = []
    recorder = VoiceSegments(segments.append, lambda: None, lambda *_: None)
    recorder.pre_roll.append(b'before-button')
    recorder.activate((10.0, False))
    recorder.feed(b'queued', True, 9.9)
    recorder.feed(b'request', True, 10.1)
    for index in range(32): recorder.feed(b'silence', False, 10.2 + index * .032)
    assert segments[0].pcm.startswith(b'request')
    assert b'before-button' not in segments[0].pcm and b'queued' not in segments[0].pcm


def test_same_activation_preserves_followup_recordings_and_preroll():
    recorder = VoiceSegments(lambda *_: None, lambda: None, lambda *_: None)
    boundary = (10.0, True)
    recorder.activate(boundary)
    for index in range(10): recorder.feed(b'end', False, 10.1 + index * .032)
    recorder.feed(b'followup-preroll', False, 10.5)
    recorder.activate(boundary)
    recorder.feed(b'request', True, 10.6)
    utterance = recorder.utterance_id
    recorder.activate(boundary)
    assert recorder.utterance_id == utterance
    assert recorder.audio == [b'followup-preroll', b'request']
