from process.app_core.runtime.interjections import Interjections


def test_short_interjections_merge_at_first_split():
    stops = []
    policy = Interjections(lambda: stops.append(True))
    policy.activity(0.4)
    policy.transcript('Tuesday.', 6, 1, 1.4)
    policy.transcript('Not Monday.', 12, 1.8, 2.1)
    messages = policy.messages('Hello world again')
    assert stops == []
    assert [m.role for m in messages] == ['assistant', 'user', 'assistant']
    assert messages[0].content == 'Hello '
    assert messages[1].content == '[speaking over you] Tuesday. Not Monday.'
    assert messages[2].content == 'world again'


def test_sustained_activity_interrupts_once_at_threshold():
    stops = []
    policy = Interjections(lambda: stops.append(True))
    policy.activity(1.49)
    assert not stops
    policy.activity(1.5)
    policy.activity(2)
    assert stops == [True]
