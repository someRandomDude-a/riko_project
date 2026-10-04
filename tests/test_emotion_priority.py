import threading
from types import SimpleNamespace
from process.app_core.emotion.worker import EmotionWorker
from process.app_core.emotion.julia import JuliaEmotionEngine
from process.app_core.emotion.models import EmotionState


def transcript_engine(context_tokens=1024):
    engine = JuliaEmotionEngine(None)
    engine.context_tokens = context_tokens
    engine._interpret = lambda stream, text: EmotionState()
    return engine


def test_partial_transcripts_replace_full_utterance_snapshot():
    engine = transcript_engine()
    events = []
    engine.on_event = events.append
    for text in ('I think', 'I think we should', 'I think we should leave'):
        engine.observe_transcript(text, 'utterance-1')
        assert list(engine._window) == [('user', text)]
    assert [event.text_delta for event in events] == ['I think', 'I think we should', 'I think we should leave']
    engine.observe_transcript('Another message', 'utterance-2')
    assert list(engine._window) == [('user', 'I think we should leave'), ('user', 'Another message')]


def test_partial_replaces_trimmed_snapshot_without_duplicate_context():
    engine = transcript_engine(context_tokens=24)
    engine.observe_transcript('A long transcript that exceeds the context budget', 'utterance-1')
    engine.observe_transcript('Corrected words', 'utterance-1')
    assert list(engine._window) == [('user', 'Corrected words')]


def test_worker_passes_cumulative_transcript_without_appending():
    completed = threading.Event()
    calls = []
    def observe(text, utterance_id, **kwargs):
        calls.append((text, utterance_id))
        completed.set()
    worker = EmotionWorker(SimpleNamespace(observe_transcript=observe))
    try:
        for text in ('I think', 'I think we should', 'I think we should leave'):
            completed.clear()
            worker.transcript(text, 'utterance-1')
            assert completed.wait(1)
        assert calls == [(text, 'utterance-1') for text in ('I think', 'I think we should', 'I think we should leave')]
    finally:
        worker.close()
        worker.thread.join(1)


def test_user_input_is_live_without_foreground_deferral():
    completed=threading.Event()
    engine=SimpleNamespace(observe_input=lambda *args,**kwargs:completed.set())
    worker=EmotionWorker(engine)
    try:
        worker.submit('input','test',final=True)
        assert completed.wait(1)
    finally: worker.close(); worker.thread.join(1)
