from process.app_core.emotion import JuliaEmotionEngine
from types import SimpleNamespace


def test_julia_rolling_cpu_fallback_emits_state():
    events = []
    engine = JuliaEmotionEngine(None, context_tokens=128, update_interval_tokens=1, on_event=events.append)
    engine.start_turn("test-turn")
    engine.observe_input("This is great!", final=True)
    engine.observe_output("I am happy to help.", final=True)
    assert events
    assert events[-1].state.turn_id == "test-turn"
    assert events[-1].state.primary in {"neutral", "joy"}


def test_julia_window_is_bounded():
    engine = JuliaEmotionEngine(None, context_tokens=128, update_interval_tokens=1000)
    engine.observe_input("word " * 1000)
    assert engine._token_count(engine._transcript()) <= 140


def test_julia_window_uses_loaded_tokenizer_not_word_count():
    captured = []
    def tokenizer(text, **kwargs): return {'input_ids': list(text.encode('utf-8'))}
    def predict(*, state, questions): captured.append(state); return {}
    engine = JuliaEmotionEngine(None, context_tokens=128, update_interval_tokens=1000)
    engine._model = SimpleNamespace(tokenizer=tokenizer, max_length=8192, predict=predict)
    engine.observe_input('界' * 500, final=True)
    assert len(captured[0].encode('utf-8')) <= 128
    assert captured[0].endswith('界')
