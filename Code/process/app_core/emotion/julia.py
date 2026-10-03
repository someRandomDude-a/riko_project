from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Callable

from .models import EMOTIONS, EmotionEvent, EmotionState

logger = logging.getLogger(__name__)


class JuliaEmotionEngine:
    """Iterative emotional interpreter for Julia 1; CPU by default, CUDA opt-in.

    The engine stores a rolling transcript made from user and assistant deltas.
    It evaluates that window after meaningful deltas rather than retaining the
    whole conversation or waiting for a completed turn.
    """

    def __init__(self, model_path: str | Path | None, *, model_id: str = "SupersonicLabs/Julia-1",
                 cache_dir: str | Path | None = None,
                 device: str = "cpu", strict_encoding: bool = True, max_length: int = 8192,
                 context_tokens: int = 1024,
                 update_interval_tokens: int = 16, temperature: float = 0.2,
                 on_event: Callable[[EmotionEvent], None] | None = None,
                 fallback: bool = True):
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.model_id = model_id
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.device = device
        if device not in {'cpu', 'cuda', 'cuda:0'}: raise ValueError('Unsupported Julia device')
        self.strict_encoding = strict_encoding
        self.max_length = max_length
        self.context_tokens = max(128, context_tokens)
        self.update_interval_tokens = max(1, update_interval_tokens)
        self.temperature = temperature
        self.on_event = on_event
        self.fallback = fallback
        self.turn_id = str(uuid.uuid4())
        self._window: deque[tuple[str, str]] = deque()
        self._pending_tokens = 0
        self._model = None
        self._load_attempted = False
        self._lock = threading.RLock()
        self.state = EmotionState(turn_id=self.turn_id)

    def _load_model(self):
        if self._model is not None or self._load_attempted:
            return self._model
        self._load_attempted = True
        try:
            from importlib import import_module
            if self.model_path and self.model_path.exists():
                source = str(self.model_path)
            else:
                # Download once to the standard Hugging Face cache. No clone,
                # checked-in model directory, or manual model installation is needed.
                from huggingface_hub import snapshot_download
                source = snapshot_download(self.model_id, cache_dir=str(self.cache_dir) if self.cache_dir else None)
            # Julia's native runtime is shipped with the model repository. Add
            # the cached snapshot to import search rather than installing it.
            import sys
            if source not in sys.path:
                sys.path.insert(0, source)
            julia = import_module("julia")
            self._resolved_source = source
            self._model = julia.load_model(source, device=self.device, strict_encoding=self.strict_encoding,
                                           max_length=self.max_length,
                                            head_length=max(128, self.max_length - self.context_tokens))
            from .compat import compatible_engine
            compatible_engine(self._model)
        except Exception as exc:
            if not self.fallback:
                raise RuntimeError(f"Unable to load Julia 1 emotion model: {exc}") from exc
            logger.warning("Julia 1 model unavailable; using fallback emotion interpreter: %s", exc)
        return self._model

    def start_turn(self, turn_id: str | None = None) -> None:
        with self._lock:
            self.turn_id = turn_id or str(uuid.uuid4())
            self.state = EmotionState(turn_id=self.turn_id)
            self._window.clear()
            self._pending_tokens = 0

    def observe_input(self, delta: str, *, final: bool = False) -> EmotionState:
        return self._observe("user", delta, final=final)

    def observe_output(self, delta: str, *, final: bool = False) -> EmotionState:
        return self._observe("assistant", delta, final=final)

    def _observe(self, stream: str, delta: str, *, final: bool) -> EmotionState:
        if not delta:
            return self.state
        with self._lock:
            self._window.append((stream, delta))
            self._pending_tokens += self._token_count(delta)
            self._trim_window()
            if final or self._pending_tokens >= self.update_interval_tokens:
                self._pending_tokens = 0
                self.state = self._interpret(stream, delta)
                event = EmotionEvent(stream, delta, self.state)
                if self.on_event:
                    self.on_event(event)
            return self.state

    def _token_count(self, text: str) -> int:
        tokenizer = getattr(self._model, 'tokenizer', None)
        return len(tokenizer(text, add_special_tokens=False)['input_ids']) if tokenizer else len(text.encode('utf-8'))

    def _trim_window(self) -> None:
        budget = self.context_tokens
        if self._model is not None:
            tokenizer = getattr(self._model, 'tokenizer', None)
            if tokenizer:
                # Julia is a decision encoder, not autoregressive KV. Reserve its
                # actual serialized question/options + CLS/SEP/marker overhead.
                reserves = []
                for question in self._questions().values():
                    criteria = question['criteria']
                    labels = list(criteria.values()) if isinstance(criteria, dict) else criteria
                    reserves.append(self._token_count(question['type'] + ' question: ' + question['instructions'])
                        + sum(self._token_count(' ' + label) + 1 for label in labels) + 4)
                budget = min(budget, getattr(self._model, 'max_length', self.max_length) - max(reserves))
        if budget <= 0: raise ValueError('Julia questions leave no state context budget')
        while len(self._window) > 1 and self._token_count(self._transcript()) > budget:
            self._window.popleft()
        if self._window and self._token_count(self._transcript()) > budget:
            stream, text = self._window[-1]
            low, high = 0, len(text)
            while low < high:
                middle = (low + high) // 2
                if self._token_count(f'{stream}: {text[middle:]}') <= budget: high = middle
                else: low = middle + 1
            self._window[-1] = (stream, text[high:])

    def _transcript(self) -> str:
        return "\n".join(f"{stream}: {text}" for stream, text in self._window)

    def _interpret(self, stream: str, delta: str) -> EmotionState:
        model = self._load_model()
        if model is None:
            return self._fallback_state(stream, self._transcript())
        try:
            self._trim_window()
            response = model.predict(state=self._transcript(), questions=self._questions())
            return self._parse_julia_result(response, stream)
        except Exception as exc:
            logger.warning("Julia 1 interpretation failed: %s", exc)
            return self._fallback_state(stream, self._transcript())

    def choose_motion(self, state, candidates):
        """Reuse this model/lock; animation selection cannot load another Julia copy."""
        if self._model is None or not self._lock.acquire(timeout=.01): return None
        try:
            import json
            criteria = {item['id']: item['description'] for item in candidates}
            response = self._model.predict(state=json.dumps(state, ensure_ascii=False), questions={
                'intent': {'type': 'choice', 'instructions': 'Choose the most appropriate expressive avatar motion from these eligible intents. Respect actual runtime state.', 'criteria': criteria}})
            answers = response.get('answers', response) if isinstance(response, dict) else {}
            result = answers.get('intent', {})
            choice = result.get('choice')
            probabilities = result.get('probabilities', {})
            confidence = probabilities.get(choice, result.get('max_probability', 0))
            return {'choice': choice, 'confidence': self._number(confidence)} if choice in criteria else None
        finally: self._lock.release()

    def label_probe(self, transcript: str):
        """Stateless genuine-Julia supervision; never label with heuristics.

        Shares the existing model and lock, but does not modify the rolling
        transcript, emit an event, or replace the visible emotion state.
        """
        with self._lock:
            model = self._load_model()
            if model is None: return None
            budget = min(self.context_tokens, self.max_length // 2)
            while transcript and self._token_count(transcript) > budget:
                transcript = transcript[max(1, len(transcript) // 8):]
            try:
                result = model.predict(state=transcript, questions=self._questions())
                answers = result.get('answers', result) if isinstance(result, dict) else {}
                if not {'emotion', 'intensity', 'valence'}.issubset(answers): return None
                choice = answers['emotion'].get('choice')
                if choice not in {*EMOTIONS, 'love'}: return None
                import math
                for key in ('intensity', 'valence'):
                    score = answers[key].get('score')
                    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 4: return None
                return self._parse_julia_result(result, 'assistant')
            except Exception:
                logger.warning('Julia probe supervision failed', exc_info=True)
                return None

    def probe_fingerprint(self):
        """Detect local weight, tokenizer or runtime changes at the same path."""
        import hashlib
        source = getattr(self, '_resolved_source', None)
        if not source: return None
        root = Path(source)
        files = [root] if root.is_file() else sorted(path for path in root.rglob('*')
            if path.is_file() and path.suffix in {'.safetensors', '.bin', '.pt', '.pth', '.json', '.py', '.model', '.txt'}
            and not {'.git', '__pycache__', '.cache'}.intersection(path.relative_to(root).parts))
        digest = hashlib.sha256()
        for path in files:
            digest.update((path.name if root.is_file() else str(path.relative_to(root))).encode())
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
        return digest.hexdigest()

    def choose_tool_input(self, tool, parameter, value, candidates):
        """Use the existing Julia instance, with an explicit abstain option."""
        if self._model is None or not self._lock.acquire(timeout=.01): return None
        try:
            import json
            criteria = {item['id']:item['description'] for item in candidates}
            criteria['reject'] = 'None fits, ambiguous, unsafe, or user intent is unclear. Do not guess.'
            response = self._model.predict(state=json.dumps({'tool':tool,'parameter':parameter,'requested':value}), questions={
                'selection':{'type':'choice','instructions':'Correct this finite-choice input only if a listed option clearly expresses the same intent. Choose reject otherwise. Never grant permissions or invent paths.', 'criteria':criteria}})
            answers = response.get('answers',response) if isinstance(response,dict) else {}
            result = answers.get('selection',{})
            choice = result.get('choice')
            confidence = result.get('probabilities',{}).get(choice,result.get('max_probability',0))
            return {'choice':choice,'confidence':confidence} if choice in criteria and choice != 'reject' else None
        finally: self._lock.release()

    def _questions(self):
        return {
            "emotion": {"type": "choice", "instructions": "What emotion best describes the current interaction?", "criteria": {
                "neutral": "No strong emotional signal", "joy": "Happiness or delight", "amusement": "Playfulness or humor", "love": "Affection or warmth", "sadness": "Sorrow or low mood", "anger": "Irritation or frustration", "fear": "Anxiety or threat", "surprise": "Unexpectedness or shock", "confusion": "Uncertainty or ambiguity", "embarrassment": "Awkwardness or shame", "calm": "Peaceful or relaxed", "excitement": "High-energy enthusiasm"}},
            "intensity": {"type": "score", "instructions": "How emotionally intense is the interaction?", "criteria": ["None", "Mild", "Moderate", "Strong", "Extreme"]},
            "valence": {"type": "score", "instructions": "How positive or negative is the emotional tone?", "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]},
        }

    def _parse_julia_result(self, response, stream: str) -> EmotionState:
        answers = response.get("answers", response) if isinstance(response, dict) else {}
        emotion = answers.get("emotion", {})
        primary = emotion.get("choice", "neutral")
        if primary == 'love': primary = 'affection'
        if primary not in EMOTIONS: primary = "neutral"
        intensity = answers.get("intensity", {}).get("score", 0)
        valence = answers.get("valence", {}).get("score", 2)
        probabilities = emotion.get("probabilities", {})
        confidence = max(probabilities.values()) if probabilities else emotion.get("max_probability", 0.0)
        return EmotionState(
            primary=primary, intensity=self._score(intensity), valence=self._score(valence, centered=True),
            arousal=self._score(intensity), confidence=self._number(confidence),
            evidence=f"Julia 1 decision from {stream} stream",
            source="julia_1", turn_id=self.turn_id,
        )

    @staticmethod
    def _score(value, centered=False):
        try:
            normalized = max(0.0, min(1.0, float(value) / 4.0))
            return normalized * 2 - 1 if centered else normalized
        except (TypeError, ValueError): return 0.0

    @staticmethod
    def _number(value, *, low=0.0, high=1.0) -> float:
        try: return max(low, min(high, float(value)))
        except (TypeError, ValueError): return 0.0

    def _fallback_state(self, stream: str, text: str) -> EmotionState:
        lower = text.lower()
        primary, valence, arousal = "neutral", 0.0, 0.0
        if any(word in lower for word in ("haha", "lol", "love", "yay", "great")):
            primary, valence, arousal = "joy", 0.8, 0.6
        elif any(word in lower for word in ("sad", "sorry", "miss", "cry")):
            primary, valence, arousal = "sadness", -0.7, -0.2
        elif any(word in lower for word in ("angry", "hate", "damn", "ugh")):
            primary, valence, arousal = "anger", -0.7, 0.8
        elif any(word in lower for word in ("?", "what", "really")):
            primary, valence, arousal = "surprise", 0.1, 0.6
        return EmotionState(primary=primary, intensity=min(1.0, abs(valence)), valence=valence,
                            arousal=arousal, confidence=0.25, evidence=f"fallback:{stream}",
                            source="julia_1_fallback", turn_id=self.turn_id)

    def close(self) -> None:
        with self._lock:
            self._model = None
