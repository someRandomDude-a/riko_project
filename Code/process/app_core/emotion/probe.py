"""Read-only latent emotion distillation. Optional torch imports stay local.

Artifacts contain activations and labels, not text. They are still private.
Training is per exact backbone/tokenizer/teacher/feature identity. Held-out
groups are whole generation sessions, never random correlated token samples.
"""
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
import logging
import math
from pathlib import Path
import threading
import uuid

from .models import EMOTIONS, EmotionState

logger = logging.getLogger(__name__)
FEATURE_VERSION = 'result_norm-pool256-v1'


@dataclass
class ProbeConfig:
    enabled: bool = False
    use_for_expression: bool = True
    hidden_units: tuple = (16384, 8192)
    rank: int = 32
    interval_tokens: int = 32
    min_samples: int = 256
    retrain_every: int = 128
    max_samples: int = 4096
    epochs: int = 12
    min_agreement: float = .8
    min_macro_f1: float = .65
    max_rmse: float = .2
    min_confidence: float = .6

    @classmethod
    def from_raw(cls, raw):
        if not isinstance(raw, dict): raise ValueError('emotion.probe must be an object')
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown: raise ValueError(f'Unknown emotion.probe keys: {sorted(unknown)}')
        value = cls(**raw)
        if type(value.interval_tokens) is not int or not 1 <= value.interval_tokens <= 512:
            raise ValueError('probe.interval_tokens must be an integer between 1 and 512')
        for key in ('enabled', 'use_for_expression'):
            if type(getattr(value, key)) is not bool: raise ValueError(f'probe.{key} must be boolean')
        for key in ('rank', 'min_samples', 'retrain_every', 'max_samples', 'epochs'):
            n = getattr(value, key)
            if type(n) is not int or not 1 <= n <= 65536: raise ValueError(f'Invalid probe.{key}')
        if value.min_samples < 32 or value.max_samples < value.min_samples: raise ValueError('Invalid probe sample budget')
        if not isinstance(value.hidden_units, (list, tuple)) or len(value.hidden_units) != 2 or any(type(n) is not int or not 8 <= n <= 32768 for n in value.hidden_units):
            raise ValueError('probe.hidden_units must contain two widths between 8 and 32768')
        for key in ('min_agreement', 'min_macro_f1', 'max_rmse', 'min_confidence'):
            n = getattr(value, key)
            if type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1: raise ValueError(f'Invalid probe.{key}')
        return value


def identity_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def held_out(group):
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0


def build_network(config):
    import torch
    import torch.nn as nn
    a, b = config.hidden_units
    # Low-rank connections make 24,576 hidden units affordable on CPU rather
    # than allocating a dense 16,384 x 8,192 matrix.
    with torch.device('cpu'):
        return nn.Sequential(nn.LayerNorm(256), nn.Linear(256, config.rank),
            nn.Linear(config.rank, a), nn.GELU(), nn.Linear(a, config.rank),
            nn.Linear(config.rank, b), nn.GELU(), nn.Linear(b, len(EMOTIONS) + 3))


def latent_features(hidden):
    import torch
    value = hidden.detach().float().reshape(-1, hidden.shape[-1])[-1]
    value = torch.nn.functional.adaptive_avg_pool1d(value[None, None], 256).flatten().cpu()
    if not torch.isfinite(value).all(): raise ValueError('Nonfinite probe activations')
    return value


def metrics(logits, target):
    import torch
    prediction, labels = logits[:, :len(EMOTIONS)].argmax(-1), target[:, 0].long()
    agreement = float((prediction == labels).float().mean())
    classes = labels.unique().tolist()
    f1 = []
    for label in classes:
        tp = ((prediction == label) & (labels == label)).sum().item()
        fp = ((prediction == label) & (labels != label)).sum().item()
        fn = ((prediction != label) & (labels == label)).sum().item()
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    values = decode_scores(logits)
    return {'agreement': agreement, 'macro_f1': sum(f1) / len(f1),
        'rmse': float(torch.sqrt(((values - target[:, 1:]) ** 2).mean())),
        'classes': len(classes), 'validation_samples': len(target)}


def decode_scores(logits):
    import torch
    raw = logits[:, len(EMOTIONS):]
    return torch.stack((raw[:, 0].sigmoid(), raw[:, 1].tanh(), raw[:, 2].sigmoid()), dim=1)


def qualified(result, config):
    return (result['validation_samples'] >= 16 and result['classes'] >= 3
        and result['agreement'] >= config.min_agreement
        and result['macro_f1'] >= config.min_macro_f1 and result['rmse'] <= config.max_rmse)


class EmotionProbe:
    def __init__(self, directory, identity, teacher, config, *, idle=lambda: True, on_prediction=None, on_fallback=None):
        import torch
        self.config, self.teacher, self.idle = config, teacher, idle
        self.on_prediction = on_prediction
        self.on_fallback = on_fallback
        with teacher._lock: teacher._load_model()
        self.identity = {**identity, 'feature_version': FEATURE_VERSION,
            'teacher': teacher.model_id, 'teacher_path': str(teacher.model_path),
            'teacher_snapshot': getattr(teacher, '_resolved_source', None),
            'teacher_questions': teacher._questions(), 'teacher_context': teacher.context_tokens,
            'teacher_max_length': teacher.max_length, 'teacher_strict_encoding': teacher.strict_encoding,
            'teacher_fingerprint': teacher.probe_fingerprint(),
            'network': [*config.hidden_units, config.rank]}
        self.key = identity_key(self.identity)
        self.path = Path(directory) / self.key / 'probe.pt'
        self.condition = threading.Condition()
        self.pending = deque(maxlen=32)
        self.samples = deque(maxlen=config.max_samples)
        self.network, self.validation = None, {}
        self.prediction_turn_id = None
        self.active_group = None
        self.closed, self.training, self.since_train = False, False, 0
        self._restore()
        self.thread = threading.Thread(target=self._run, name='emotion-probe', daemon=True)
        self.thread.start()

    @property
    def ready(self): return self.network is not None

    def activate(self, group):
        with self.condition:
            self.active_group = group
            self.prediction_turn_id = None
            self.pending.clear()

    def status(self):
        with self.condition:
            return {'model_key': self.key, 'ready': self.ready, 'training': self.training,
                'samples': len(self.samples), 'pending': len(self.pending), 'validation': dict(self.validation)}

    def capture(self, hidden, transcript, group, *, cancelled=lambda: False):
        if cancelled() or self.closed: return None
        features = latent_features(hidden)
        with self.condition:
            if cancelled() or self.closed: return None
            self.pending.append((features, transcript, group, cancelled))
            self.condition.notify_all()
        return None

    def _predict(self, features, group, cancelled):
        import torch
        network = self.network
        if network is None or not self.config.use_for_expression: return None
        with torch.inference_mode():
            logits = network(features[None])
            if not torch.isfinite(logits).all(): return None
            probabilities = logits[0, :len(EMOTIONS)].softmax(-1)
            confidence, label = probabilities.max(0)
            if float(confidence) < self.config.min_confidence: return None
            intensity, valence, arousal = decode_scores(logits)[0].tolist()
        if cancelled(): return None
        return EmotionState(primary=EMOTIONS[int(label)], intensity=intensity, valence=valence,
            arousal=arousal, confidence=float(confidence), source='latent_probe', turn_id=group,
            evidence='Julia-supervised read-only hidden-state probe')

    def publish_teacher(self, event, publish):
        with self.condition:
            if self.closed: return
            if event.stream == 'assistant':
                if self.active_group is not None and event.state.turn_id != self.active_group: return
                if self.prediction_turn_id == event.state.turn_id: return
            publish(event.state)

    def _publish(self, state, group, cancelled, *, fallback=False):
        with self.condition:
            if self.closed or cancelled() or self.active_group != group: return
            self.prediction_turn_id = None if fallback or state is None else group
            callback = self.on_fallback if fallback else self.on_prediction
            if state is not None and callback: callback(replace(state, turn_id=group))

    def _restore(self):
        if not self.path.exists(): return
        try:
            import torch
            saved = torch.load(self.path, map_location='cpu', weights_only=True)
            if saved['identity'] != self.identity: return
            self.samples.extend(saved['samples'][-self.config.max_samples:])
            self.since_train = min(len(self.samples), saved.get('since_train', len(self.samples)))
            self.validation = saved['validation']
            if saved.get('weights') is not None and qualified(self.validation, self.config):
                network = build_network(self.config)
                network.load_state_dict(saved['weights'])
                self.network = network.eval().requires_grad_(False)
        except Exception:
            logger.warning('Ignoring incompatible emotion probe artifact', exc_info=True)

    def _run(self):
        try: self._work()
        finally:
            try: self._save(list(self.samples))
            except Exception: logger.exception('Unable to save emotion probe on shutdown')

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending, timeout=1)
                if self.closed: return
                sample = self.pending.popleft() if self.pending else None
            if sample is None:
                if len(self.samples) >= self.config.min_samples and self.since_train >= self.config.retrain_every and self.idle():
                    try: self._train()
                    except Exception: logger.exception('Emotion probe training failed')
                continue
            features, transcript, group, cancelled = sample
            if cancelled(): continue
            try:
                prediction = self._predict(features, group, cancelled)
                self._publish(prediction, group, cancelled)
                label = self.teacher.label_probe(transcript)
                if label is None or label.source != 'julia_1' or cancelled(): continue
                if prediction is None: self._publish(label, group, cancelled, fallback=True)
                target = [EMOTIONS.index(label.primary), label.intensity, label.valence, label.arousal]
                if not all(math.isfinite(n) for n in target): continue
                with self.condition:
                    self.samples.append((features, target, group))
                    self.since_train += 1
                if self.since_train % 16 == 0: self._save(list(self.samples))
                # Foreground collection is lightweight; train only while idle.
                if len(self.samples) >= self.config.min_samples and self.since_train >= self.config.retrain_every and self.idle():
                    self._train()
            except Exception:
                self._publish(None, group, cancelled)
                logger.exception('Emotion probe supervision/training failed; retaining Julia fallback')

    def _train(self):
        import torch
        with self.condition:
            data = list(self.samples)
            self.training = True
        try:
            train = [row for row in data if not held_out(row[2])]
            valid = [row for row in data if held_out(row[2])]
            if len(train) < 16 or len(valid) < 16: return
            network = build_network(self.config)
            optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
            x = torch.stack([row[0] for row in train])
            y = torch.tensor([row[1] for row in train], dtype=torch.float32, device='cpu')
            for _ in range(self.config.epochs):
                for batch in torch.randperm(len(train), device='cpu').split(32):
                    if self.closed or not self.idle(): return
                    logits = network(x[batch])
                    loss = torch.nn.functional.cross_entropy(logits[:, :len(EMOTIONS)], y[batch, 0].long())
                    loss += torch.nn.functional.mse_loss(decode_scores(logits), y[batch, 1:])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1)
                    optimizer.step()
            network.eval().requires_grad_(False)
            vx = torch.stack([row[0] for row in valid])
            vy = torch.tensor([row[1] for row in valid], dtype=torch.float32, device='cpu')
            with torch.inference_mode(): result = metrics(network(vx), vy)
            incumbent = None
            if self.network is not None:
                with torch.inference_mode():
                    incumbent = metrics(self.network(vx), vy)
            with self.condition:
                if qualified(result, self.config) and (incumbent is None or result['macro_f1'] >= incumbent['macro_f1']):
                    self.network = network
                    self.validation = result
                elif self.network is None: self.validation = result
                self.since_train = 0
            self._save(data)
        finally:
            with self.condition: self.training = False

    def _save(self, samples):
        import torch
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        torch.save({'identity': self.identity, 'samples': samples, 'validation': self.validation, 'since_train': self.since_train,
            'weights': {key: value.cpu() for key, value in self.network.state_dict().items()} if self.network is not None else None}, temporary)
        temporary.replace(self.path)

    def close(self):
        with self.condition:
            if self.closed: return
            self.closed = True
            self.pending.clear()
            self.condition.notify_all()
        self.thread.join(timeout=2)
