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
import time

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
    auto_train: bool = True
    idle_seconds: int = 300
    retain_sample_text: bool = True

    @classmethod
    def from_raw(cls, raw):
        if not isinstance(raw, dict): raise ValueError('emotion.probe must be an object')
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown: raise ValueError(f'Unknown emotion.probe keys: {sorted(unknown)}')
        value = cls(**raw)
        if type(value.interval_tokens) is not int or not 1 <= value.interval_tokens <= 512:
            raise ValueError('probe.interval_tokens must be an integer between 1 and 512')
        for key in ('enabled', 'use_for_expression', 'auto_train', 'retain_sample_text'):
            if type(getattr(value, key)) is not bool: raise ValueError(f'probe.{key} must be boolean')
        for key in ('rank', 'min_samples', 'retrain_every', 'max_samples', 'epochs'):
            n = getattr(value, key)
            if type(n) is not int or not 1 <= n <= 65536: raise ValueError(f'Invalid probe.{key}')
        if value.min_samples < 32 or value.max_samples < value.min_samples: raise ValueError('Invalid probe sample budget')
        if type(value.idle_seconds) is not int or not 0 <= value.idle_seconds <= 86400: raise ValueError('Invalid probe.idle_seconds')
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
    def __init__(self, directory, identity, teacher, config, *, idle=lambda: True, on_prediction=None, on_fallback=None, legacy_directory=None, training_directory=None):
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
        self.data_path = Path(training_directory) / self.key / 'training.pt' if training_directory else self.path
        self.legacy_path = Path(legacy_directory) / self.key / 'probe.pt' if legacy_directory else None
        self.condition = threading.Condition()
        self.pending = deque(maxlen=32)
        self.samples = deque(maxlen=config.max_samples)
        self.network, self.validation = None, {}
        self.prediction_turn_id = None
        self.active_group = None
        self.closed, self.training, self.since_train = False, False, 0
        self.idle_since = None
        self.manual_training = False
        self.data_revision = str(uuid.uuid4())
        self.metadata = {}
        self.save_lock = threading.Lock()
        self.error = ''
        self.segment_features=deque(maxlen=512)
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
            self.segment_features.clear()

    def status(self):
        with self.condition:
            return {'model_key': self.key, 'ready': self.ready, 'training': self.training,
                'samples': len(self.samples), 'pending': len(self.pending), 'validation': dict(self.validation),
                'mode': 'probe' if self.ready and self.config.use_for_expression else 'julia_collecting',
                'manual_training': self.manual_training, 'idle_seconds': self.config.idle_seconds,
                'idle_elapsed': max(0,time.monotonic()-self.idle_since) if self.idle_since is not None else 0,
                'revision': self.data_revision, 'error': self.error, 'emotions': list(EMOTIONS)}

    def request_training(self):
        with self.condition:
            if self.training or self.manual_training: raise ValueError('Training is already running or queued')
            if len(self.samples)<self.config.min_samples: raise ValueError('Collect more samples before training')
            self.manual_training = True
            self.condition.notify_all()
            return self.status()

    def training_due(self):
        if not self.idle(): self.idle_since=None; return False
        now=time.monotonic()
        if self.idle_since is None: self.idle_since=now
        return self.manual_training or (self.config.auto_train and now-self.idle_since>=self.config.idle_seconds
            and self.since_train>=self.config.retrain_every)

    @staticmethod
    def sample_id(row):
        return hashlib.sha256(row[2].encode()+row[0].detach().cpu().numpy().tobytes()).hexdigest()

    def data_page(self, offset=0,limit=40,group=None):
        with self.condition:
            rows=[row for row in self.samples if group is None or row[2]==group]
            return {'revision':self.data_revision,'total':len(rows),'emotions':list(EMOTIONS),'samples':[
                {'id':self.sample_id(row),'group':row[2],'held_out':held_out(row[2]),
                 'emotion':EMOTIONS[int(row[1][0])],'intensity':row[1][1],'valence':row[1][2],'arousal':row[1][3],
                 **self.metadata.get(self.sample_id(row),{})} for row in rows[offset:offset+limit]]}

    def data_groups(self,offset=0,limit=20):
        with self.condition:
            groups={}
            for row in self.samples:
                meta=self.metadata.get(self.sample_id(row),{})
                entry=groups.setdefault(row[2],{'id':row[2],'input':meta.get('input_text'),'samples':0,'held_out':held_out(row[2]),'collected_at':meta.get('collected_at')})
                entry['samples']+=1
            entries=list(groups.values())
            return {'groups':entries[offset:offset+limit],'total':len(entries),'revision':self.data_revision}

    def edit_sample(self, sample_id, values, revision):
        if not isinstance(values,dict) or set(values)!={'emotion','intensity','valence','arousal'}: raise ValueError('Edit only emotion and scores')
        if values['emotion'] not in EMOTIONS: raise ValueError('Unknown emotion')
        for key,low in (('intensity',0),('valence',-1),('arousal',0)):
            value=values[key]
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=1: raise ValueError('Invalid expression score')
        with self.condition:
            if revision!=self.data_revision: raise RuntimeError('Dataset changed; reload before editing')
            found=False
            for index,row in enumerate(self.samples):
                if self.sample_id(row)!=sample_id: continue
                target=[EMOTIONS.index(values['emotion']),values['intensity'],values['valence'],values['arousal']]
                self.samples[index]=(row[0],target,row[2]);found=True
            if not found: raise ValueError('Sample is no longer available')
            self.metadata.setdefault(sample_id,{})['edited']=True
            self.network=None;self.validation={};self.since_train=max(self.since_train,self.config.retrain_every)
            self.data_revision=str(uuid.uuid4())
            self._save(list(self.samples))
            return self.status()

    def capture(self, hidden, transcript, group, *, cancelled=lambda: False, replay=False,input_text=None,offset=0):
        if cancelled() or self.closed: return None
        features = latent_features(hidden)
        with self.condition:
            if cancelled() or self.closed: return None
            if not replay:self.segment_features.append((offset,features,group))
            self.pending.append((features, transcript, group, cancelled, replay,input_text))
            self.condition.notify_all()
        return None

    def expression_for_segment(self,end_offset):
        with self.condition:
            if not self.ready or not self.config.use_for_expression:return False
            eligible=[item for item in self.segment_features if item[0]<=end_offset]
            if not eligible:return False
            _,features,group=eligible[-1]
        state=self._predict(features,group,lambda:self.closed or self.active_group!=group)
        if state is None:return False
        if self.on_prediction:self.on_prediction(state)
        return True

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
        source = self.data_path if self.data_path.exists() else self.path if self.path.exists() else self.legacy_path
        if source is None or not source.exists(): return
        try:
            import torch
            saved = torch.load(source, map_location='cpu', weights_only=True)
            if saved['identity'] != self.identity: return
            self.samples.extend(saved['samples'][-self.config.max_samples:])
            self.since_train = min(len(self.samples), saved.get('since_train', len(self.samples)))
            self.validation = saved['validation']
            self.metadata = saved.get('metadata', {})
            if source == self.data_path and self.data_path != self.path and self.path.exists():
                artifact = torch.load(self.path, map_location='cpu', weights_only=True)
                if artifact.get('identity') == self.identity:
                    saved['weights'] = artifact.get('weights')
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
                if len(self.samples) >= self.config.min_samples and self.training_due():
                    try: self._train()
                    except Exception: logger.exception('Emotion probe training failed')
                continue
            features, transcript, group, cancelled, replay,input_text = sample
            if cancelled(): continue
            try:
                prediction = self._predict(features, group, cancelled)
                if not replay: self._publish(prediction, group, cancelled)
                if prediction is not None and self.ready and self.config.use_for_expression and not replay: continue
                label = self.teacher.label_probe(transcript)
                if label is None or label.source != 'julia_1' or cancelled(): continue
                if prediction is None and not replay: self._publish(label, group, cancelled, fallback=True)
                target = [EMOTIONS.index(label.primary), label.intensity, label.valence, label.arousal]
                if not all(math.isfinite(n) for n in target): continue
                with self.condition:
                    self.samples.append((features, target, group))
                    self.since_train += 1
                    row=(features,target,group)
                    self.metadata[self.sample_id(row)]={'teacher':{'emotion':label.primary,'intensity':label.intensity,'valence':label.valence,'arousal':label.arousal},
                        'prediction':{'emotion':prediction.primary,'confidence':prediction.confidence} if prediction else None,
                        'text':transcript[:2000] if self.config.retain_sample_text else None,
                        'input_text':input_text[:2000] if self.config.retain_sample_text and input_text else None,
                        'collected_at':time.time(),'edited':False}
                    self.data_revision=str(uuid.uuid4())
                    keep={self.sample_id(row) for row in self.samples}
                    self.metadata={key:value for key,value in self.metadata.items() if key in keep}
                if self.since_train % 16 == 0: self._save(list(self.samples))
                # Foreground collection is lightweight; train only while idle.
                if len(self.samples) >= self.config.min_samples and self.training_due():
                    self._train()
            except Exception:
                self._publish(None, group, cancelled)
                logger.exception('Emotion probe supervision/training failed; retaining Julia fallback')

    def _train(self):
        import torch
        with self.condition:
            data = list(self.samples)
            revision=self.data_revision
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
                if revision!=self.data_revision: return
                if qualified(result, self.config) and (incumbent is None or result['macro_f1'] >= incumbent['macro_f1']):
                    self.network = network
                    self.validation = result
                elif self.network is None: self.validation = result
                self.since_train = 0
            self._save(data)
        finally:
            with self.condition: self.training = False; self.manual_training=False

    def _save(self, samples):
        import torch
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        with self.save_lock:
            dataset = {'identity': self.identity, 'samples': samples, 'validation': self.validation, 'since_train': self.since_train,
                'metadata':self.metadata,
                'weights': {key: value.cpu() for key, value in self.network.state_dict().items()} if self.network is not None else None}
            if self.data_path != self.path:
                data_temporary = self.data_path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
                torch.save({key: value for key, value in dataset.items() if key != 'weights'}, data_temporary)
                data_temporary.replace(self.data_path)
                artifact = {key: value for key, value in dataset.items() if key not in {'samples', 'metadata', 'since_train'}}
            else:
                artifact = dataset
            torch.save(artifact, temporary)
            temporary.replace(self.path)
            examples=[{'id':self.sample_id(row),'text':self.metadata.get(self.sample_id(row),{}).get('text'),
                'emotion':EMOTIONS[int(row[1][0])],'scores':row[1][1:]} for row in samples]
            examples=[row for row in examples if row['text']]
            text_path=self.data_path.with_name('examples.json')
            temporary_text=text_path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            temporary_text.write_text(json.dumps({'model_key':self.key,'examples':examples},ensure_ascii=False),encoding='utf-8')
            temporary_text.replace(text_path)

    def close(self):
        with self.condition:
            if self.closed: return
            self.closed = True
            self.pending.clear()
            self.condition.notify_all()
        self.thread.join(timeout=2)
