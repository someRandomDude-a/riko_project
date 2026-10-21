"""The one faster-whisper factory, and the session's one model built with it (AsrService). auto/default resolve to what
this CTranslate2 build can run here (it has no Metal backend, and its CPU path rejects float16), and a configured pair
it cannot run falls back once, with one warning, instead of failing every utterance."""
from functools import lru_cache
import logging
import threading

from ..kernel.audio_config import SAMPLE_RATE
from ..kernel.schema import Section, Setting, register

logger = logging.getLogger(__name__)
MODEL = 'distil-small.en'
# Fastest first, matching CTranslate2's own 'auto'. CPU 'int8' runs as int8_float32.
PREFERRED = {'cuda': ('int8_float16', 'int8_float32', 'float16', 'float32'), 'cpu': ('int8', 'float32')}
_warned = set()


@lru_cache(maxsize=1)
def supported_types():
    """{device: compute types} that this CTranslate2 build supports; 'cuda' only with a visible device."""
    try: import ctranslate2
    except ImportError: return {'cpu': frozenset({'float32'})}
    types = {'cpu': frozenset(ctranslate2.get_supported_compute_types('cpu'))}
    try:
        if ctranslate2.get_cuda_device_count() > 0: types['cuda'] = frozenset(ctranslate2.get_supported_compute_types('cuda'))
    except Exception as exc: logger.warning('CTranslate2 CUDA probe failed; speech recognition stays on the CPU: %s', exc)
    return types


def resolve(voice):
    """(device, compute_type, note) to construct with; note says why a configured pair was replaced, else ''."""
    wanted = str(voice.get('asr_device') or 'auto'), str(voice.get('asr_compute_type') or 'default')
    supported, (device, precision), notes = supported_types(), wanted, []
    if device == 'auto': device = 'cuda' if 'cuda' in supported else 'cpu'
    if device not in supported:
        notes.append('no CUDA device is visible to CTranslate2' if device == 'cuda' else f'unknown device {device}')
        device = 'cpu'
    best = next((name for name in PREFERRED[device] if name in supported[device]), 'float32')
    if precision in ('default', 'auto'): precision = best
    elif precision not in supported[device]: notes.append(f'{device} does not support {precision}'); precision = best
    return device, precision, '; '.join(notes)


def review_pair(candidate, draft, changes):
    """Settings' check of the speech recognition pair, only when it is edited: a pair this machine cannot run falls back at
    startup, so one already in the file never blocks saving other settings. It names the field whose value cannot run
    here, which may be the one the user did not edit."""
    if not {'voice.asr_device', 'voice.asr_compute_type'} & changes.keys(): return {}
    voice, errors = candidate.raw.get('voice') or {}, {}
    device, _, note = resolve(voice)
    for part in filter(None, note.split('; ')):
        if ' does not support ' in part: errors['voice.asr_compute_type'] = f'{device} cannot run {voice.get("asr_compute_type")} here; choose default'
        else: errors['voice.asr_device'] = f'Not available on this machine ({part}); choose auto or cpu'
    return errors


ASR = dict(group='models', section='Speech recognition model')
register(Section('voice', review=review_pair, settings=(  # the rest of voice is kernel/audio_config.py's
    Setting('asr_model', MODEL, label='Speech recognition model', **ASR),
    Setting('asr_device', 'auto', options=('auto', 'cuda', 'cpu'), label='Speech recognition device', **ASR,
        help='auto uses CUDA when CTranslate2 sees a supported GPU (NVIDIA, or AMD with a HIP-built CTranslate2), otherwise the CPU. CTranslate2 has no Metal backend, so Macs transcribe on the CPU. Packaged builds bundle no CUDA libraries for speech recognition; keep cpu there. Save and restart Python.'),
    Setting('asr_compute_type', 'default', options=('default', 'int8_float16', 'float16', 'int8', 'float32'), label='Speech recognition precision', **ASR,
        help='default picks the fastest precision the device supports: int8_float16 on recent NVIDIA GPUs, int8 on the CPU. float16 and int8_float16 need a GPU. A pair this machine cannot run is refused here; one already in the file is replaced at startup with a logged warning. Save and restart Python.'))))


def create_whisper(voice):
    from faster_whisper import WhisperModel
    device, precision, note = resolve(voice)
    if note and note not in _warned:
        _warned.add(note)
        logger.warning('Speech recognition %s/%s cannot run here (%s); using %s/%s. Choose auto and default in Settings.',
            voice.get('asr_device', 'auto'), voice.get('asr_compute_type', 'default'), note, device, precision)
    return WhisperModel(voice.get('asr_model', MODEL), device=device, compute_type=precision)


class AsrService:
    """The session's one Whisper model (SessionManager.asr), shared by the microphone (VoiceInput), startup warmup and
    Discord's /transcribe: built through create_whisper on first use, kept until close(), one decode at a time."""
    def __init__(self, voice, *, model=None):
        self.voice, self.model, self.lock, self.closed = voice, model, threading.Lock(), False

    def transcribe(self, samples, **options):
        """The text of float32 16 kHz mono samples; options go to WhisperModel.transcribe."""
        with self.lock:
            if self.closed: raise RuntimeError('Speech recognition is closed')
            model = self.model
            if model is None:  # a caller that arrives meanwhile waits for this one instead of building another
                model = create_whisper(self.voice)
                if not self.closed: self.model = model  # one finished after close() is not kept
            segments, _ = model.transcribe(samples, **options)
            return ' '.join(segment.text.strip() for segment in segments).strip()  # decoding runs while iterating

    def transcribe_pcm(self, pcm, **options):
        """The text of mono 16 kHz signed little-endian PCM16."""
        import numpy as np
        return self.transcribe(np.frombuffer(pcm, dtype='<i2').astype('float32') / 32768, **options)

    def warm(self):
        import numpy as np
        self.transcribe(np.zeros(SAMPLE_RATE, dtype='float32'), beam_size=1, vad_filter=False)  # one second of silence

    def close(self):  # never waits for a decode in progress
        self.closed, self.model = True, None
