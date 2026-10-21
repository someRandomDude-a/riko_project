"""The microphone's audio geometry and the typed voice, speech and sovits_ping_config sections. load_config builds each once
with from_raw (kernel/validation.py checks), so a bad value stops startup with its key named instead of failing every
turn or audio frame; the audio and runtime packages read the fields, not config.raw (RAW_KEYS excepted). Each section is
registered with the settings schema (kernel/schema.py) beside its Settings metadata; keys it does not declare are reported
(configuration/schema.py unknown_settings), never a startup failure."""
from dataclasses import dataclass, fields
from functools import partial
from typing import ClassVar

from .schema import Rule, Section, Setting, dataclass_defaults, register
from .validation import boolean, choice, number, section, text

SAMPLE_RATE = 16000  # Hz, mono PCM16: microphone capture, Silero VAD, the wake detector and Whisper
FRAME_SAMPLES = 512  # one capture block, one VAD window and one wake-detector step
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE  # 0.032
BARGE_IN = ('full_duplex', 'echo_gate', 'stop_only')  # voice.barge_in


def declared(cls):
    """The keys a section's YAML mapping may hold: its fields, and keys other modules read from config.raw."""
    return {item.name for item in fields(cls)} | cls.RAW_KEYS


def _build(cls, path, raw, checks):
    raw = section(path, raw)
    return cls(**{key: check(f'{path}.{key}', raw[key]) for key, check in checks.items() if key in raw})


def _device(path, value):  # sounddevice takes an index or a name; null is the system microphone
    return value if value is None or isinstance(value, str) else number(path, value, ge=0, integer=True)


@dataclass(slots=True, frozen=True)
class VoiceConfig:
    mode: str = 'wake_word'
    wake_word: object = None  # the name as YAML read it: wake_word.wake_phrase reports a number or boolean, never fails
    wake_threshold: float = .9  # a new enrollment's starting threshold; an enrolled profile keeps its own (WakeWord.bind_device)
    follow_up_seconds: float = 10.0
    input_device: int | str | None = None
    vad_threshold: float = .5
    pre_roll_seconds: float = 1.0
    transcription_gap_seconds: float = .3
    utterance_end_seconds: float = 1.0
    max_segment_seconds: float = 15.0
    live_transcript_interval_seconds: float = 2.0
    interruption_seconds: float = 1.5
    interjection_debounce_seconds: float = 1.0
    assertive_interruption_seconds: float = 6.0
    assertive_window_seconds: float = 15.0
    playback_words_per_second: float = 2.5
    # Speech while Riko's own voice plays (audio/echo_gate.py): full_duplex counts it all (her echo too, through speakers).
    barge_in: str = 'full_duplex'
    barge_in_vad_threshold: float = .85
    echo_coupling: float | None = None  # None: learned for each reply while she talks
    echo_tail_seconds: float = .3
    # Read from config.raw by audio/asr.py, which resolves them against this machine and falls back on its own.
    RAW_KEYS: ClassVar[frozenset] = frozenset({'asr_model', 'asr_device', 'asr_compute_type'})

    @classmethod
    def from_raw(cls, raw):
        positive = partial(number, gt=0)
        voice = _build(cls, 'voice', raw, {'mode': partial(choice, options=('wake_word', 'continuous', 'manual')),
            'wake_word': lambda path, value: value, 'wake_threshold': partial(number, gt=0, lt=1),
            'follow_up_seconds': partial(number, gt=0, le=300), 'input_device': _device, 'vad_threshold': partial(number, ge=0, le=1),
            'pre_roll_seconds': positive, 'transcription_gap_seconds': positive, 'utterance_end_seconds': positive,
            'max_segment_seconds': positive, 'live_transcript_interval_seconds': partial(number, ge=.5, le=10),
            'interruption_seconds': positive, 'interjection_debounce_seconds': positive,
            'assertive_interruption_seconds': partial(number, gt=0, le=30), 'assertive_window_seconds': partial(number, gt=0, le=60),
            'playback_words_per_second': positive, 'barge_in': partial(choice, options=BARGE_IN),
            'barge_in_vad_threshold': partial(number, ge=0, le=1),
            'echo_coupling': lambda path, value: None if value is None else number(path, value, gt=0, le=100),
            'echo_tail_seconds': partial(number, ge=0, le=2)})
        if GAP.broken({GAP.path: voice.transcription_gap_seconds, GAP.below: voice.utterance_end_seconds}): raise ValueError(GAP.message)
        return voice


GAP = Rule('voice.transcription_gap_seconds', 'voice.transcription_gap_seconds must be shorter than voice.utterance_end_seconds',
    below='voice.utterance_end_seconds')
WINDOW = Rule('speech.split_window_words', 'speech.split_window_words cannot exceed speech.max_words', at_most='speech.max_words')


def wake_phrase(voice, character_name):
    """The wake name and, when the detector cannot use it, why ('' otherwise). Without voice.wake_word it is the first
    word of the companion's name: the packaged setup accepts any name, and enrollment needs one short word. voice is a
    VoiceConfig, or the YAML mapping Settings edits (None when the section is absent)."""
    configured = voice.get('wake_word') if isinstance(voice, dict) else voice.wake_word if isinstance(voice, VoiceConfig) else None
    if configured is None: return (str(character_name or '').split() or ['Riko'])[0], ''
    if not isinstance(configured, str):  # PyYAML reads an unquoted 42 or yes as a number or boolean
        return str(configured), f'YAML read it as {type(configured).__name__}, not text: quote numbers and yes, no, on or off'
    phrase = configured.strip()
    return phrase, '' if len(phrase.split()) == 1 else 'Use one short wake name (a single word)'


def _split_priority(path, value):
    if not isinstance(value, list) or not value or any(not isinstance(group, str) or not group for group in value):
        raise ValueError(f'{path} must be a nonempty list of punctuation groups')
    characters = ''.join(value)
    if any(c not in '.!?;:,\n。！？；：，' for c in characters) or len(set(characters)) != len(characters):
        raise ValueError(f'{path} contains unsupported or repeated punctuation')
    return tuple(value)


@dataclass(slots=True, frozen=True)
class SpeechConfig:
    max_words: int = 40
    split_window_words: int = 15
    split_priority: tuple = ('.!?', ';:', ',', '\n')
    RAW_KEYS: ClassVar[frozenset] = frozenset()

    @classmethod
    def from_raw(cls, raw):
        speech = _build(cls, 'speech', raw, {'max_words': partial(number, ge=1, le=1000, integer=True),
            'split_window_words': partial(number, ge=0, le=1000, integer=True), 'split_priority': _split_priority})
        if WINDOW.broken({WINDOW.path: speech.split_window_words, WINDOW.at_most: speech.max_words}): raise ValueError(WINDOW.message)
        return speech


def _arguments(path, value):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value): raise ValueError(f'{path} must be a list of strings')
    return tuple(value)


@dataclass(slots=True, frozen=True)
class SovitsConfig:
    url: str = 'http://127.0.0.1:9880/tts'
    ref_audio_path: str = 'character_files/main_sample.wav'
    sample_rate: int = 32000  # must match the server's output
    max_in_flight_requests: int = 4
    streaming_mode: object = True
    auto_start: bool = False  # these three start the server in the packaged app (electron/release.cjs reads them)
    executable: str | None = None
    arguments: tuple = ()
    text_lang: str | None = None  # the rest are sent to GPT-SoVITS as they are, when set
    prompt_lang: str | None = None
    prompt_text: str | None = None
    batch_size: object = None
    text_split_method: object = None
    split_bucket: object = None
    parallel_infer: object = None
    fragment_interval: object = None
    speed_factor: object = None
    min_chunk_length: object = None
    overlap_length: object = None
    RAW_KEYS: ClassVar[frozenset] = frozenset()
    FORWARDED: ClassVar[tuple] = ('text_lang', 'prompt_lang', 'prompt_text', 'batch_size', 'text_split_method', 'split_bucket',
        'parallel_infer', 'fragment_interval', 'speed_factor', 'min_chunk_length', 'overlap_length')

    @classmethod
    def from_raw(cls, raw):
        optional, same = partial(text, optional=True), lambda path, value: value
        return _build(cls, 'sovits_ping_config', raw, {'url': text, 'ref_audio_path': text,
            'sample_rate': partial(number, ge=8000, le=192000, integer=True), 'max_in_flight_requests': partial(number, ge=1, integer=True),
            'streaming_mode': same, 'auto_start': boolean, 'executable': optional, 'arguments': _arguments,
            'text_lang': optional, 'prompt_lang': optional, 'prompt_text': optional,
            **{key: same for key in cls.FORWARDED if key not in {'text_lang', 'prompt_lang', 'prompt_text'}}})

    def request(self):
        """The GPT-SoVITS settings sent with every request: those set, as written."""
        return {key: getattr(self, key) for key in self.FORWARDED if getattr(self, key) is not None}


def audio_sections(raw):
    """{'voice', 'speech', 'sovits'}: the typed sections of a raw YAML mapping, as AppConfig holds them."""
    return {'voice': VoiceConfig.from_raw(raw.get('voice')), 'speech': SpeechConfig.from_raw(raw.get('speech')),
        'sovits': SovitsConfig.from_raw(raw.get('sovits_ping_config'))}


def _wake_word(candidate, draft, changes):  # Settings: the phrase the runtime will use, as PyYAML reads voice.wake_word
    problem = wake_phrase(candidate.voice, candidate.character_name)[1]
    return {'voice.wake_word': problem} if problem else {}


VOICE, SPEECH, SOVITS = dataclass_defaults(VoiceConfig), dataclass_defaults(SpeechConfig), dataclass_defaults(SovitsConfig)
LANGUAGES = ('en', 'zh', 'ja', 'ko', 'yue', 'auto')
# Settings offers most voice keys only once the YAML has them (listed=False), as before the schema existed. Each range is
# what VoiceConfig.from_raw accepts (None: no upper bound), so a value that loads can always be saved alongside others.
register(Section('voice', group='voice', title='Settings', strict=True, known=declared(VoiceConfig), check=VoiceConfig.from_raw,
    review=_wake_word, rules=(GAP,), settings=(
        Setting('mode', listed=False, options=('wake_word', 'continuous', 'manual')),
        Setting('wake_threshold', VOICE['wake_threshold'], listed=False, range=(.001, .999),
            help='Starting activation threshold for a wake word you enroll. An enrolled wake word keeps the threshold saved with it (Activation threshold in the wake word panel), which overrides this value for that name and microphone. Restart Python.'),
        Setting('follow_up_seconds', VOICE['follow_up_seconds'], listed=False, range=(.001, 300)),
        Setting('interruption_seconds', VOICE['interruption_seconds'], listed=False, range=(.001, None),
            help='Seconds of your speech over a spoken reply before it stops to listen. Restart Python.'),
        Setting('input_device', listed=False, kind='number', integer=True, nullable=True, range=(0, 10000), label='Microphone',
            help='Automatic uses the system microphone. Changes apply on backend restart.'),
        Setting('live_transcript_interval_seconds', VOICE['live_transcript_interval_seconds'], range=(.5, 10),
            help='Rolling provisional ASR updates while you are still speaking. One partial job at a time; final transcription replaces partial text. Smaller intervals increase ASR work. Restart Python.'),
        Setting('interjection_debounce_seconds', VOICE['interjection_debounce_seconds'], listed=False, range=(.001, None)),
        Setting('assertive_interruption_seconds', VOICE['assertive_interruption_seconds'], listed=False, range=(.001, 30)),
        Setting('assertive_window_seconds', VOICE['assertive_window_seconds'], listed=False, range=(.001, 60)),
        Setting('playback_words_per_second', VOICE['playback_words_per_second'], listed=False, range=(.001, None)),
        Setting('barge_in', VOICE['barge_in'], options=BARGE_IN, label='Talking over Riko',
            help="How your speech counts while Riko's own voice plays. full_duplex (default): any speech counts, so through speakers "
                'her echo can interrupt her and be saved as your words; use headphones. echo_gate: while she plays, only speech clearly '
                'louder than her echo counts: about 4 to 6 times her echo at the microphone (12-15 dB), for 2-3 seconds, to interrupt '
                'her. Each reply first learns how loud her echo is from about 2 seconds of her speech, ignoring the microphone '
                'meanwhile (so if you talk over the start of a reply, that reply may not hear you: press Stop), and a change of '
                'speakers, headphones or volume counts from her next reply (turning her up during a reply can let her echo '
                'through until then). stop_only: the microphone ignores everything while she plays; press Stop to interrupt. '
                'Restart Python.'),
        Setting('barge_in_vad_threshold', VOICE['barge_in_vad_threshold'], range=(0, 1), label='Speech certainty over playback',
            visible_when=(('voice.barge_in', ('echo_gate',)),),
            help='echo_gate only: the voice detector certainty a frame needs while Riko plays; never below voice.vad_threshold. '
                'Higher ignores more of her echo but needs clearer speech. Restart Python.'),
        Setting('echo_coupling', None, nullable=True, kind='number', range=(.001, 100), label='Echo level',
            visible_when=(('voice.barge_in', ('echo_gate',)),),
            help='echo_gate only: how loud her echo reaches the microphone at its loud end (microphone level / her playback level). '
                'Empty learns it for every reply from its first seconds of her speech, and the log reports the value to pin here. '
                'While she plays, a frame counts as your speech only above twice this times her playback level. A pinned value does '
                'not follow a change of speakers, headphones or volume: raise it if her echo still interrupts her. Restart Python.'),
        Setting('echo_tail_seconds', VOICE['echo_tail_seconds'], range=(0, 2), label='Echo tail (seconds)',
            visible_when=(('voice.barge_in', ('echo_gate', 'stop_only')),),
            help='How long after her audio stops sound is still treated as her echo (output buffering, room reverberation; in '
                'echo_gate her level fades over it); raise it for Bluetooth speakers, whose delay the system may under-report. '
                'Restart Python.'),
        Setting('vad_threshold', VOICE['vad_threshold'], listed=False, range=(0, 1),
            help='Speech probability (0–1) at which the voice detector counts a frame as speech. Higher ignores more background noise but may miss quiet speech. Restart Python.'),
        Setting('pre_roll_seconds', VOICE['pre_roll_seconds'], listed=False, range=(.001, None)),
        Setting('transcription_gap_seconds', VOICE['transcription_gap_seconds'], listed=False, range=(.001, None),
            help='Pause that sends the words so far to speech recognition. Must be shorter than the utterance end. Restart Python.'),
        Setting('utterance_end_seconds', VOICE['utterance_end_seconds'], listed=False, range=(.001, None)),
        Setting('max_segment_seconds', VOICE['max_segment_seconds'], listed=False, range=(.001, None)),
        Setting('wake_word', lambda config: wake_phrase(config.voice, config.character_name)[0], quoted=True,
            help='One short name supported by EfficientWord-Net. Without one, the first word of the companion name is used. Changing name/device requires its matching enrollment. Restart Python.'))),
    Section('speech', group='speech', title='Response chunking', strict=True, known=declared(SpeechConfig), check=SpeechConfig.from_raw,
        rules=(WINDOW,), settings=(
        Setting('max_words', SPEECH['max_words'], range=(1, 1000), label='Speech chunk limit (words)',
            help='Soft word limit. Short replies stay whole; longer replies split at preferred punctuation. No forced mid-sentence cut.'),
        Setting('split_window_words', SPEECH['split_window_words'], range=(0, 1000), label='Boundary search window (words)',
            help='Search this many words before the limit. If no boundary exists, wait for the next punctuation or generation end. Must not exceed the word limit.'),
        Setting('split_priority', SPEECH['split_priority'], label='Punctuation priority',
            help='JSON list of punctuation groups, highest priority first. Default: sentence endings, semicolon/colon, comma, newline. Each character must appear only once.'))),
    Section('sovits_ping_config', group='speech', title='Settings', strict=True, known=declared(SovitsConfig), check=SovitsConfig.from_raw, settings=(
        Setting('url', listed=False, label='GPT-SoVITS endpoint'), Setting('ref_audio_path', listed=False, label='Reference audio file'),
        Setting('prompt_text', listed=False, label='Reference transcript'),
        Setting('sample_rate', SOVITS['sample_rate'], listed=False, range=(8000, 192000)),
        Setting('max_in_flight_requests', SOVITS['max_in_flight_requests'], listed=False, range=(1, 32)),
        Setting('text_lang', listed=False, options=LANGUAGES), Setting('prompt_lang', listed=False, options=LANGUAGES),
        Setting('auto_start', SOVITS['auto_start'], restart='electron',
            help='Packaged app only: explicitly launch your GPT-SoVITS API executable on startup. The app stops only its owned direct process on quit. Save and restart the app, not just Python.'),
        Setting('executable', SOVITS['executable'], nullable=True, file=True, restart='electron',
            help='Absolute path to a GPT-SoVITS API-server executable, not a GUI launcher. Its dependencies and model weights must already be installed. Save and restart the packaged app.'),
        Setting('arguments', list(SOVITS['arguments']), restart='electron',
            help='JSON list of command-line arguments passed directly without a shell. Configure its API port to match the endpoint URL. Restart the packaged app.'),
        Setting('media_type', obsolete=True, options=('raw',), help='Client requires raw PCM streaming; keep raw.'))))
