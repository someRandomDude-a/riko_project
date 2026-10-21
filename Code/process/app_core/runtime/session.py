from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import replace

from ..events.bus import event_bus
from .actions import ActionController
from ..audio.speech import SpeechQueue
from ..audio.speech_chunks import SpeechChunks
from ..kernel.cancellation import TurnCancelled
from ..kernel.turns import RuntimeStatus, TurnContext, TurnGate
from ..audio.asr import AsrService
from .interjections import Interjections
from ..kernel.messages import ChatMessage
from ..audio.wake_word import WakeWord
from ..kernel.lifecycle import close_bounded
from ..audio.wake_feedback import WakeFeedback

logger = logging.getLogger(__name__)


class SessionManager:
    """Own turns independently of capture, transcription and ordered playback."""
    def __init__(self, config, chat, state, actions=None):
        self.config, self.chat, self.state = config, chat, state
        self._owns_actions = actions is None  # the factory's own ActionController is closed with the chat
        self.actions = actions or ActionController()
        # Lock order: _capture_lock (request handlers only) -> _voice_lock -> any component lock.
        # Bus listeners take _voice_lock, so components must not publish while holding their own
        # locks (use events.outbox.Outbox); tests/test_lock_discipline.py enforces this.
        self._turns = TurnGate()  # one foreground turn: respond() and present_initiative()
        self._voice_lock = threading.RLock()
        self._capture_lock = threading.Lock()
        self._closed = False
        self._cancel = threading.Event()
        self.asr = AsrService(config.raw.get('voice') or {})  # the one Whisper model: microphone, warmup and Discord
        self.warmed_vad = None  # warm_session's Silero model, which each VoiceInput copies
        self._turn_speak = True
        # Build the wake word before starting audio workers, and wake feedback before subscribing, so a failed
        # construction leaves no worker or bus listener behind (the app's lifespan then keeps Settings up).
        self.wake = WakeWord(config)
        try:
            self.speech = SpeechQueue(config, state)
            self.wake_feedback = WakeFeedback(config, self.speech, self.actions)
        except BaseException:
            if hasattr(self, 'speech'): self.speech.close()
            self.wake.close()
            raise
        self._active_turn = None
        self._interjections = None
        self._generated = ""
        self._spoken_offset = 0
        self._speech_cursor = 0
        self._playing = None
        self._cutoff = None
        self._history_start = None
        self._history_end = None
        self._interrupt_notified = False
        # The user message the current reply answers, which speech during its reasoning extends (voice_transcript).
        self._input_history_base = self._input_turn_id = None
        self._input_text, self._input_user_name, self._input_continued = '', 'User', False
        self._placed = {}  # cut turn -> where voice_transcript put the words its redo answers (_unplace)
        self._generation_active = False
        self._speech_pending = 0
        self._speech_failed = False  # TTS failed this turn: the user read the text instead of hearing it
        self._unsubscribe_speech = event_bus.subscribe(self._playback_event)
        self.voice = None
        self._voice_status = 'stopped'
        self._voice_error = ''
        self._voice_phase = 'stopped'
        self._user_speaking = False
        self._live_transcript = None
        self._assertive_until = 0.0
        self._assertive_used = False
        self._assertive_reason = ''
        self._interaction_revision = 0
        self._origin = {'source': 'message'}
        # The probe asks from its own threads, maybe under its own locks, so this never waits for _voice_lock.
        chat.provider.set_expression_idle(lambda: self.status(locked=False).is_quiet())
        self.initiative = None
        self._unsubscribe_wake_feedback = event_bus.subscribe(self._wake_feedback_event)
        self.animation = None
        self.animation_error = ''
        deps = chat.deps  # what the factory built besides the provider (ChatDeps, empty for a ChatService built directly)
        if deps.task_mcp: deps.task_mcp.source_turn = lambda: self._active_turn  # task changes record the turn making them
        try:
            from ..animation.runtime import AnimationRuntime
            animation = config.raw.get('animation', {})
            # `animation: false` turns it off; any other non-section value is rejected by AnimationRuntime (ValueError).
            if animation is not False and (not isinstance(animation, dict) or animation.get('enabled', True)):
                self.animation = AnimationRuntime(self)
                if deps.desktop: deps.desktop.avatar_motion = self.animation  # move_avatar walks instead of teleporting
        except (ValueError, OSError) as exc:
            self.animation_error = str(exc)
        registry = getattr(self.chat, 'tool_registry', None)
        if registry:
            from ..tools.tool import RIKO
            for tool in self.runtime_tools(): registry.register(tool, source=RIKO)

    def runtime_tools(self):
        """The session's own tools, registered like every other (ToolRegistry.register raises on a clash)."""
        from ..tools.tool import RegisteredTool
        tools = [RegisteredTool('runtime_status',
            'Inspect actual listening/speaking state, user speech, tools, actions and whiteboard. No screen/app perception is implied.',
            {'type': 'object', 'properties': {}, 'additionalProperties': False}, lambda args: self.runtime_snapshot(), owner='session'),
            RegisteredTool('interrupt_user',
            'Request temporary speaking priority for this reply when you need to interject or finish an important point. '
            'Call BEFORE speaking the interjection. User speech is still captured. Protection is bounded, once per turn; '
            'sustained speech can still interrupt and explicit Stop always works. Does not start microphone capture or create speech by itself.',
            {'type': 'object', 'properties': {'reason': {'type': 'string'}}, 'required': ['reason'], 'additionalProperties': False},
            lambda args: self.interrupt_user(**args), owner='session')]
        if self.animation:
            def animation_choices(arguments=None):  # an empty asset_id is the default the model filled in, not a choice
                ids = {'asset_id': [entry['id'] for entry in self.animation.library.list()]} if not arguments or arguments.get('asset_id') != '' else {}
                return {'action': ['list', 'preview', 'stop'], **ids}
            tools += [RegisteredTool('walk_avatar',
                'Walk the avatar to x/y pixels within the current display. User dragging overrides movement; actual completion is reported by the renderer.',
                {'type': 'object', 'properties': {'x': {'type': 'integer'}, 'y': {'type': 'integer'}}, 'required': ['x', 'y'], 'additionalProperties': False},
                lambda args: {'queued': self.animation.walk_to(**args).id}, owner='session'),
                RegisteredTool('avatar_animation',
                'List imported animations and current VRM rig capabilities, preview a compatible animation by ID, or stop walking. Choose only a listed ID; missing bones are rejected.',
                {'type':'object','properties':{'action':{'type':'string','enum':['list','preview','stop']},'asset_id':{'type':'string'}},'required':['action'],'additionalProperties':False},
                self.avatar_animation, choices=animation_choices, owner='session')]
        return tools

    def avatar_animation(self, arguments):
        action = arguments.get('action')
        if action == 'list': return {'entries':self.animation.library.list(),'rig':self.animation.status()['capabilities']}
        if action == 'stop': self.animation.stop_movement(); return {'stopped':True}
        if action == 'preview': return {'queued':self.animation.preview(arguments.get('asset_id','')).id}
        raise ValueError('Use list, preview or stop')

    def interruption_threshold(self):
        voice = self.config.voice
        return max(voice.interruption_seconds, voice.assertive_interruption_seconds) if time.monotonic() < self._assertive_until else voice.interruption_seconds

    def interrupt_user(self, reason):
        if not isinstance(reason, str) or not reason.strip(): raise ValueError('A reason is required')
        with self._voice_lock:
            if not self._generation_active or self._cancel.is_set(): raise RuntimeError('No active reply to grant speaking priority')
            if self._assertive_used: return {'granted': False, 'reason': 'Speaking priority was already used this turn'}
            duration = self.config.voice.assertive_window_seconds  # 0-60 s and its threshold 0-30 s (VoiceConfig.from_raw)
            self._assertive_used = True
            self._assertive_until = time.monotonic() + duration
            self._assertive_reason = reason.strip()
            result = {'granted': True, 'window_seconds': duration, 'interruption_seconds': self.interruption_threshold(),
                      'reason': self._assertive_reason, 'user_speech_preserved': True}
        event_bus.publish('voice.speaking_priority', turn_id=self._active_turn, **result)
        return result

    def user_interrupted(self, anchor):
        with self._voice_lock:
            return bool(anchor and anchor[0] == self._active_turn and self._interjections and self._interjections.interrupted)

    @property
    def is_open(self): return not self._closed

    def status(self, *, locked=True):
        """One RuntimeStatus reading, whose predicates are the only definition of busy and idle. It takes _voice_lock, so a
        caller holding a component lock, or a callback a provider or tool may run under its own lock (a cancel poll,
        expression_idle), passes locked=False and gets the fields read one by one instead."""
        if not locked: return self._read_status()
        with self._voice_lock: return self._read_status()

    def _read_status(self):  # takes no lock itself, so the session's own code calls it under _voice_lock
        voice, wake = self.voice, self.wake
        capture = bool(voice and not voice.closed.is_set())
        return RuntimeStatus(closed=self._closed, active_turn=self._active_turn, generating=self._generation_active,
            speaking=self._playing is not None, pending_audio=self._speech_pending, spoken_here=self._turn_speak,
            user_speaking=self._user_speaking, capture_running=capture,
            listening=capture and bool(self.state.mic_enabled) and self._voice_status == 'ready',
            voice_status=self._voice_status, voice_phase=self._voice_phase, sleeping=bool(self.state.sleep_mode),
            calibrating=bool(wake.calibrating or wake.testing),
            speaking_priority=time.monotonic() < self._assertive_until, interaction_revision=self._interaction_revision)

    def active_turn(self):
        """The latest foreground turn's id (None before the first), which may already have finished."""
        with self._voice_lock: return self._active_turn

    def cancel_turn(self, turn_id):
        """Stop turn_id if it is still generating. False, stopping nothing, when another turn or none is running."""
        with self._voice_lock:
            matched = turn_id is not None and turn_id == self._active_turn and self._generation_active
            if matched: self.cancel()  # still under the lock, so a turn that starts meanwhile is never the one stopped
        return matched

    def set_user_speaking(self, speaking):
        """VoiceInput reports what VAD hears on the open microphone, frame by frame."""
        with self._voice_lock: self._user_speaking = bool(speaking)

    def transcribe(self, pcm, **options):
        """The text of mono 16 kHz PCM16 through the session's one Whisper model (options go to WhisperModel.transcribe)."""
        return self.asr.transcribe_pcm(pcm, **options)

    def history_snapshot(self):
        """A copy of the conversation history, never one that an interjection or a cut is halfway through rewriting: each
        rewrite is one step of the history's owner (ConversationHistory)."""
        return self.chat.conversation.snapshot()

    def runtime_snapshot(self):
        from copy import deepcopy
        from datetime import datetime
        with self._voice_lock:
            status = self._read_status()
            active_priority = status.speaking_priority
            runtime = {'turn_id': status.active_turn, 'generating': status.generating,
                        'generated_text': self._generated[-12000:], 'input_origin': deepcopy(getattr(self, '_origin', {'source':'message'})),
                       'speaking': status.speaking, 'pending_speech': status.pending_audio, 'listening': status.listening,
                        'microphone_status': status.voice_status, 'microphone_error': self._voice_error,
                        'voice_phase': status.voice_phase,
                        'speech_error': getattr(self.speech, 'last_error', ''),
                       'user_speaking': status.user_speaking, 'latest_transcript': deepcopy(self._live_transcript),
                       'speaking_over': [{'text': item.text, 'timestamp': item.timestamp, 'audible_offset': item.offset}
                                         for item in (self._interjections.items[-10:] if self._interjections else [])],
                       'interrupted': self._interrupt_notified, 'audible_offset': self._audible_offset(),
                       'playback_alignment': 'estimated_words', 'wake': self.wake.status(),
                       'speaking_priority': {'active': active_priority, 'reason': self._assertive_reason if active_priority else '',
                            'remaining_seconds': round(max(0, self._assertive_until - time.monotonic()), 2),
                            'interruption_seconds': self.interruption_threshold()}}
        desktop = deepcopy(self.state.snapshot())
        # Full text remains in UI/history. A small recent-input window keeps
        # source awareness from injecting thousands of repeated prompt tokens.
        for key, limit in (('incoming', 5), ('notifications', 3)):
            items = desktop.get(key, [])
            desktop[key] = [{**item, 'text': item.get('text', '')[:240]} for item in items[:limit]]
            desktop[key + '_omitted'] = max(0, len(items) - limit)
        for key, limit in (('tools', 10), ('actions', 10), ('whiteboard', 20)):
            items = desktop.get(key, [])
            desktop[key] = items[-limit:] if key == 'whiteboard' else items[:limit]
            desktop[key + '_omitted'] = max(0, len(items) - limit)
        # Task contents are fetched deliberately, not re-injected through the
        # generic recent-tool observation on every later turn.
        desktop['tools'] = [{k: v for k, v in item.items() if k not in {'arguments', 'result'}}
                             if item.get('name', '').startswith('task_') or item.get('name') == 'todo_list' else item
                            for item in desktop.get('tools', [])]
        initiative = self.initiative.snapshot() if self.initiative else None
        return {'observed_at': datetime.now().astimezone().isoformat(timespec='seconds'),
                'runtime': runtime, 'desktop': desktop,
                # idle/active_application: off, unsupported (this OS or desktop), waiting, available or error, from the last sample
                'perception': {'screen': 'unavailable',
                    'computer': initiative['environment'] if initiative else {},
                    'idle': initiative['perception']['idle'] if initiative else 'unavailable',
                    'active_application': initiative['perception']['active_app'] if initiative else 'unavailable'}}

    def present_initiative(self, message, *, spoken=False, guard=lambda: False):
        """Commit a background proposal only while foreground interaction is idle."""
        if not self._turns.try_begin(): return False
        try:
            with self._voice_lock:
                if guard() or not self._read_status().idle_for_background(): return False
                # Build the speech helpers before committing: a failure fails this presentation without leaving a message
                # nobody saw or heard in history (and another on every retry).
                voice = self.config.voice
                interjections = Interjections(self.cancel, voice.interruption_seconds, voice.interjection_debounce_seconds) if spoken else None
                sentences = SpeechChunks(self.config.speech) if spoken else None
                self._input_history_base = None  # No later speech may rewrite the input of a reply this message follows.
                start = self.chat.conversation.append([ChatMessage('assistant', message)])
                if spoken:
                    self._active_turn = str(uuid.uuid4())
                    self._turn_speak = True
                    self._cancel.clear()
                    self._interrupt_notified = False
                    self._generated = message
                    self._spoken_offset = self._speech_cursor = 0
                    self._speech_failed = False
                    self._playing = self._cutoff = None
                    self._assertive_until = 0
                    self._assertive_used = False
                    self._interjections = interjections
                    self._history_start, self._history_end = start, start + 1
                    self.wake.responding()
                if spoken:
                    cursor = 0
                    for text in sentences.feed(message, final=True):
                        pattern = r'\s+'.join(re.escape(part) for part in text.split())
                        match = re.search(pattern, message[cursor:])
                        start = cursor + match.start() if match else cursor
                        end = cursor + match.end() if match else start + len(text)
                        if self.speech.submit(text, self._active_turn, start, end): self._speech_pending += 1
                        cursor = end
                    if not self._speech_pending: self.wake.response_finished()
                self.state.set_speech(message, seconds=20)
                event_bus.publish('chat.completed', turn_id=self._active_turn if spoken else str(uuid.uuid4()), text=message,
                    initiative=True, spoken=spoken)
            return True
        finally: self._turns.end()

    def memory_runtime_context(self):
        return self.runtime_snapshot()

    def start_listening(self):
        from ..audio.voice_input import VoiceInput
        with self._capture_lock:
            if self.voice is None or self.voice.closed.is_set():
                if self.voice: self.voice.close()
                self._voice_status = 'starting'
                self._voice_error = ''
                event_bus.publish('voice.starting')
                self.voice = VoiceInput(self)

    def stop_listening(self):
        with self._capture_lock:
            if self.voice:
                self.voice.close()
                self.voice = None
                self._voice_status = 'stopped'
                self._user_speaking = False
            event_bus.publish('voice.stopped')

    def _wake_feedback_event(self, event):
        if event.type != 'voice.activated' or event.payload.get('source') != 'keyword': return
        with self._voice_lock:
            if self._closed or not self.state.mic_enabled: return
            desktop, status = self.state.snapshot(), self._read_status()
            audio_enabled = self.state.audio_enabled
        # The state the avatar shows, except that the user speech it heard is the wake word itself.
        model_state = status.model_state(tool_running=any(item.get('status') == 'running' for item in desktop['tools']))
        emotion = desktop['emotion'] or {}
        self.wake_feedback.trigger(emotion.get('primary', 'neutral'), 'idle' if model_state == 'listening' else model_state, audio_enabled=audio_enabled)

    def _playback_event(self, event):
        # Only voice/speech events change playback state; skip the rest before locking.
        if not event.type.startswith(('voice.', 'speech.')): return
        with self._voice_lock:
            if event.type == 'voice.starting': self._voice_phase = 'starting'
            elif event.type == 'voice.activated': self._voice_phase = 'awake'
            elif event.type == 'voice.follow_up': self._voice_phase = 'follow_up'
            elif event.type == 'voice.waiting' and self._voice_phase not in {'capturing','transcribing'}: self._voice_phase = 'waiting'
            elif event.type == 'voice.started':
                self._voice_phase = 'capturing'
                self._live_transcript = {'utterance_id':event.payload.get('utterance_id'), 'text':'', 'final':False}
            elif event.type == 'voice.resumed':
                if self._live_transcript and self._live_transcript.get('utterance_id') == event.payload.get('utterance_id'): self._voice_phase = 'capturing'
            elif event.type in {'voice.utterance_ended','voice.transcribing'}:
                if not self._live_transcript or self._live_transcript.get('utterance_id') == event.payload.get('utterance_id'): self._voice_phase = 'transcribing'
            if event.type == 'voice.ready':
                self._voice_status, self._voice_error = 'ready', ''
                if self._voice_phase == 'starting': self._voice_phase = 'awake' if self.wake.active() else 'waiting'
            elif event.type == 'voice.stopped':
                self._voice_status, self._user_speaking = 'stopped', False
                self._voice_phase = 'stopped'
            elif event.type == 'voice.error':
                self._voice_error = event.payload.get('error', '')
            elif event.type == 'voice.transcript':
                if event.payload.get('text'):
                    worker = getattr(self.chat, 'emotion_worker', None)
                    if worker: worker.transcript(event.payload['text'],event.payload.get('utterance_id'))
                if not self._live_transcript or self._live_transcript.get('utterance_id') == event.payload.get('utterance_id'):
                    self._live_transcript = {**event.payload, 'text': ' '.join(event.payload.get('text', '').split()[:400])}
                    if event.payload.get('final'): self._voice_phase = 'awake' if self.wake.active() else 'waiting'
            if event.turn_id != self._active_turn: return
            # A chunk rejected at submit (queued=False, e.g. queue full) was never pending and does not
            # mean synthesis failed; later chunks still play. Only real TTS/playback errors count.
            if event.type == "speech.error" and event.payload.get('queued', True): self._speech_failed = True
            if event.type in {"speech.completed", "speech.cancelled", "speech.error"} and event.payload.get('queued', True):
                self._speech_pending = max(0, self._speech_pending - 1)
            if event.type == "speech.started" and not self._interrupt_notified:
                self._playing = dict(event.payload)
                worker = getattr(self.chat, 'emotion_worker', None)
                probe = self._probe()
                if not (probe and probe.expression_for_segment(event.payload.get('end_offset') or 0)) and worker:
                    worker.submit('speech', event.payload.get('text',''), final=True)
            elif event.type == "speech.completed":
                if not self._interrupt_notified:
                    self._spoken_offset = event.payload.get("end_offset") or self._spoken_offset
                self._playing = None
            elif event.type in {"speech.cancelled", "speech.error"}:
                self._playing = None
            if not self._generation_active and self._speech_pending == 0 and not self._interrupt_notified:
                if event.type in {"speech.completed", "speech.cancelled", "speech.error"}:
                    self.wake.response_finished()
                    self._assertive_until = 0.0

    def _audible_offset(self):
        if not self._playing: return self._spoken_offset
        # Raw PCM has no phoneme/word timestamps. Estimate within the CURRENT
        # playing segment only; never count synthesized-but-unplayed segments.
        playing = self._playing
        age = max(0, time.monotonic() - playing["started_at"])
        speed = self.config.voice.playback_words_per_second
        words = list(re.finditer(r"\S+", playing["text"]))
        count = min(len(words), int(age * speed))
        start = playing.get("start_offset", self._spoken_offset)
        end = playing.get("end_offset") or start + len(playing["text"])
        return min(end, start + (words[count - 1].end() if count else 0))

    def voice_anchor(self):
        # Only a reply spoken here can be talked over: a remote (speak=False) turn never anchors local speech.
        with self._voice_lock:
            return (self._active_turn, self._audible_offset(), bool(self._generated.strip())) if self._read_status().local_reply() else None

    def voice_speaking_over(self, anchor):
        return bool(anchor and (anchor[2] if len(anchor) > 2 else self._generated.strip()))

    def voice_activity(self, seconds, anchor=None):
        with self._voice_lock:
            if not self._turn_speak or (anchor and anchor[0] != self._active_turn): return
            if self._interjections and self._turn_in_flight():  # speech after Stop cut nothing
                self._interjections.threshold = self.interruption_threshold()
                self._interjections.activity(seconds)

    def _response_history(self, response):
        if self._cutoff is not None: response = response[:self._cutoff]
        origin = getattr(self, '_origin', {})
        return [replace(message, source='microphone' if message.role == 'user' else origin.get('source'),
            conversation_id=origin.get('conversation_id')) for message in self._interjections.messages(response)]

    def _rewrite_history(self, reply=None):
        """Rewrite the current turn's recorded reply (speech over it, a cut) and its continued input in one step. Caller
        holds _voice_lock."""
        if self._history_start is None: return
        start, end, messages = self._history_start, self._history_end, self._response_history(self._generated if reply is None else reply)
        def change(history):
            self._continue_input(history)
            history[start:end] = messages
        self.chat.conversation.rewrite(change)
        self._history_end = start + len(messages)

    def _continue_input(self, history):
        """In a history being rewritten, extend the user message this reply answers with the speech that continued it."""
        base = self._input_history_base
        if self._input_continued and base is not None and len(history) > base and history[base].role == 'user':
            message = history[base]
            prefix = f"{self._input_user_name}: " if message.content.startswith(f"{self._input_user_name}: ") else ''
            history[base] = replace(message, content=prefix + self._input_text)

    def voice_transcript(self, text, started_at, ended_at, anchor=None):
        """Place the final transcript of speech that began during a spoken reply, and return what the caller must do:
        'preserved' (kept as speaking over; no reply), 'reply' (kept, and the reply it cut must be redone with
        respond(record_user=False, reply_to=anchor[0])), 'fresh' (not part of that reply: dispatch it as a new turn)
        or 'ignored' (no words)."""
        addition = text.strip()
        if not addition: return 'ignored'
        with self._voice_lock:
            anchor = anchor or self.voice_anchor()
            # No reply in flight, a remote (speak=False) one, or one a newer turn replaced: the words are a turn of their own.
            if not anchor or anchor[0] != self._active_turn or not self._turn_speak or not self._interjections: return 'fresh'
            # Follow actual VAD cancellation, not a threshold re-evaluated after slow ASR or speaking-priority expiry.
            interrupted = self._interjections.interrupted
            cut = interrupted and self._interrupt_notified # sustained speech cut this reply short
            if not self.voice_speaking_over(anchor):
                # Nothing was visible when the user spoke, so the words extend the input this reply answers and the reply
                # is redone. A reply that already finished (or was stopped) answered only the earlier words: a new turn.
                if self._input_history_base is None or not (cut or self._generation_active and not self._cancel.is_set()): return 'fresh'
                self._input_text += '\n' + addition
                self._input_continued = True
                self._interaction_revision += 1
                # Refresh inference with the combined user input; reasoning is not
                # visible output and must not create a speaking-over annotation.
                if self._generation_active: self.cancel()
                # Nothing was visible when the user resumed, so a reply cut for the refresh (now, or
                # earlier by sustained speech) is discarded, never kept as read, even when muted.
                if self._interrupt_notified: self._cutoff = 0
                self._rewrite_history()
                self._place(('input', self._input_user_name, self._input_text))
                event_bus.publish('chat.input', turn_id=self._input_turn_id, text=self._input_text, user_name=self._input_user_name, **getattr(self, '_origin', {}))
                return 'reply'
            # Sustained speech that cut nothing: the reply had already ended, so the words start a turn of their own.
            if interrupted and not cut: return 'fresh'
            offset = anchor[1]
            # On a sustained interruption, keep the user turn after the portion
            # that was audible at cancellation, not after generated future text.
            if self._cutoff is not None: offset = self._cutoff
            self._interjections.transcript(text, offset, started_at, ended_at)
            self._interaction_revision += 1
            self._rewrite_history()
            event_bus.publish("chat.interjection", turn_id=self._active_turn,
                              text="[speaking over you] " + text.strip(), display_text=text.strip(), system_label='Speaking over you', offset=offset,
                               started_at=started_at, ended_at=ended_at,
                               debounce_seconds=self._interjections.debounce, source='microphone', conversation_id=getattr(self, '_origin', {}).get('conversation_id'))
            if cut: self._place(('interjections', self._interjections, offset))
            return 'reply' if cut else 'preserved'

    def _place(self, placement):
        self._placed[self._active_turn] = placement
        while len(self._placed) > 8: self._placed.pop(next(iter(self._placed)))  # redos that never ran

    def _unplace(self, placement, text, user_name):
        """A cut reply's redo that another turn overtook becomes a new turn: take back the words voice_transcript placed
        in the cut turn's history and return them as its input, so they are recorded once. Caller holds _voice_lock."""
        if placement and placement[0] == 'interjections':
            _, interjections, offset = placement
            with interjections.lock: items = [item for item in interjections.items if item.offset >= offset]  # at the cut
            placed = {('[speaking over you] ' + item.text, item.timestamp) for item in items}
        def change(history):
            if placement and placement[0] == 'input':
                _, name, merged = placement
                for index in range(len(history) - 1, -1, -1):
                    if history[index].role == 'user' and history[index].content in {merged, f'{name}: {merged}'}:
                        # Its reply was discarded for the redo, so the whole merged question is still unanswered.
                        if index + 1 < len(history) and history[index + 1].role == 'assistant': break
                        del history[index]
                        return merged, name
            elif placement and placement[0] == 'interjections':
                kept = [message for message in history if not (message.role == 'user' and (message.content, message.timestamp) in placed)]
                if len(kept) < len(history):
                    history[:] = kept
                    return ' '.join(item.text for item in items), user_name
            return text, user_name
        return self.chat.conversation.rewrite(change)

    def respond(self, text: str, user_name="User", *, record_user=True, speak=True, turn_id=None, origin=None, reply_to=None, wait=None):
        """Run one foreground turn. Without `wait` a busy session refuses at once (TurnBusy, which every route answers with
        409); a queued voice turn passes wait, a callable that abandons it (TurnCancelled), and waits for the running turn
        instead of being lost. reply_to redoes that turn's cut reply (record_user=False); if any turn or initiative came
        after it, the text is a new turn."""
        if self._closed: raise RuntimeError("The companion session is closed")
        origin = dict(origin or {'source':'message'})
        if origin.get('source') not in {'message','microphone','discord'}: raise ValueError('Unknown input source')
        turn_id = turn_id or str(uuid.uuid4())
        with self._voice_lock: self._interaction_revision += 1  # Before waiting: a background initiative check backs off.
        self._turns.begin(None if wait is None else lambda: self._closed or wait())
        try:
            if wait is not None and (self._closed or wait()): raise TurnCancelled()
            # Only a turn that runs records its input: a refused one was never taken.
            if hasattr(self.state, 'observe_input'):
                self.state.observe_input(origin['source'], text, message_id=origin.get('message_id') or turn_id,
                    context={key: origin[key] for key in ('user_id','channel_id','guild_id','conversation_id') if key in origin})
            with self._voice_lock:
                placement = self._placed.pop(reply_to, None) if reply_to is not None else None
                if reply_to is not None and (reply_to != self._active_turn or self._history_end != len(self.chat.conversation)):
                    text, user_name = self._unplace(placement, text, user_name)
                    record_user = True
            if self._playing or self._speech_pending:
                self.cancel() # Foreground input supersedes queued initiative/previous speech.
            base = len(self.chat.conversation)
            from ..tools.approval import approval_turn
            approval_token = approval_turn.set(turn_id)
        except BaseException:
            self._turns.end() # A leaked turn would refuse every later one until restart.
            raise
        try:
            voice = self.config.voice
            # Build interjections before touching turn state: a failure must fail this turn
            # cleanly, not leave the previous turn's interjections attached to the new one.
            interjections = Interjections(self.cancel, voice.interruption_seconds, voice.interjection_debounce_seconds)
            with self._voice_lock:
                base = len(self.chat.conversation)  # measured as this turn takes over: a late transcript for the last one can no longer move it
                self._cancel.clear()
                self._assertive_until = 0.0
                self._assertive_used = False
                self._assertive_reason = ''
                self._generation_active = True
                self._origin = origin
                self._speech_pending = 0
                if speak: self.wake.responding() # A remote reply never opens the local microphone's wake window.
                self._interrupt_notified = False
                self._active_turn = turn_id
                self._turn_speak = speak
                self._generated = ""
                # The input speech during reasoning may extend: this turn's own, or for a redo the input just before it.
                # Any other turn has none, so an older user message is never rewritten.
                if record_user: self._input_history_base, self._input_text, self._input_user_name, self._input_turn_id, self._input_continued = base, text, user_name, turn_id, False
                elif self._input_history_base != base - 1: self._input_history_base = None
                self._spoken_offset = self._speech_cursor = 0
                self._speech_failed = False
                self._playing = None
                self._cutoff = self._history_start = self._history_end = None
                self._interjections = interjections
            if record_user:
                event_bus.publish("chat.input", turn_id=turn_id, text=text, user_name=user_name, **origin)
            event_bus.publish("model.started", turn_id=turn_id, **origin)
            sentences = SpeechChunks(self.config.speech)

            def send_sentence(sentence):
                if not speak: return # Remote clients render/export audio themselves.
                # Splitter collapses whitespace; recover original token offsets.
                pattern = r"\s+".join(re.escape(part) for part in sentence.split())
                match = re.search(pattern, self._generated[self._speech_cursor:])
                start = self._speech_cursor + match.start() if match else self._speech_cursor
                end = self._speech_cursor + match.end() if match else start + len(sentence)
                self._speech_cursor = end
                if self.speech.submit(sentence, turn_id, start, end): self._speech_pending += 1

            def on_delta(delta):
                with self._voice_lock:
                    if self._cancel.is_set(): raise TurnCancelled()
                    self._generated += delta
                    worker = getattr(self.chat, 'emotion_worker', None)
                    probe = self._probe()
                    if worker and not (probe and probe.ready and probe.config.use_for_expression): worker.generation(delta)
                    event_bus.publish("chat.delta", turn_id=turn_id, text=delta, **origin)
                    for sentence in sentences.feed(delta): send_sentence(sentence)

            def on_reasoning(text):
                # Like on_delta: Stop ends a long reasoning phase now, not at the first visible word.
                if self._cancel.is_set(): raise TurnCancelled()
                event_bus.publish('model.reasoning', turn_id=turn_id, text=text)

            self.chat.provider.set_foreground(True)
            if getattr(self.chat, "memory_store", None): self.chat.memory_store.set_foreground(getattr(self.config.runtime, 'pause_background_on_live', True))
            response = self.chat.respond(text, user_name,
                max_iterations=self.config.tools.max_iterations, on_delta=on_delta,
                on_metrics=lambda value: event_bus.publish('model.metrics', turn_id=turn_id, **value) if not self._cancel.is_set() else None,
                on_reasoning=on_reasoning,
                cancelled=self._cancel.is_set, response_history=self._response_history,
                record_user=record_user, context=TurnContext(origin=origin, runtime=self.runtime_snapshot,
                    memory_runtime=self.memory_runtime_context, emotion_from_playback=True))  # sentences feed Julia as they play
            with self._voice_lock:
                if self._cancel.is_set(): raise TurnCancelled()
                worker = getattr(self.chat, 'emotion_worker', None)
                if worker: worker.generation('', final=True)
                for sentence in sentences.feed("", final=True): send_sentence(sentence)
                # A reply with no visible text (reasoning used the whole budget) keeps ChatService's placeholder in history,
                # but not in _generated: that stays what was shown or spoken, which voice anchors read.
                shown = self._generated if self._generated.strip() else response.message.content
                self._history_start = base + int(record_user)
                self._history_end = len(self.chat.conversation)
                self._rewrite_history(shown)
            # Frontend offsets refer to the unnormalized stream, not an added name prefix.
            event_bus.publish("chat.completed", turn_id=turn_id, text=shown, **origin)
            return response
        except TurnCancelled:
            # Shutdown cancels too: still keep what the user said and what they already saw or heard.
            self._preserve_turn(base, turn_id, text, user_name, record_user, origin)
            if not self._closed: event_bus.publish("chat.cancelled", turn_id=turn_id, **origin)
            raise
        except Exception as exc:
            # A failed reply is not a user interruption: stop output without cutting history,
            # then keep the user's message (and any partial reply) so the model sees what was asked.
            # Neither step may hide the original error from the caller or the model.error event.
            try: self._stop(interrupt=False)
            except Exception: logger.exception('Could not stop output after a failed reply')
            self._preserve_turn(base, turn_id, text, user_name, record_user, origin)
            if not self._closed: event_bus.publish("model.error", turn_id=turn_id, error=str(exc), **origin)
            raise
        finally:
            self.chat.provider.set_foreground(False)
            if getattr(self.chat, "memory_store", None): self.chat.memory_store.set_foreground(False)
            with self._voice_lock:
                self._generation_active = False
                if self._speech_pending == 0:
                    if speak: self.wake.response_finished() # A remote reply opens no local follow-up window.
                    self._assertive_until = 0.0
            self._turns.end()
            approval_turn.reset(approval_token)

    def _preserve_turn(self, base, turn_id, text, user_name, record_user, origin):
        """Keep an unfinished turn in history. Never raises: callers re-raise the real reason."""
        try:
            with self._voice_lock:
                # Replace, rather than append: ChatService may have committed just before the
                # turn was cancelled or failed. Never duplicate the original turn.
                asked = ChatMessage("user", f"{user_name}: {text}", source=origin.get('source'), conversation_id=origin.get('conversation_id'))
                reply = self._response_history(self._generated) if self._active_turn == turn_id else []
                def change(history):
                    prior_user = history[base] if record_user and len(history) > base and history[base].role == 'user' else asked
                    history[base:] = [*([prior_user] if record_user else []), *reply]
                    self._continue_input(history)
                    return len(history)
                self._history_end = self.chat.conversation.rewrite(change)
                self._history_start = base + int(record_user)
        except Exception: logger.exception('Could not keep the unfinished turn in chat history')

    def _probe(self):
        """The running emotion probe, or None (no ProbeHost, not enabled, or it did not start)."""
        host = self.chat.provider.probe_host
        return host.probe if host is not None else None

    def _turn_in_flight(self):
        return bool(self._active_turn and not self._interrupt_notified and self._read_status().replying())

    def _speech_heard(self):
        """False when speech is muted or TTS failed this turn: the user read the text instead."""
        return bool(self._turn_speak and getattr(self.state, 'audio_enabled', True) and not self._speech_failed)

    def cancel(self):
        self._stop(interrupt=True)

    def _stop(self, *, interrupt):
        worker = getattr(self.chat, 'emotion_worker', None)
        if worker: worker.invalidate()
        with self._voice_lock:
            self._cancel.set()
            self._assertive_until = 0.0
            # Only a reply still generating or playing is interrupted. A finished turn stays
            # intact when Stop, Sleep or shutdown arrive afterwards.
            if interrupt and self._turn_in_flight():
                heard = self._speech_heard()
                self._cutoff = self._audible_offset() if heard else len(self._generated)
                self._interrupt_notified = True
                try: self._rewrite_history()
                except Exception: logger.exception('Could not save chat history after an interruption')
                event_bus.publish("chat.interrupted", turn_id=self._active_turn,
                                  text=self._generated, offset=self._cutoff,
                                  alignment="estimated_words" if heard else "generated_text")
            self._playing = None
            self._speech_pending = 0
            if not self._generation_active: self.wake.response_finished()
        # This only signals playback/HTTP cleanup. It never closes microphone input.
        event_bus.publish('turn.cancel_requested', turn_id=self._active_turn)
        self.chat.provider.cancel()  # the live request only: initiative and reflection keep their slots
        self.speech.cancel()

        for action in self.actions.active():
            if action['kind'] == 'wake_animation': self.actions.cancel(action['id'])
            if action['kind'] in {'motion.locomotion', 'motion.preview'}: self.actions.cancel(action['id'])
        if self.animation: self.animation.stop_movement()

    def close(self):
        if self._closed: return
        self._closed = True
        # Unsubscribe first, so a stalled cancel() can never block bus publishers on this session.
        self._unsubscribe_speech()
        self._unsubscribe_wake_feedback()
        self.cancel()
        self.state.set_speech('', seconds=0)
        for resource in (self.animation, self.initiative, self.voice, self.speech, self.wake, self.asr):
            if resource: close_bounded(resource)
        self.voice = None
        close_bounded(self.chat, timeout=6)  # ChatService.close: the factory's ExitStack, or what the chat was built with
        if self._owns_actions: close_bounded(self.actions)
