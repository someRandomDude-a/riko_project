from __future__ import annotations

import re
import threading
import time
import uuid
import math

from ..events.bus import event_bus
from .actions import ActionController
from ..audio.speech import SpeechQueue
from ..audio.speech_chunks import SpeechChunks
from .cancellation import TurnCancelled
from .interjections import Interjections
from ..conversation.messages import ChatMessage
from ..audio.wake_word import WakeWord
from .lifecycle import close_bounded
from ..audio.wake_feedback import WakeFeedback


class SessionManager:
    """Own turns independently of capture, transcription and ordered playback."""
    def __init__(self, config, chat, state, actions=None):
        self.config, self.chat, self.state = config, chat, state
        self.actions = actions or ActionController()
        self._turn_lock = threading.Lock()
        self._voice_lock = threading.RLock()
        self._capture_lock = threading.Lock()
        self._closed = False
        self._cancel = threading.Event()
        self.asr_lock = threading.Lock()
        self._turn_speak = True
        # Validate wake configuration before starting audio workers.
        self.wake = WakeWord(config)
        try: self.speech = SpeechQueue(config, state)
        except BaseException:
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
        self._generation_active = False
        self._speech_pending = 0
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
        self.initiative = None
        self.wake_feedback = WakeFeedback(config, self.speech, self.actions)
        self._unsubscribe_wake_feedback = event_bus.subscribe(self._wake_feedback_event)
        self.animation = None
        self.state.avatar_motion = None
        self.animation_error = ''
        if getattr(self.chat, 'task_mcp', None): self.chat.task_mcp.source_turn = lambda: self._active_turn
        self.chat.memory_runtime_context = self.memory_runtime_context
        self.chat.runtime_context = self.runtime_snapshot
        registry = getattr(self.chat, 'tool_registry', None)
        if registry:
            from ..tools.registry import RegisteredTool
            registry.tools['runtime_status'] = RegisteredTool('runtime_status',
                'Inspect actual listening/speaking state, user speech, tools, actions and whiteboard. No screen/app perception is implied.',
                {'type': 'object', 'properties': {}, 'additionalProperties': False}, lambda args: self.runtime_snapshot())
            registry.tools['interrupt_user'] = RegisteredTool('interrupt_user',
                'Request temporary speaking priority for this reply when you need to interject or finish an important point. '
                'Call BEFORE speaking the interjection. User speech is still captured. Protection is bounded, once per turn; '
                'sustained speech can still interrupt and explicit Stop always works. Does not start microphone capture or create speech by itself.',
                {'type': 'object', 'properties': {'reason': {'type': 'string'}}, 'required': ['reason'], 'additionalProperties': False},
                lambda args: self.interrupt_user(**args))
        try:
            from ..animation.runtime import AnimationRuntime
            if config.raw.get('animation', {}).get('enabled', True):
                self.animation = AnimationRuntime(self)
                self.state.avatar_motion = self.animation
                if registry:
                    registry.tools['walk_avatar'] = RegisteredTool('walk_avatar',
                        'Walk the avatar to x/y pixels within the current display. User dragging overrides movement; actual completion is reported by the renderer.',
                        {'type': 'object', 'properties': {'x': {'type': 'integer'}, 'y': {'type': 'integer'}}, 'required': ['x', 'y'], 'additionalProperties': False},
                        lambda args: {'queued': self.animation.walk_to(**args).id})
                    registry.tools['avatar_animation'] = RegisteredTool('avatar_animation',
                        'List imported animations and current VRM rig capabilities, preview a compatible animation by ID, or stop walking. Choose only a listed ID; missing bones are rejected.',
                        {'type':'object','properties':{'action':{'type':'string','enum':['list','preview','stop']},'asset_id':{'type':'string'}},'required':['action'],'additionalProperties':False},
                        self.avatar_animation,
                        choices=lambda:{'action':['list','preview','stop'],'asset_id':[entry['id'] for entry in self.animation.library.list()]})
        except (ValueError, OSError) as exc:
            self.animation_error = str(exc)

    def avatar_animation(self, arguments):
        action = arguments.get('action')
        if action == 'list': return {'entries':self.animation.library.list(),'rig':self.animation.status()['capabilities']}
        if action == 'stop': self.animation.stop_movement(); return {'stopped':True}
        if action == 'preview': return {'queued':self.animation.preview(arguments.get('asset_id','')).id}
        raise ValueError('Use list, preview or stop')

    def interruption_threshold(self):
        settings = self.config.raw.get('voice', {})
        normal = float(settings.get('interruption_seconds', 1.5))
        return max(normal, float(settings.get('assertive_interruption_seconds', 6.0))) if time.monotonic() < self._assertive_until else normal

    def interrupt_user(self, reason):
        if not isinstance(reason, str) or not reason.strip(): raise ValueError('A reason is required')
        with self._voice_lock:
            if not self._generation_active or self._cancel.is_set(): raise RuntimeError('No active reply to grant speaking priority')
            if self._assertive_used: return {'granted': False, 'reason': 'Speaking priority was already used this turn'}
            settings = self.config.raw.get('voice', {})
            duration = float(settings.get('assertive_window_seconds', 15.0))
            threshold = float(settings.get('assertive_interruption_seconds', 6.0))
            if not math.isfinite(duration) or not 0 < duration <= 60 or not math.isfinite(threshold) or not 0 < threshold <= 30:
                raise ValueError('Assertive window must be 0–60 seconds and threshold 0–30 seconds (exclusive zero)')
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

    def runtime_snapshot(self):
        from copy import deepcopy
        from datetime import datetime
        with self._voice_lock:
            active_priority = time.monotonic() < self._assertive_until
            voice = self.voice
            runtime = {'turn_id': self._active_turn, 'generating': self._generation_active,
                        'generated_text': self._generated[-12000:],
                       'speaking': self._playing is not None, 'pending_speech': self._speech_pending,
                       'listening': bool(voice and not voice.closed.is_set() and self.state.mic_enabled and self._voice_status == 'ready'),
                        'microphone_status': self._voice_status, 'microphone_error': self._voice_error,
                        'voice_phase': self._voice_phase,
                        'speech_error': getattr(self.speech, 'last_error', ''),
                       'user_speaking': self._user_speaking, 'latest_transcript': deepcopy(self._live_transcript),
                       'speaking_over': [{'text': item.text, 'timestamp': item.timestamp, 'audible_offset': item.offset}
                                         for item in (self._interjections.items[-10:] if self._interjections else [])],
                       'interrupted': self._interrupt_notified, 'audible_offset': self._audible_offset(),
                       'playback_alignment': 'estimated_words', 'wake': self.wake.status(),
                       'speaking_priority': {'active': active_priority, 'reason': self._assertive_reason if active_priority else '',
                            'remaining_seconds': round(max(0, self._assertive_until - time.monotonic()), 2),
                            'interruption_seconds': self.interruption_threshold()}}
        desktop = deepcopy(self.state.snapshot())
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
                'perception': {'screen': 'unavailable',
                    'computer': initiative['environment'] if initiative else {},
                    'active_application': 'enabled' if initiative and initiative['settings']['enabled'] and initiative['settings']['observe_active_app'] else 'unavailable'}}

    def present_initiative(self, message, *, spoken=False, guard=lambda: False):
        """Commit a background proposal only while foreground interaction is idle."""
        if not self._turn_lock.acquire(blocking=False): return False
        try:
            with self._voice_lock:
                if guard() or self._closed or self._generation_active or self._playing or self._speech_pending or self._user_speaking or self.state.sleep_mode: return False
                base = len(self.chat.history)
                if spoken:
                    self._active_turn = str(uuid.uuid4())
                    self._turn_speak = True
                    self._cancel.clear()
                    self._interrupt_notified = False
                    self._generated = message
                    self._spoken_offset = self._speech_cursor = 0
                    self._playing = self._cutoff = None
                    self._assertive_until = 0
                    self._assertive_used = False
                    self._interjections = Interjections(self.cancel,
                        float(self.config.raw.get('voice', {}).get('interruption_seconds', 1.5)),
                        float(self.config.raw.get('voice', {}).get('interjection_debounce_seconds', 1.0)))
                    self._history_start, self._history_end = base, base + 1
                    self.wake.responding()
                self.chat.history.append(ChatMessage('assistant', message))
                self.chat._save_history()
                if spoken:
                    settings = self.config.raw.get('speech', {})
                    sentences = SpeechChunks(settings)
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
        finally: self._turn_lock.release()

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
            desktop = self.state.snapshot()
            model_state = ('sleeping' if self.state.sleep_mode else 'speaking' if self._playing else
                           'tool' if any(item.get('status') == 'running' for item in desktop['tools']) else
                           'thinking' if self._generation_active else 'idle')
            emotion = desktop['emotion'] or {}
            audio_enabled = self.state.audio_enabled
        self.wake_feedback.trigger(emotion.get('primary', 'neutral'), model_state, audio_enabled=audio_enabled)

    def _playback_event(self, event):
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
                if not self._live_transcript or self._live_transcript.get('utterance_id') == event.payload.get('utterance_id'):
                    self._live_transcript = {**event.payload, 'text': ' '.join(event.payload.get('text', '').split()[:400])}
                    if event.payload.get('final'): self._voice_phase = 'awake' if self.wake.active() else 'waiting'
            if event.turn_id != self._active_turn: return
            if event.type in {"speech.completed", "speech.cancelled", "speech.error"}:
                self._speech_pending = max(0, self._speech_pending - 1)
            if event.type == "speech.started" and not self._interrupt_notified:
                self._playing = dict(event.payload)
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
        speed = float(self.config.raw.get("voice", {}).get("playback_words_per_second", 2.5))
        words = list(re.finditer(r"\S+", playing["text"]))
        count = min(len(words), int(age * speed))
        start = playing.get("start_offset", self._spoken_offset)
        end = playing.get("end_offset") or start + len(playing["text"])
        return min(end, start + (words[count - 1].end() if count else 0))

    def voice_anchor(self):
        with self._voice_lock:
            return (self._active_turn, self._audible_offset(), bool(self._generated.strip())) if self._playing or self._generation_active or self._speech_pending else None

    def voice_speaking_over(self, anchor):
        return bool(anchor and (anchor[2] if len(anchor) > 2 else self._generated.strip()))

    def voice_activity(self, seconds, anchor=None):
        with self._voice_lock:
            if anchor and anchor[0] != self._active_turn: return
            if self._interjections:
                self._interjections.threshold = self.interruption_threshold()
                self._interjections.activity(seconds)

    def _response_history(self, response):
        if self._cutoff is not None: response = response[:self._cutoff]
        return self._interjections.messages(response)

    def _rewrite_history(self):
        if self._history_start is None: return
        self._rewrite_user_input()
        messages = self._response_history(self._generated)
        self.chat.history[self._history_start:self._history_end] = messages
        self._history_end = self._history_start + len(messages)
        self.chat._save_history()

    def _rewrite_user_input(self):
        base = getattr(self, '_input_history_base', None)
        if getattr(self, '_input_continued', False) and base is not None and len(self.chat.history) > base and self.chat.history[base].role == 'user':
            message = self.chat.history[base]
            prefix = f"{self._input_user_name}: " if message.content.startswith(f"{self._input_user_name}: ") else ''
            message.content = prefix + self._input_text

    def voice_transcript(self, text, started_at, ended_at, anchor=None):
        with self._voice_lock:
            if not self._interjections or (anchor and anchor[0] != self._active_turn): return False
            if getattr(self, '_input_history_base', None) is not None and not self.voice_speaking_over(anchor):
                addition = text.strip()
                if not addition: return False
                self._input_text += '\n' + addition
                self._input_continued = True
                self._interaction_revision += 1
                self._rewrite_history()
                event_bus.publish('chat.input', turn_id=self._input_turn_id, text=self._input_text, user_name=self._input_user_name)
                # Refresh inference with the combined user input; reasoning is not
                # visible output and must not create a speaking-over annotation.
                if self._generation_active: self.cancel()
                return True
            offset = anchor[1] if anchor else self._audible_offset()
            # On a sustained interruption, keep the user turn after the portion
            # that was audible at cancellation, not after generated future text.
            if self._cutoff is not None: offset = self._cutoff
            self._interjections.transcript(text, offset, started_at, ended_at)
            self._interaction_revision += 1
            self._rewrite_history()
            event_bus.publish("chat.interjection", turn_id=self._active_turn,
                              text="[speaking over you] " + text.strip(), display_text=text.strip(), system_label='Speaking over you', offset=offset,
                              started_at=started_at, ended_at=ended_at,
                              debounce_seconds=self._interjections.debounce)
            return True

    def respond(self, text: str, user_name="User", *, record_user=True, speak=True, turn_id=None):
        if self._closed: raise RuntimeError("The companion session is closed")
        self._interaction_revision += 1
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("Riko is already handling another turn")
        if self._playing or self._speech_pending:
            self.cancel() # Foreground input supersedes queued initiative/previous speech.
        base = len(self.chat.history)
        turn_id = turn_id or str(uuid.uuid4())
        from ..tools.approval import approval_turn
        approval_token = approval_turn.set(turn_id)
        try:
            settings = self.config.raw.get("voice", {})
            with self._voice_lock:
                self._cancel.clear()
                self._assertive_until = 0.0
                self._assertive_used = False
                self._assertive_reason = ''
                self._generation_active = True
                self._speech_pending = 0
                self.wake.responding()
                self._interrupt_notified = False
                self._active_turn = turn_id
                self._turn_speak = speak
                self._generated = ""
                if record_user:
                    self._input_history_base = base
                    self._input_text = text
                    self._input_user_name = user_name
                    self._input_turn_id = turn_id
                    self._input_continued = False
                self._spoken_offset = self._speech_cursor = 0
                self._playing = None
                self._cutoff = self._history_start = self._history_end = None
                self._interjections = Interjections(self.cancel,
                    float(settings.get("interruption_seconds", 1.5)),
                    float(settings.get("interjection_debounce_seconds", 1.0)))
            if record_user:
                event_bus.publish("chat.input", turn_id=turn_id, text=text, user_name=user_name)
            event_bus.publish("model.started", turn_id=turn_id)
            settings = self.config.raw.get('speech', {})
            sentences = SpeechChunks(settings)

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
                    event_bus.publish("chat.delta", turn_id=turn_id, text=delta)
                    for sentence in sentences.feed(delta): send_sentence(sentence)

            provider = getattr(self.chat, 'provider', None)
            if hasattr(provider, 'set_foreground'): provider.set_foreground(True)
            if getattr(self.chat, "memory_store", None): self.chat.memory_store.set_foreground(getattr(self.config.runtime, 'pause_background_on_live', True))
            response = self.chat.respond(text, user_name,
                max_iterations=self.config.tools.max_iterations, on_delta=on_delta,
                on_reasoning=lambda text: event_bus.publish('model.reasoning', turn_id=turn_id, text=text) if not self._cancel.is_set() else None,
                cancelled=self._cancel.is_set, response_history=self._response_history,
                record_user=record_user)
            with self._voice_lock:
                if self._cancel.is_set(): raise TurnCancelled()
                for sentence in sentences.feed("", final=True): send_sentence(sentence)
                self._history_start = base + int(record_user)
                self._history_end = len(self.chat.history)
                self._rewrite_history()
            # Frontend offsets refer to the unnormalized stream, not an added name prefix.
            event_bus.publish("chat.completed", turn_id=turn_id, text=self._generated)
            return response
        except TurnCancelled:
            if self._closed: raise
            with self._voice_lock:
                # Replace, rather than append: ChatService may have committed just
                # before cancellation arrived. Never duplicate the original turn.
                prior_user = self.chat.history[base] if record_user and len(self.chat.history) > base and self.chat.history[base].role == 'user' else ChatMessage("user", f"{user_name}: {text}")
                self.chat.history[base:] = [*([prior_user] if record_user else []),
                                           *self._response_history(self._generated)]
                self._history_start = base + int(record_user)
                self._history_end = len(self.chat.history)
                self._rewrite_user_input()
                self.chat._save_history()
            event_bus.publish("chat.cancelled", turn_id=turn_id)
            raise
        except Exception as exc:
            self.cancel()
            if not self._closed: event_bus.publish("model.error", turn_id=turn_id, error=str(exc))
            raise
        finally:
            provider = getattr(self.chat, 'provider', None)
            if hasattr(provider, 'set_foreground'): provider.set_foreground(False)
            if getattr(self.chat, "memory_store", None): self.chat.memory_store.set_foreground(False)
            with self._voice_lock:
                self._generation_active = False
                if self._speech_pending == 0:
                    self.wake.response_finished()
                    self._assertive_until = 0.0
            self._turn_lock.release()
            approval_turn.reset(approval_token)

    def cancel(self):
        with self._voice_lock:
            self._cancel.set()
            self._assertive_until = 0.0
            if self._active_turn and not self._interrupt_notified:
                self._cutoff = self._audible_offset() if self._turn_speak else len(self._generated)
                self._interrupt_notified = True
                self._rewrite_history()
                event_bus.publish("chat.interrupted", turn_id=self._active_turn,
                                  text=self._generated, offset=self._cutoff,
                                  alignment="estimated_words" if self._turn_speak else "generated_text")
            self._playing = None
            self._speech_pending = 0
            if not self._generation_active: self.wake.response_finished()
        # This only signals playback/HTTP cleanup. It never closes microphone input.
        event_bus.publish('turn.cancel_requested', turn_id=self._active_turn)
        cancel_provider = getattr(self.chat.provider, 'cancel', None)
        if cancel_provider: cancel_provider()
        self.speech.cancel()

        for action in self.actions.active():
            if action['kind'] == 'wake_animation': self.actions.cancel(action['id'])
            if action['kind'] in {'motion.locomotion', 'motion.preview'}: self.actions.cancel(action['id'])
        if self.animation: self.animation.stop_movement()

    def close(self):
        if self._closed: return
        self._closed = True
        self.cancel()
        self.state.set_speech('', seconds=0)
        self._unsubscribe_speech()
        self._unsubscribe_wake_feedback()
        for resource in (self.animation, self.initiative, self.voice, self.speech, self.wake):
            if resource: close_bounded(resource)
        self.voice = None
        if getattr(self.chat, 'close', None): close_bounded(self.chat, timeout=6)
        else:
            for resource in (getattr(self.chat, 'emotion_worker', None), self.actions,
                             getattr(self.chat, 'memory_store', None), getattr(self.chat, 'tool_registry', None), self.chat.provider):
                if resource: close_bounded(resource)
