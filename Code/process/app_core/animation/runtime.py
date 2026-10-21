"""Coalesced Julia selection off capture/render threads, with immediate rule fallback."""
from copy import deepcopy
import math
import re
import json
import threading
import time

from .library import AnimationLibrary, BONES, validate_settings
from .policy import AnimationState, eligible_intents, motion_intent
from ..events.bus import event_bus
from ..kernel.workers import DaemonExecutor


class AnimationRuntime:
    def __init__(self, session, *, start=True):
        self.session = session
        self.settings = validate_settings(session.config.raw.get('animation', {}))
        self.library = AnimationLibrary(session.config.root)
        engine = getattr(session.chat, 'emotion_engine', None)
        self.selector = getattr(engine, 'choose_motion', None) if self.settings['julia_selection'] else None
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.wake = threading.Event()
        self.last_status = None
        self.executor = DaemonExecutor(max_workers=1, thread_name_prefix='animation-policy', max_pending=1)
        self.capabilities = {'bones': [], 'expressions': [], 'ready': False}
        self.interaction_state = {'held': False, 'near': False, 'pointer': None, 'event': '', 'until': 0.0, 'updated': 0.0}
        self.revision = self.library_revision = 0
        self.last_key = None
        self.last_apply = -math.inf
        self.last_julia_revision = -1
        self.pending = None
        self.current = None
        self.state = None
        self.error = ''
        self.failed_assets = set()
        self.movement = None
        self.unsubscribe_events = event_bus.subscribe(self._event)
        self.unsubscribe_state = session.state.subscribe(lambda *_: self.wake.set())
        self.thread = threading.Thread(target=self._run, daemon=True, name='animation-state')
        if start: self.thread.start()

    def status(self):
        with self.lock:
            return {'version': 1, 'enabled': self.settings['enabled'], 'settings': dict(self.settings),
                    'revision': self.revision, 'state': self.state.as_dict() if self.state else None,
                    'intent': deepcopy(self.current), 'capabilities': deepcopy(self.capabilities),
                    'pending': self.pending is not None, 'error': self.error, 'movement': deepcopy(self.movement)}

    def _event(self, event):
        if event.type.startswith(('resource.', 'animation.', 'state.')) or event.type in {'voice.level','voice.wake_score','voice.wake_test','chat.delta','model.reasoning'}: return
        if event.type.startswith(('voice.','speech.','audio.','model.','chat.','action.')): self.wake.set()

    def _notify(self):
        key = json.dumps(self.status(), sort_keys=True)
        if key != self.last_status:
            self.last_status = key
            event_bus.publish('animation.changed')

    def report_capabilities(self, bones, expressions):
        if not isinstance(bones, list) or len(bones) > 128 or any(not isinstance(b, str) or b not in BONES for b in bones): raise ValueError('Invalid renderer bones')
        if not isinstance(expressions, list) or len(expressions) > 64 or any(not isinstance(e, str) or not 1 <= len(e) <= 64 for e in expressions): raise ValueError('Invalid renderer expressions')
        with self.lock:
            self.capabilities = {'bones': sorted(set(bones)), 'expressions': sorted(set(expressions)), 'ready': True}
        event_bus.publish('animation.capabilities')
        self.wake.set()

    def interact(self, kind, pointer=None, target=None):
        if kind not in {'hold', 'drag', 'release', 'click', 'pointer', 'leave', 'hover', 'hoverDelayed', 'ragdoll', 'recover'}: raise ValueError('Invalid avatar interaction')
        if target is not None:
            if not isinstance(target, dict) or set(target) != {'bone', 'region'} or not isinstance(target['bone'], str) or not (target['bone'] in BONES or len(target['bone']) <= 512 and re.fullmatch(r'root(?:/\d+){1,128}', target['bone'])) or not isinstance(target['region'], str) or target['region'] not in {'head', 'torso', 'arms', 'legs', 'spring'}:
                raise ValueError('Invalid avatar hit target')
        if pointer is not None:
            if not isinstance(pointer, dict) or set(pointer) != {'x', 'y', 'near'} or type(pointer['near']) is not bool or any(type(pointer[k]) not in (int, float) or not math.isfinite(pointer[k]) or abs(pointer[k]) > 2 for k in ('x', 'y')):
                raise ValueError('Pointer coordinates must be normalized within [-2, 2]')
        cancel_movement = kind in {'hold', 'drag', 'release', 'click'}
        with self.lock:
            now = time.monotonic()
            self.interaction_state['updated'] = now
            if not self.interaction_state['held'] or kind == 'hold': self.interaction_state['target'] = deepcopy(target)
            if kind == 'ragdoll': self.interaction_state['ragdoll'] = True
            if kind == 'recover': self.interaction_state.update(ragdoll=False, held=False, event='settling', until=now + 2)
            if kind in {'hold', 'drag'}: self.interaction_state['held'] = True
            elif kind in {'release', 'click', 'leave'}: self.interaction_state['held'] = False
            if kind in {'release', 'click'}:
                self.interaction_state.update(event='settling' if kind == 'release' else 'clicked', until=now + .8)
            if pointer and self.settings['mouse_tracking']:
                self.interaction_state.update(pointer=deepcopy(pointer), near=pointer['near'])
            if kind == 'leave': self.interaction_state.update(pointer=None, near=False)
        if cancel_movement: self.stop_movement()
        self.wake.set()

    def invalidate_library(self):
        with self.lock:
            self.library_revision += 1
            self.failed_assets.clear()
        event_bus.publish('animation.library_changed')
        self.wake.set()

    def renderer_error(self, action, error):
        with self.lock:
            self.error = str(error)[:2000]
            if action['kind'] == 'motion.base' and action['payload'].get('asset'):
                self.failed_assets.add(action['payload']['asset']['id'])
                self.current = None
                self.last_key = None
        event_bus.publish('animation.error', error=self.error)
        self.wake.set()

    def _choices(self, state):
        return eligible_intents(state, [entry for entry in self.library.list() if entry['id'] not in self.failed_assets])

    def preview(self, identifier):
        entry = self.library.get(identifier)
        self.library.path(identifier)
        with self.lock:
            if not self.capabilities['ready']: raise ValueError('Load an avatar before previewing animation')
            missing = set(entry['mask'] or entry['bones']) - set(self.capabilities['bones'])
            if missing: raise ValueError('Avatar is missing bones: ' + ', '.join(sorted(missing)))
        return self.session.actions.start('motion.preview', {'version': 1, 'asset': entry, 'layer': entry['layer'],
            'procedural': 'idle', 'transition_seconds': entry['transition_seconds'], 'strength': 1.0}, duration=8)

    def walk_to(self, x, y):
        if type(x) is not int or type(y) is not int: raise ValueError('Walking destination must use integer pixels')
        desktop = self.session.state.snapshot()
        geometry = desktop['avatar_geometry']
        display = next((item for item in desktop['displays'] if item['index'] == geometry['screen']), None)
        if display is None: raise ValueError('Connect Electron and select a display before walking')
        with self.lock:
            if self.interaction_state['held']: raise ValueError('User is holding the avatar')
        target = {'x': max(0, min(x, max(0, display['bounds']['width'] - geometry['width']))),
                  'y': max(0, min(y, max(0, display['bounds']['height'] - geometry['height'])))}
        distance = math.hypot(target['x'] - geometry['x'], target['y'] - geometry['y'])
        payload = {'version': 1, 'target': target, 'screen': geometry['screen'], 'speed': self.settings['walk_speed'],
                   'transition_seconds': self.settings['transition_seconds']}
        action = self.session.actions.start('motion.locomotion', payload, duration=distance / self.settings['walk_speed'] * 2 + 2)
        with self.lock: self.movement = {'action_id': action.id, **payload}
        self.wake.set()
        self._notify()
        return action

    def stop_movement(self):
        with self.lock:
            movement, self.movement = self.movement, None
        if movement: self.session.actions.cancel(movement['action_id'])
        self.wake.set()

    def _observe(self):
        now = time.monotonic()
        desktop = self.session.state.snapshot()
        status = self.session.status()  # the session's one reading of busy and idle, shared with wake feedback
        voice = {'listening': status.listening, 'user_speaking': status.user_speaking, 'speaking': status.speaking,
                 'generating': status.generating, 'pending_audio': status.pending_audio > 0}
        with self.lock:
            if now - self.interaction_state['updated'] > 2:
                self.interaction_state.update(held=False, near=False, pointer=None)
            interaction = deepcopy(self.interaction_state)
            moving = self.movement is not None
            caps = deepcopy(self.capabilities)
        event = interaction['event'] if now < interaction['until'] else ''
        mode = ('held' if interaction['held'] else event if event else 'sleeping' if desktop['sleep'] else
                'walking' if moving else status.model_state(tool_running=any(t['status'] == 'running' for t in desktop['tools'])))
        emotion = desktop['emotion'] or {}
        state = AnimationState(0, mode, emotion.get('primary', 'neutral'), emotion.get('intensity', .5),
            geometry=desktop['avatar_geometry'], interaction=interaction, bones=caps['bones'], expressions=caps['expressions'], voice=voice)
        key = (mode, state.emotion, round(state.intensity, 1), interaction['near'], tuple(caps['bones']), tuple(caps['expressions']), self.library_revision)
        return state, key

    def step(self):
        if self.closed.is_set() or not self.settings['enabled']: return
        if self.movement and not any(a['id'] == self.movement['action_id'] for a in self.session.actions.active()):
            with self.lock: self.movement = None
        state, key = self._observe()
        now, apply, submit = time.monotonic(), None, None
        with self.lock:
            changed = key != self.last_key
            urgent = self.current is None or state.mode != (self.state.mode if self.state else '')
            if changed and (urgent or now - self.last_apply >= self.settings['min_dwell_seconds']):
                self.revision += 1
                self.last_key = key
                state.revision = self.revision
                self.state = state
                choices = self._choices(state)
                apply = motion_intent(state, choices, self.settings)
            elif self.state:
                state.revision = self.revision
                self.state = state
            if self.pending:
                job = self.pending
                if now - job['started'] > self.settings['policy_timeout_seconds']: job['expired'] = True
                if job['future'].done():
                    self.pending = None
                    try: result = job['future'].result()
                    except Exception as exc:
                        self.error = f'Julia animation selection unavailable: {exc}'
                        result = None
                    if not job['expired'] and job['state'].revision == self.revision and job['key'] == key and result:
                        apply = motion_intent(job['state'], job['choices'], self.settings, result)
            if self.selector and self.pending is None and self.state and self.last_julia_revision != self.revision and self.state.mode not in {'held', 'clicked', 'settling', 'walking'}:
                choices = self._choices(self.state)
                if len(choices) > 1:
                    submit = (deepcopy(self.state), choices)
                    self.last_julia_revision = self.revision
        if apply: self._apply(apply)
        if submit:
            state, choices = submit
            future = self.executor.submit(self.selector, state.as_dict(), [{'id': c['id'], 'description': c['description']} for c in choices])
            with self.lock: self.pending = {'future': future, 'state': state, 'choices': choices, 'key': self.last_key, 'started': now, 'expired': False}
            future.add_done_callback(lambda _: self.wake.set())
        self._notify()

    def _apply(self, intent):
        update_id = None
        with self.lock:
            if self.closed.is_set() or intent.state_revision != self.revision: return
            if self.current and self.current['intent_id'] == intent.intent_id:
                self.current.update(intent.as_dict())
                update_id = self.current['action_id']
        if update_id:
            self.session.actions.update(update_id, intent.as_dict())
            return
        action = self.session.actions.start('motion.base', intent.as_dict())
        with self.lock:
            stale = self.closed.is_set() or intent.state_revision != self.revision
            if not stale:
                self.current = {**intent.as_dict(), 'action_id': action.id}
                self.last_apply = time.monotonic()
        if stale: self.session.actions.cancel(action.id)
        else: event_bus.publish('animation.state', revision=intent.state_revision, intent=intent.intent_id, source=intent.source)

    def _run(self):
        while not self.closed.is_set():
            self.wake.clear()
            try: self.step()
            except Exception as exc:
                with self.lock: self.error = str(exc)
                self._notify()
            with self.lock:
                now = time.monotonic()
                deadlines = [self.interaction_state['until'], self.interaction_state['updated'] + 2,
                    self.last_apply + self.settings['min_dwell_seconds']]
                if self.pending and not self.pending['expired']:
                    deadlines.append(self.pending['started'] + self.settings['policy_timeout_seconds'])
                future = [deadline for deadline in deadlines if deadline > now]
            self.wake.wait(min(future) - now + .001 if future else None)

    def close(self):
        self.closed.set()
        self.wake.set()
        self.unsubscribe_events()
        self.unsubscribe_state()
        self.stop_movement()
        self.executor.shutdown()
        for action in self.session.actions.active():
            if action['kind'].startswith('motion.'): self.session.actions.cancel(action['id'])
