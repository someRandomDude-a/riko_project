"""The foreground turn's shared vocabulary. TurnGate lets one turn run at a time; RuntimeStatus is one reading of what the
session is doing (SessionManager.status()), and its predicates are the only definition of busy and idle, so the avatar,
wake feedback, initiative, the emotion probe, wake calibration and barge-in always agree. TurnContext is what one
ChatService.respond call learns from the turn running it."""
from dataclasses import dataclass, field
import threading
from typing import Callable

from .cancellation import TurnBusy, TurnCancelled


class TurnGate:
    """One foreground turn at a time (SessionManager.respond and present_initiative). begin() takes the turn or raises
    TurnBusy; given wait, a callable polled while the running turn finishes, it waits instead and raises TurnCancelled
    once wait() is true. Each begin(), and each try_begin() that returned True, is paired with one end()."""
    def __init__(self, poll=0.1): self._lock, self._poll = threading.Lock(), poll

    def begin(self, wait=None):
        if wait is None:
            if not self._lock.acquire(blocking=False): raise TurnBusy()
            return
        while not self._lock.acquire(timeout=self._poll):
            if wait(): raise TurnCancelled()

    def try_begin(self):
        """Take the turn only if it is free now: a background proposal never waits."""
        return self._lock.acquire(blocking=False)

    def end(self): self._lock.release()

    @property
    def busy(self): return self._lock.locked()


@dataclass(frozen=True)
class RuntimeStatus:
    closed: bool = False
    active_turn: str | None = None  # the latest foreground turn, which may already have finished
    generating: bool = False  # a foreground reply is being generated
    speaking: bool = False  # one of its sentences is playing now
    pending_audio: int = 0  # its sentences queued or playing
    spoken_here: bool = True  # False for a remote (speak=False) turn such as Discord's: nothing plays here
    user_speaking: bool = False  # voice activity on the open microphone
    listening: bool = False  # capture running and ready, with the microphone enabled
    capture_running: bool = False
    voice_status: str = 'stopped'
    voice_phase: str = 'stopped'
    sleeping: bool = False
    calibrating: bool = False  # wake-word calibration or a detector test owns the microphone
    speaking_priority: bool = False  # interrupt_user's window is open
    interaction_revision: int = 0  # bumped by every foreground input; a background check that saw it change backs off

    def replying(self):
        """A reply is in flight: generating, playing or with sentences queued (the gap before the first one plays). Wake
        calibration and detector tests wait for it to end, so Riko's own voice never becomes a sample."""
        return self.generating or self.speaking or self.pending_audio > 0

    def local_reply(self):
        """A reply spoken here is in flight, so speech on the microphone talks over it (SessionManager.voice_anchor)."""
        return self.spoken_here and self.replying()

    def is_quiet(self):
        """No reply in flight and no user speech (the emotion probe trains only then: provider.set_expression_idle)."""
        return not (self.replying() or self.user_speaking)

    def idle_for_background(self):
        """A background proposal (initiative) may run and be presented now."""
        return self.is_quiet() and not (self.closed or self.sleeping or self.calibrating)

    def model_state(self, *, tool_running=False):
        """What the avatar shows, shared by animation and wake feedback: sleeping, speaking, tool, thinking (generating or
        sentences queued), listening (user speech) or idle."""
        if self.sleeping: return 'sleeping'
        if self.speaking: return 'speaking'
        if tool_running: return 'tool'
        return 'thinking' if self.generating or self.pending_audio > 0 else 'listening' if self.user_speaking else 'idle'


@dataclass(frozen=True)
class TurnContext:
    """Per-turn data SessionManager passes to ChatService.respond, instead of attributes set on the service before each turn.
    A plain call (tests, a tool-free script) passes none and gets no origin, runtime observation or memory context."""
    origin: dict = field(default_factory=dict)  # source (message, microphone, discord) and conversation_id, kept on the user message
    runtime: Callable[[], dict] | None = None  # the runtime observation, re-read before each model request of the tool loop
    memory_runtime: Callable[[], dict] | None = None  # added to the context a memory capture records
    emotion_from_playback: bool = False  # the caller feeds Julia the sentences as they play, so respond does not feed it the stream
