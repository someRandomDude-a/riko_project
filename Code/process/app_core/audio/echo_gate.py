"""Half-duplex barge-in (voice.barge_in): whether microphone speech counts while Riko's own voice plays.

This is not echo cancellation. Through speakers her echo reaches the microphone as speech, so in full_duplex (the
default) it can interrupt her and be saved as the user's words. echo_gate compares each microphone frame's level with
the level of the audio SpeechQueue wrote around that moment (a Geigel-style double-talk test: the user counts only when
clearly louder than her echo can be); stop_only ignores the microphone while she plays. A frame judged to be her echo
reaches the wake detector and the segmenter as silence, so her words never become an utterance's audio or pre-roll.

echo_gate learns how loud her echo is afresh for every reply (and every wake cue), from its first seconds of her
speech, and acts as stop_only until it knows (or until her first second shows the echo is no louder than in her last
reply): a change of speakers, headphones or volume between replies is picked up by the next one. A change in the middle
of a reply is not (pin voice.echo_coupling high enough, or use stop_only).

The proper fix is acoustic echo cancellation with the exact PCM SpeechQueue writes as the far-end reference (CLAUDE.md,
Voice pipeline); this gate would stay behind it as a residual-echo guard.
"""
from collections import deque
import logging
from statistics import median
import threading
import time

logger = logging.getLogger(__name__)
LEAD = .05  # s: a microphone frame may be stamped this long before a block written just after it starts playing
REACH = .12  # s: a frame's audio is up to this old when stamped (the frame itself, input latency, scheduling)
FLOOR = 1e-4  # RMS: quieter playback around a frame is silence, not something to gate
LEARN_FLOOR = .003  # RMS (-50 dBFS): her level must be at least this to learn her echo level from a frame
FADE = 1.5  # after a block has certainly played, her level counts 30 dB (10**-1.5) lower by the end of echo_tail_seconds
LEARN_FRAMES = 64  # frames of her playback (about 2 s of her speech) each reply learns her echo level from
CHECK_FRAMES, SAME = 32, 1.25  # after 32 frames a reply whose echo's median is no louder than the last reply's uses its level
SPREAD = 2.0  # her echo level: the 90th percentile of mic/playback, at most twice the median (a user talking over part of it)
# The user counts at MARGIN times her echo level. Chosen with a model of SpeechQueue's blocking writes into an output
# buffer, real Silero VAD and synthetic or macOS `say` speech through rooms of RT60 0-0.6 s, couplings 0.1-1 and output
# latency 0.02-0.2 s, then checked in real time through the real SpeechQueue, VoiceInput, VoiceSegments and Interjections:
# with the latency reported correctly one echo frame passed in 140 runs and none interrupted; a user about 4 times louder
# than her echo (RMS while each is audible) interrupted in two thirds of runs after 3.2-5 s of speech, 6 times louder in
# all, after 2.2-3 s (with headphones, 1.9 s, against 1.6 s in full_duplex). Latency under-reported by 0.18 s (Bluetooth)
# let up to 9 frames per reply through in a third of runs, none interrupting (in real time none was transcribed:
# voice_input.MIN_VOICED); raising the volume 6 dB or more, or moving from headphones to speakers, in the middle of a reply
# lets her echo through until her next reply (10 dB interrupted her).
MARGIN = 2.0


def pcm16_rms(pcm):
    """RMS of little-endian PCM16 bytes, full scale 1.0."""
    import numpy as np
    samples = np.frombuffer(pcm, dtype='<i2').astype('float32') / 32768
    return float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0


def float_rms(samples):
    import numpy as np
    samples = np.asarray(samples, dtype='float32')
    return float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0


def echo_level(ratios):
    """Her echo's level at its loud end from microphone/playback ratios: the 90th percentile, at most SPREAD times the
    median, so a user talking over less than half of the frames cannot inflate it."""
    import numpy as np
    return float(min(np.percentile(ratios, 90), SPREAD * np.median(ratios)))


class PlaybackLevel:
    """The level of each block SpeechQueue writes, over the time it can be heard: from the write until it has certainly
    played (latency + its duration), plus its echo's tail. Written by the sentence-playback thread and read by voice-vad;
    its lock guards the list only (nothing is published or called under it). run names the reply (its turn id) or wake
    cue the block belongs to; coupling and typical are the echo level the last reply learned and its median ratio."""
    def __init__(self, keep=256):  # several seconds of blocks: more than any frame looks back
        self._lock = threading.Lock()
        self._blocks = deque(maxlen=keep)
        self.coupling = self.typical = None

    def wrote(self, rms, seconds, latency=0.0, now=None, run=None):
        now = time.monotonic() if now is None else now
        with self._lock: self._blocks.append((now, now + max(0.0, latency or 0.0) + seconds, rms, run))

    def levels(self, at, tail):
        """(level, loudest, run) for a microphone frame stamped at. loudest: the loudest block that may be heard or still
        echo in it (else 0: nothing to gate); level: her level to compare the frame with, each block counting in full from
        its write until it has certainly played and fading after; run: the run of the newest such block."""
        level = loudest = 0.0
        run = None
        with self._lock:
            for written, ends, rms, block_run in self._blocks:
                if written - LEAD > at or ends + tail + REACH < at: continue
                late = at - ends - REACH
                loudest, run = max(loudest, rms), block_run
                level = max(level, rms if late <= 0 else rms * 10 ** (-FADE * late / max(tail, .05)))
        return level, loudest, run


class EchoGate:
    """One per VoiceInput: decide(probability, rms, at) -> (speaking, echo) for each microphone frame."""
    def __init__(self, voice, playback):
        self.voice, self.playback = voice, playback  # playback is None in full_duplex: nothing is gated
        self.threshold = max(voice.vad_threshold, voice.barge_in_vad_threshold)
        self.run, self.ratios, self.learned = object(), [], None

    def coupling(self):
        """voice.echo_coupling, else her echo level (microphone RMS / her level) learned from this reply's first 64 frames
        of her speech; meanwhile the last reply's level once 32 frames show this echo is no louder (by their median, which a
        user talking over part of them does not move), else None (learning)."""
        if self.voice.echo_coupling is not None: return self.voice.echo_coupling
        if self.learned is not None: return self.learned
        previous, typical = self.playback.coupling, self.playback.typical
        if len(self.ratios) >= LEARN_FRAMES:
            self.learned, self.playback.typical = echo_level(self.ratios), median(self.ratios)
            self.playback.coupling = self.learned
            changed = previous is None or not previous / SAME <= self.learned <= previous * SAME
            logger.log(logging.INFO if changed else logging.DEBUG, 'Echo level learned for this reply: %.3g (pin it as voice.echo_coupling to keep it; voice.barge_in: echo_gate)', self.learned)
            return self.learned
        if previous is not None and typical is not None and len(self.ratios) >= CHECK_FRAMES and median(self.ratios) <= SAME * typical: return previous
        return None

    def decide(self, probability, rms, at):
        heard = probability >= self.voice.vad_threshold
        if self.playback is None: return heard, False
        level, loudest, run = self.playback.levels(at, self.voice.echo_tail_seconds)
        if loudest < FLOOR: return heard, False
        if self.voice.barge_in == 'stop_only': return False, True
        if run != self.run: self.run, self.ratios, self.learned = run, [], None  # her next reply or cue: learn its echo afresh
        if level >= LEARN_FLOOR and self.learned is None and self.voice.echo_coupling is None: self.ratios.append(rms / level)
        coupling = self.coupling()
        if coupling is None: return False, True  # still learning how loud her echo is: as stop_only, never a false interruption
        speaking = probability >= self.threshold and rms >= MARGIN * coupling * (level if level >= FLOOR else loudest)
        return speaking, not speaking
