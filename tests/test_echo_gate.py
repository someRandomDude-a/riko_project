"""voice.barge_in (audio/echo_gate.py): while Riko's own voice plays, full_duplex counts every speech frame (her echo too),
stop_only none, and echo_gate only frames clearly louder than her echo, whose level each reply learns afresh."""
import logging

import pytest

from process.app_core.audio import echo_gate
from process.app_core.audio.echo_gate import (CHECK_FRAMES, FADE, FLOOR, LEAD, LEARN_FRAMES, MARGIN, REACH, SAME, EchoGate,
    PlaybackLevel, echo_level, float_rms, pcm16_rms)
from process.app_core.kernel.audio_config import VoiceConfig


def voice(**values): return VoiceConfig.from_raw(values)


def playing(rms=.2, start=10.0, seconds=30.0, run='turn-1', level=None):
    level = level or PlaybackLevel()
    level.wrote(rms, seconds, now=start, run=run)
    return level


def test_levels_are_pcm_rms_at_full_scale():
    assert pcm16_rms((16384).to_bytes(2, 'little', signed=True) * 4) == pytest.approx(.5) and pcm16_rms(b'') == 0
    assert float_rms([[.25, -.25]]) == pytest.approx(.25)


def test_a_block_counts_in_full_from_its_write_until_it_has_certainly_played_then_fades_over_the_tail():
    level = PlaybackLevel(keep=3)
    level.wrote(.2, .1, latency=.05, now=10, run='a')  # certainly played by 10.15
    assert level.levels(10 - LEAD - .005, .3) == (0, 0, None)  # stamped before it could sound
    assert level.levels(10 - LEAD + .005, .3) == (.2, .2, 'a') and level.levels(10.15 + REACH, .3)[:2] == (.2, .2)
    faded, loudest, _ = level.levels(10.15 + REACH + .29, .3)  # near the end of its echo's tail: almost 30 dB down, still gated
    assert loudest == .2 and faded == pytest.approx(.2 * 10 ** (-FADE * .29 / .3))
    assert level.levels(10.15 + REACH + .305, .3) == (0, 0, None)
    level.wrote(.5, .1, now=10.1, run='b')
    assert level.levels(10.12, .3) == (.5, .5, 'b')  # the loudest block that may be sounding; the newest block's run
    for start in (20, 21, 22): level.wrote(.1, .1, now=start)
    assert level.levels(10.12, .3) == (0, 0, None)  # only the newest blocks are kept
    level.wrote(.3, .1, latency=None, now=30)
    assert level.levels(30.2, 0)[0] == .3


def test_echo_level_is_a_high_percentile_capped_by_the_median():
    assert echo_level([.1] * 9 + [.2]) == pytest.approx(.11)  # the 90th percentile of her echo
    assert echo_level([.1] * 6 + [5.0] * 4) == pytest.approx(.2)  # a user talking over 40% of it: twice the median


def test_full_duplex_never_looks_at_playback():
    gate = EchoGate(voice(), None)
    assert gate.decide(.6, .001, 10) == (True, False) and gate.decide(.4, .9, 10) == (False, False)


def test_stop_only_ignores_the_microphone_while_she_plays_and_for_the_tail_after():
    level = playing(seconds=1)  # sounding until 11.0
    gate = EchoGate(voice(barge_in='stop_only', echo_tail_seconds=.3), level)
    assert gate.decide(.99, .9, 10.5) == (False, True) and gate.decide(.99, .9, 11.0 + .3 + REACH - .005) == (False, True)
    assert gate.decide(.99, .9, 11.0 + .3 + REACH + .005) == (True, False)
    assert gate.decide(.4, .9, 12) == (False, False)
    quiet = PlaybackLevel()
    quiet.wrote(FLOOR / 2, 5, now=10)  # silence in her audio is not gated
    assert EchoGate(voice(barge_in='stop_only'), quiet).decide(.9, .1, 11) == (True, False)


def learned(gate, ratio=.1, far=.2, at=10.5):
    for _ in range(LEARN_FRAMES): gate.decide(.95, ratio * far, at)
    return gate


def test_echo_gate_learns_her_echo_level_first_acting_as_stop_only_meanwhile(caplog):
    level = playing()
    gate = EchoGate(voice(barge_in='echo_gate'), level)
    for _ in range(LEARN_FRAMES - 1): assert gate.decide(.99, .9, 10.5) == (False, True)  # even a shout, until it knows
    assert level.coupling is None
    with caplog.at_level(logging.INFO, logger=echo_gate.__name__): gate.decide(.95, .02, 10.5)
    assert level.coupling is not None and 'Echo level learned for this reply' in caplog.text


def test_echo_gate_passes_only_speech_clearly_louder_than_her_echo():
    level = playing(rms=.2)
    gate = learned(EchoGate(voice(barge_in='echo_gate'), level), ratio=.1)
    assert level.coupling == pytest.approx(.1)
    loud = .1 * .2  # her echo's level at the microphone
    assert gate.decide(.99, (MARGIN - .2) * loud, 11) == (False, True)  # an echo peak
    assert gate.decide(.95, (MARGIN + .2) * loud, 11) == (True, False)  # the user, speaking up
    assert gate.decide(.8, 10 * loud, 11) == (False, True)  # loud, but not clearly speech (barge_in_vad_threshold .85)
    assert gate.decide(.99, 10 * loud, 50) == (True, False) and gate.decide(.4, 10 * loud, 50) == (False, False)  # she stopped


def test_her_level_fades_after_a_block_so_her_reverberation_is_still_compared_with_her_last_words():
    level = PlaybackLevel()
    level.wrote(.2, 1, now=10, run='turn')  # certainly played by 11
    gate = EchoGate(voice(barge_in='echo_gate', echo_tail_seconds=.3, echo_coupling=.1), level)
    at = 11 + REACH + .15  # halfway through her tail: her level counts 15 dB down
    far = .2 * 10 ** (-FADE / 2)
    assert gate.decide(.95, MARGIN * .1 * far * .9, at) == (False, True) and gate.decide(.95, MARGIN * .1 * far * 1.1, at) == (True, False)


def test_the_speech_certainty_over_playback_is_never_below_the_usual_threshold():
    level = playing()
    gate = EchoGate(voice(barge_in='echo_gate', vad_threshold=.7, barge_in_vad_threshold=.3, echo_coupling=.1), level)
    assert gate.threshold == .7 and gate.decide(.6, 1, 11) == (False, True) and gate.decide(.7, 1, 11) == (True, False)


def test_a_pinned_echo_level_needs_no_learning():
    level = playing(rms=.2)
    gate = EchoGate(voice(barge_in='echo_gate', echo_coupling=.05), level)
    assert gate.decide(.95, MARGIN * .05 * .2 + .001, 10.5) == (True, False) and gate.decide(.95, .005, 10.5) == (False, True)
    assert level.coupling is None and not gate.ratios


def test_a_user_talking_over_part_of_the_learning_does_not_lock_them_out():
    level = playing(rms=.2)
    gate = EchoGate(voice(barge_in='echo_gate'), level)
    for index in range(LEARN_FRAMES): gate.decide(.95, (1.0 if index % 10 < 3 else .1) * .2, 10.5)  # 30% the user, ten times louder
    assert level.coupling == pytest.approx(.2)  # twice the median at most
    assert gate.decide(.95, .2, 11) == (True, False)  # the user, as loud as they were, now counts


def test_each_reply_learns_its_echo_afresh_so_a_change_of_speakers_between_replies_never_interrupts_her():
    level = playing(rms=.2, start=10, seconds=5, run='turn-1')
    gate = learned(EchoGate(voice(barge_in='echo_gate'), level), ratio=.01, at=11)  # headphones: almost no echo
    assert level.coupling == pytest.approx(.01) and gate.decide(.95, .05, 12) == (True, False)
    playing(rms=.2, start=20, seconds=5, run='turn-2', level=level)  # her next reply, through speakers
    for _ in range(LEARN_FRAMES - 1): assert gate.decide(.95, .3 * .2, 21) == (False, True)  # her echo: learning again
    gate.decide(.95, .3 * .2, 21)
    assert level.coupling == pytest.approx(.3)
    assert gate.decide(.99, .3 * .2 * (MARGIN - .2), 22) == (False, True) and gate.decide(.99, .3 * .2 * (MARGIN + .2), 22) == (True, False)


def test_a_reply_whose_echo_is_no_louder_than_the_last_uses_its_level_while_learning_and_a_louder_one_waits():
    level = playing(rms=.2, start=10, seconds=5, run='turn-1')
    learned(EchoGate(voice(barge_in='echo_gate'), level), ratio=.1, at=11)
    assert (level.coupling, level.typical) == (pytest.approx(.1), pytest.approx(.1))
    playing(rms=.2, start=20, seconds=5, run='turn-2', level=level)
    same = EchoGate(voice(barge_in='echo_gate'), level)
    for index in range(CHECK_FRAMES - 1):  # her echo as before, with the user talking over a third of it
        assert same.decide(.95, (.5 if index % 3 == 0 else .1) * .2, 21) == (False, True)
    same.decide(.95, .1 * .2, 21)
    assert same.decide(.95, .1 * .2 * (MARGIN + .2), 21) == (True, False)  # the last reply's level, before 64 frames
    playing(rms=.2, start=30, seconds=5, run='turn-3', level=level)
    louder = EchoGate(voice(barge_in='echo_gate'), level)
    for _ in range(LEARN_FRAMES - 1): assert louder.decide(.95, .1 * SAME * 1.2 * .2, 31) == (False, True)  # her echo grew


def test_a_wake_cue_is_its_own_run():
    level = playing(rms=.2, start=10, seconds=5, run='turn-1')
    gate = learned(EchoGate(voice(barge_in='echo_gate'), level), ratio=.1, at=11)
    level.wrote(.2, .5, now=20, run=object())  # a cue: the next frames start learning again
    assert gate.decide(.99, .9, 20.1) == (False, True) and len(gate.ratios) == 1 and gate.learned is None


def test_too_quiet_playback_teaches_nothing():
    level = playing(rms=.002)  # below LEARN_FLOOR: microphone noise would dominate
    gate = EchoGate(voice(barge_in='echo_gate'), level)
    for _ in range(LEARN_FRAMES * 2): assert gate.decide(.95, .01, 10.5) == (False, True)
    assert level.coupling is None and not gate.ratios


@pytest.mark.parametrize('values, message', [({'barge_in': 'half'}, 'voice.barge_in must be one of full_duplex, echo_gate, stop_only'),
    ({'barge_in_vad_threshold': 1.5}, 'voice.barge_in_vad_threshold'), ({'echo_coupling': 0}, 'voice.echo_coupling must be a finite number above 0'),
    ({'echo_coupling': 101}, 'at most 100'), ({'echo_tail_seconds': -1}, 'voice.echo_tail_seconds'), ({'echo_tail_seconds': 2.5}, 'at most 2')])
def test_barge_in_settings_are_checked_by_name(values, message):
    with pytest.raises(ValueError, match=message): VoiceConfig.from_raw(values)


def test_barge_in_defaults_keep_full_duplex_and_accept_an_empty_echo_level():
    assert (VoiceConfig().barge_in, VoiceConfig().echo_coupling) == ('full_duplex', None)
    loaded = VoiceConfig.from_raw({'barge_in': 'echo_gate', 'echo_coupling': None, 'echo_tail_seconds': '0.5'})
    assert (loaded.barge_in, loaded.echo_coupling, loaded.echo_tail_seconds) == ('echo_gate', None, .5)
    assert VoiceConfig.from_raw({'echo_coupling': 2}).echo_coupling == 2.0
