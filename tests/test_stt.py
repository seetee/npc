"""Hallucination guard: pure functions, no whisper model needed."""

from types import SimpleNamespace

import numpy as np
import pytest

from npc.audio.recorder import AudioClip
from npc.stt import join_segments, looks_like_hallucination


@pytest.mark.parametrize("text", [
    "Thanks for watching!",
    "Thank you for watching.",
    "Please subscribe!",
    "See you in the next video.",
    "Tack för att du tittade.",
    "Tack för att ni tittade!",
    "Vi ses i nästa video!",
    "Undertexter av amara.org",
    "Undertexter från BTI Studios",
    "Svensktextning.nu",
    "Thanks for watching! Please subscribe!",   # several phantoms in one clip
])
def test_pure_phantom_transcripts_are_flagged(text):
    assert looks_like_hallucination(text)


@pytest.mark.parametrize("text", [
    "Who are you?",
    "Vad kostar svärdet?",
    "Thanks for watching my back in that fight.",
    "Tack för att du tittade efter min häst.",
    "I subscribe to the old ways.",
    "Undertexter av Karin.",   # phantom prefix, but a name survives — keep it
    "",                        # empty is 'heard nothing', not a hallucination
])
def test_real_speech_is_never_flagged(text):
    assert not looks_like_hallucination(text)


def seg(text, no_speech_prob=0.0):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob)


def test_join_segments_drops_probable_non_speech():
    segments = [seg(" Hello there. "), seg("breathing", 0.95), seg("Well met.", 0.2)]
    assert join_segments(segments) == "Hello there. Well met."


def test_join_segments_all_doubtful_yields_empty():
    assert join_segments([seg("hmm", 0.7), seg("wind", 0.99)]) == ""


def test_dbfs_levels():
    silence = AudioClip(np.zeros(16000, dtype=np.int16))
    assert silence.dbfs() == float("-inf")
    assert AudioClip(np.zeros(0, dtype=np.int16)).dbfs() == float("-inf")

    half_scale = AudioClip(np.full(16000, 16384, dtype=np.int16))
    assert half_scale.dbfs() == pytest.approx(-6.0, abs=0.1)

    quiet = AudioClip((np.random.default_rng(0).normal(0, 30, 16000)).astype(np.int16))
    assert quiet.dbfs() < -45.0
