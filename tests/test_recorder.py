"""SilenceTracker + VadRecorder: tap-to-talk auto-stop, no audio hardware."""

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from npc.audio.recorder import AudioClip, SilenceTracker, VadRecorder

LOUD = -20.0
QUIET = -60.0
THRESHOLD = -45.0


def test_silence_after_speech_stops():
    tracker = SilenceTracker(THRESHOLD, silence_blocks=3, max_blocks=100)
    verdicts = [tracker.feed(db) for db in (LOUD, LOUD, QUIET, QUIET, QUIET)]
    assert verdicts == [None, None, None, None, "silence"]


def test_leading_silence_alone_never_stops_before_max():
    tracker = SilenceTracker(THRESHOLD, silence_blocks=3, max_blocks=10)
    verdicts = [tracker.feed(QUIET) for _ in range(10)]
    assert verdicts == [None] * 9 + ["max-duration"]


def test_speech_resets_the_quiet_run():
    tracker = SilenceTracker(THRESHOLD, silence_blocks=2, max_blocks=100)
    verdicts = [tracker.feed(db) for db in (LOUD, QUIET, LOUD, QUIET, QUIET)]
    assert verdicts == [None] * 4 + ["silence"]


def test_max_duration_wins_even_mid_speech():
    tracker = SilenceTracker(THRESHOLD, silence_blocks=100, max_blocks=3)
    assert [tracker.feed(LOUD) for _ in range(3)] == [None, None, "max-duration"]


# ---------- VadRecorder with a fake sounddevice ----------

class FakeStream:
    def __init__(self, samplerate, channels, dtype, device, blocksize, callback):
        self.callback = callback
        self.closed = False

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        self.closed = True


def make_recorder(monkeypatch):
    streams = []

    def input_stream(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setitem(sys.modules, "sounddevice",
                        SimpleNamespace(InputStream=input_stream))
    # block_ms=30 → 480 samples/block; silence_seconds=0.06 → 2 quiet blocks stop
    recorder = VadRecorder(threshold_db=THRESHOLD, silence_seconds=0.06,
                           max_seconds=10.0, block_ms=30)
    return recorder, streams


LOUD_BLOCK = np.full((480, 1), 8000, dtype=np.int16)
QUIET_BLOCK = np.zeros((480, 1), dtype=np.int16)


def test_vad_recorder_fires_on_auto_stop_exactly_once(monkeypatch):
    recorder, streams = make_recorder(monkeypatch)
    clips = []
    fired = threading.Event()

    def on_auto(clip):
        clips.append(clip)
        fired.set()

    recorder.on_auto_stop = on_auto
    recorder.start()
    callback = streams[0].callback
    callback(LOUD_BLOCK, 480, None, None)
    callback(QUIET_BLOCK, 480, None, None)
    callback(QUIET_BLOCK, 480, None, None)   # 2nd quiet block → verdict

    assert fired.wait(timeout=2)
    assert len(clips) == 1
    assert clips[0].duration == pytest.approx(3 * 480 / 16000)
    assert streams[0].closed

    # a manual stop afterwards returns an empty clip and never re-fires
    assert recorder.stop().duration == 0
    assert len(clips) == 1


def test_manual_stop_wins_and_auto_never_fires(monkeypatch):
    recorder, streams = make_recorder(monkeypatch)
    fired = []
    recorder.on_auto_stop = fired.append
    recorder.start()
    streams[0].callback(LOUD_BLOCK, 480, None, None)

    clip = recorder.stop()
    assert isinstance(clip, AudioClip)
    assert clip.duration == pytest.approx(480 / 16000)
    assert streams[0].closed

    # the finalizer thread was released by stop() and must not call back
    for thread in threading.enumerate():
        if thread.name == "vad-finalizer":
            thread.join(timeout=2)
    assert fired == []
