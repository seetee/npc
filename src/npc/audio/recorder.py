"""Microphone capture. 16 kHz mono int16 — feeds faster-whisper directly.

Two recorders behind one protocol: PushToTalkRecorder records between
start() (key down) and stop() (key up); VadRecorder starts on a tap and
fires `on_auto_stop(clip)` itself once trailing silence (or a max-duration
safety cap) ends the utterance. Nothing downstream cares which one runs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

SAMPLE_RATE = 16_000


@dataclass
class AudioClip:
    samples: np.ndarray  # int16 mono
    sample_rate: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def to_float32(self) -> np.ndarray:
        return self.samples.astype(np.float32) / 32768.0

    def dbfs(self) -> float:
        """RMS level in dB relative to int16 full scale (0 = max, -inf = silence)."""
        if len(self.samples) == 0:
            return float("-inf")
        rms = float(np.sqrt(np.mean(self.to_float32() ** 2)))
        return 20 * float(np.log10(rms)) if rms > 0 else float("-inf")


class Recorder(Protocol):
    on_auto_stop: Callable[[AudioClip], None] | None

    def start(self) -> None: ...
    def stop(self) -> AudioClip: ...


class PushToTalkRecorder:
    """v1: records between start() (key down) and stop() (key up)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: int | str | None = None):
        self.sample_rate = sample_rate
        self.device = device
        self.on_auto_stop: Callable[[AudioClip], None] | None = None
        self._blocks: list[np.ndarray] = []
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._blocks = []

        def callback(indata, frames, time_info, status):
            self._blocks.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> AudioClip:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._blocks:
            samples = np.concatenate(self._blocks).reshape(-1)
        else:
            samples = np.zeros(0, dtype=np.int16)
        self._blocks = []
        return AudioClip(samples=samples, sample_rate=self.sample_rate)


class SilenceTracker:
    """Pure per-block state machine deciding when a tap-to-talk recording ends.

    Feed each ~30 ms block's dBFS: returns None to keep going, "silence" once
    speech has been heard and `silence_blocks` consecutive quiet blocks follow,
    or "max-duration" after `max_blocks` total (even if speech never came —
    the whisper-side guards absorb a silent clip). Block count is time, so
    tests need no clock."""

    def __init__(self, threshold_db: float, silence_blocks: int, max_blocks: int):
        self.threshold_db = threshold_db
        self.silence_blocks = silence_blocks
        self.max_blocks = max_blocks
        self._blocks = 0
        self._quiet_run = 0
        self._heard_speech = False

    def feed(self, block_dbfs: float) -> str | None:
        self._blocks += 1
        if self._blocks >= self.max_blocks:
            return "max-duration"
        if block_dbfs >= self.threshold_db:
            self._heard_speech = True
            self._quiet_run = 0
        elif self._heard_speech:
            self._quiet_run += 1
            if self._quiet_run >= self.silence_blocks:
                return "silence"
        return None


class VadRecorder:
    """v2 tap-to-talk: start() on a tap; trailing silence (or the max-duration
    cap) ends the recording and fires on_auto_stop(clip).

    Thread story: the sounddevice callback only appends blocks and feeds the
    SilenceTracker — PortAudio forbids closing a stream from its own callback —
    and sets an event on a stop verdict. A finalizer thread waits on that
    event and closes/concatenates under the lock; stop() (second tap or
    shutdown) does the same and is idempotent. Guarantees: on_auto_stop fires
    at most once, never after stop() has returned the clip, and always
    outside the lock."""

    is_auto_stop = True  # lets the app render "pause to send" instead of "release"

    def __init__(self, *, threshold_db: float, silence_seconds: float = 1.2,
                 max_seconds: float = 30.0, sample_rate: int = SAMPLE_RATE,
                 device: int | str | None = None, block_ms: int = 30):
        self.sample_rate = sample_rate
        self.device = device
        self.threshold_db = threshold_db
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds
        self.block_size = max(1, int(sample_rate * block_ms / 1000))
        self.on_auto_stop: Callable[[AudioClip], None] | None = None
        self._blocks: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()
        self._verdict = threading.Event()
        self._finalized = False

    def start(self) -> None:
        import sounddevice as sd

        blocks_per_second = self.sample_rate / self.block_size
        tracker = SilenceTracker(
            self.threshold_db,
            silence_blocks=max(1, round(self.silence_seconds * blocks_per_second)),
            max_blocks=max(1, round(self.max_seconds * blocks_per_second)),
        )
        self._blocks = []
        self._verdict.clear()
        self._finalized = False

        def callback(indata, frames, time_info, status):
            block = indata.copy()
            self._blocks.append(block)
            clip = AudioClip(samples=block.reshape(-1), sample_rate=self.sample_rate)
            if tracker.feed(clip.dbfs()) is not None:
                self._verdict.set()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
            blocksize=self.block_size,
            callback=callback,
        )
        self._stream.start()
        threading.Thread(target=self._await_verdict, daemon=True,
                         name="vad-finalizer").start()

    def _await_verdict(self) -> None:
        self._verdict.wait()
        with self._lock:
            if self._finalized:
                return  # a manual stop() won the race
            clip = self._finalize()
        if self.on_auto_stop is not None:
            self.on_auto_stop(clip)

    def _finalize(self) -> AudioClip:
        """Close the stream and build the clip. Call with self._lock held."""
        self._finalized = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._blocks:
            samples = np.concatenate(self._blocks).reshape(-1)
        else:
            samples = np.zeros(0, dtype=np.int16)
        self._blocks = []
        return AudioClip(samples=samples, sample_rate=self.sample_rate)

    def stop(self) -> AudioClip:
        """Manual stop (second tap / shutdown); empty clip if VAD already won."""
        with self._lock:
            if self._finalized:
                clip = AudioClip(np.zeros(0, dtype=np.int16), self.sample_rate)
            else:
                clip = self._finalize()
        self._verdict.set()  # release the finalizer thread
        return clip
