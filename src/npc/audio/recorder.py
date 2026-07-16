"""Microphone capture. 16 kHz mono int16 — feeds faster-whisper directly.

The Recorder protocol is the v2 seam: a future VadRecorder (tap to start,
voice-activity detection stops) implements the same interface and fires
`on_auto_stop` itself; nothing downstream changes.
"""

from __future__ import annotations

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
