"""Local text-to-speech via Piper."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .audio.player import AudioPlayer


class Speaker(Protocol):
    def say(self, text: str) -> None: ...
    def stop(self) -> None: ...


def download_hint(voice: str, voices_dir: Path) -> str:
    return (f"uv run python -m piper.download_voices {voice} "
            f"--data-dir {voices_dir}")


class PiperSpeaker:
    def __init__(self, voice_path: Path, player: AudioPlayer | None = None):
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(voice_path))
        self._player = player or AudioPlayer()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        chunks = []
        sample_rate = 22_050
        for chunk in self._voice.synthesize(text):
            sample_rate = chunk.sample_rate
            if hasattr(chunk, "audio_int16_array"):
                chunks.append(np.asarray(chunk.audio_int16_array, dtype=np.int16))
            else:
                chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if not chunks:
            return np.zeros(0, dtype=np.int16), sample_rate
        return np.concatenate(chunks), sample_rate

    def say(self, text: str) -> None:
        samples, sample_rate = self.synthesize(text)
        if len(samples):
            self._player.play(samples, sample_rate)

    def stop(self) -> None:
        self._player.stop()
