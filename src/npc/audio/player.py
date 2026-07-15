"""Speaker playback with barge-in support (stop() from another thread)."""

from __future__ import annotations

import numpy as np


class AudioPlayer:
    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Blocks until playback finishes or stop() is called."""
        import sounddevice as sd

        sd.play(samples, samplerate=sample_rate)
        sd.wait()

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()
