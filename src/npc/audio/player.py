"""Speaker playback with barge-in support (stop() from another thread).

One persistent OutputStream is reused for every clip. The ALSA PipeWire
plugin can double-free inside snd_pcm_close (glibc aborts with "free():
invalid pointer" — seen in a real table-session core dump), and
sounddevice's play() convenience function closes and reopens a stream on
every call, which in the streaming pipeline meant several closes per reply.
Opening the PCM once and closing it only at process exit removes that
exposure. stop() only sets a flag; every PortAudio call stays on the
writing thread.
"""

from __future__ import annotations

import threading

import numpy as np

BLOCK_FRAMES = 2048  # ~93 ms at 22.05 kHz — the barge-in reaction granularity


class AudioPlayer:
    def __init__(self):
        self._stream = None
        self._rate: int | None = None
        self._abort = threading.Event()

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Blocks until playback finishes or stop() is called."""
        import sounddevice as sd

        self._abort.clear()
        if self._stream is not None and self._rate != sample_rate:
            self._stream.close()  # voice change — the only mid-session close
            self._stream = None
        if self._stream is None:
            self._stream = sd.OutputStream(samplerate=sample_rate, channels=1,
                                           dtype="int16")
            self._rate = sample_rate
        stream = self._stream
        if not stream.active:
            stream.start()
        for start in range(0, len(samples), BLOCK_FRAMES):
            if self._abort.is_set():
                stream.abort()  # drop the buffered tail; restarted on next play
                return
            stream.write(samples[start:start + BLOCK_FRAMES])

    def stop(self) -> None:
        """Cross-thread barge-in: flag only — the writer thread reacts within
        one block and aborts the stream itself."""
        self._abort.set()
