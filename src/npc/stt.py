"""Local speech-to-text via faster-whisper.

GPU is used when available. CUDA runtime libraries can come from the optional
`npc[cuda]` extra (nvidia-cublas-cu12 / nvidia-cudnn-cu12); if CUDA fails for
any reason we fall back to CPU rather than crash mid-session.

Whisper hallucinates on (near-)silence: it was trained on YouTube, so silence
"transcribes" to outro phrases and subtitle credits. Three guards keep those
away from the LLM: an energy gate in the app (AudioClip.dbfs), a per-segment
no_speech_prob filter here, and the PHANTOM_PHRASES blocklist below.
"""

from __future__ import annotations

import ctypes
import glob
import os
import re
from typing import Protocol

import numpy as np

from .audio.recorder import SAMPLE_RATE, AudioClip


class Transcriber(Protocol):
    def transcribe(self, clip: AudioClip) -> str: ...


NO_SPEECH_MAX = 0.6  # drop segments whisper itself doubts contain speech

# What whisper "hears" in noise — YouTube outros and subtitle credits, in both
# table languages. Lowercase; matched as substrings by looks_like_hallucination.
PHANTOM_PHRASES = (
    # English
    "thanks for watching",
    "thank you for watching",
    "thank you so much for watching",
    "please subscribe",
    "subscribe to the channel",
    "subscribe to my channel",
    "see you in the next video",
    # Swedish
    "tack för att du tittade",
    "tack för att ni tittade",
    "tack för att du har tittat",
    "vi ses i nästa video",
    "undertexter av",
    "undertexter från",
    "svensktextning.nu",
    "amara.org",
    "bti studios",
    "btistudios",
    "sdi media",
)


def looks_like_hallucination(text: str) -> bool:
    """True when the WHOLE transcript is phantom phrases (plus punctuation).
    A real utterance that merely contains one — "thanks for watching my back"
    — is never filtered."""
    remainder = text.lower()
    for phrase in PHANTOM_PHRASES:
        remainder = remainder.replace(phrase, " ")
    return bool(text.strip()) and re.search(r"\w", remainder) is None


def join_segments(segments) -> str:
    """Concatenate whisper segments, dropping those whisper itself flags as
    probably-not-speech (pure so it is testable without a model)."""
    return " ".join(
        s.text.strip() for s in segments if s.no_speech_prob <= NO_SPEECH_MAX
    ).strip()


def _preload_cuda_libs() -> None:
    """Make pip-installed NVIDIA libs loadable without LD_LIBRARY_PATH."""
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
    except ImportError:
        return
    for pkg in (nvidia.cublas.lib, nvidia.cudnn.lib):
        for so in sorted(glob.glob(os.path.join(pkg.__path__[0], "*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class WhisperTranscriber:
    def __init__(self, model_size: str = "small", language: str = "auto",
                 device: str = "auto"):
        from faster_whisper import WhisperModel

        self.language = None if language == "auto" else language
        if device == "auto":
            _preload_cuda_libs()
            try:
                self._model = WhisperModel(model_size, device="auto")
                self._warm_up()
            except Exception:
                self._model = WhisperModel(model_size, device="cpu")
                self._warm_up()
        else:
            self._model = WhisperModel(model_size, device=device)
            self._warm_up()
        self.device = self._model.model.device

    def _warm_up(self) -> None:
        """Forces the first (slow) model invocation — and surfaces broken CUDA
        setups at startup instead of mid-session."""
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        segments, _ = self._model.transcribe(silence, language="en", vad_filter=False)
        for _ in segments:
            pass

    def transcribe(self, clip: AudioClip) -> str:
        if clip.duration == 0:
            return ""
        audio = clip.to_float32()
        if clip.sample_rate != SAMPLE_RATE:
            # whisper expects 16 kHz; linear resample is fine for speech
            n_target = round(len(audio) * SAMPLE_RATE / clip.sample_rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_target),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
        )
        return join_segments(segments)
