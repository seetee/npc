"""Render the README audio sample(s) with the real Piper voice.

    uv run python scripts/render_samples.py

Writes docs/samples/*.mp3 (wav → ffmpeg). Re-run after changing the default
voice or the sample lines. Needs the voice downloaded (`npc doctor --fix`)
and ffmpeg on PATH.
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from npc.config import Config
from npc.tts import PiperSpeaker

SAMPLES = {
    "vess-greeting": (
        "Approach, traveler. The monolith hums tonight — storms, or something "
        "older. Ask what you came to ask."
    ),
}


def main() -> int:
    out_dir = Path(__file__).parent.parent / "docs" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(campaign_dir=Path("."))
    speaker = PiperSpeaker(config.tts.voice_path)
    for name, text in SAMPLES.items():
        samples, rate = speaker.synthesize(text)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            with wave.open(tmp.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(samples.tobytes())
            target = out_dir / f"{name}.mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                            tmp.name, "-b:a", "96k", str(target)], check=True)
            print(f"wrote {target} ({target.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
