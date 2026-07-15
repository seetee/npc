"""Startup checks with copy-pasteable fixes. `npc doctor` runs the deep
version (pre-downloads the whisper model); `npc run` runs the quick version
and refuses to start only on hard failures (Ollama + model)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .tts import download_hint


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    hard: bool = False  # hard failures prevent `npc run` from starting


def _whisper_cache_dir(model_size: str) -> Path:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return hub / f"models--Systran--faster-whisper-{model_size}"


def run_checks(config: Config, deep: bool = False) -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1. LLM server (Ollama or any OpenAI-compatible app)
    from .llm import OllamaClient, make_llm_client

    client = make_llm_client(config.llm)
    is_ollama = isinstance(client, OllamaClient)
    up = client.is_up()
    checks.append(CheckResult(
        "LLM server", up, hard=True,
        detail=f"{config.llm.backend} at {config.llm.host}",
        fix=("curl -fsSL https://ollama.com/install.sh | sh   "
             "# then: systemctl --user start ollama  (or)  sudo systemctl start ollama"
             if is_ollama else
             "start your LLM app (Jan, LM Studio, …) and enable its local API "
             "server; put its address in config.toml under [llm] host"),
    ))

    # 2. LLM model present
    if up:
        has = client.has_model()
        detail = config.llm.model
        if not has:
            others = client.available_models()
            if others:
                detail += f" (available: {', '.join(others[:5])})"
        checks.append(CheckResult(
            "LLM model", has, detail=detail, hard=True,
            fix=(f"ollama pull {config.llm.model}" if is_ollama else
                 "download/load the model in your LLM app, or set [llm] model in "
                 "config.toml to one of the available names"),
        ))

    # 3. Whisper model cached (offline after first download)
    cached = _whisper_cache_dir(config.stt.model).exists()
    if deep and not cached:
        try:
            from .stt import WhisperTranscriber

            WhisperTranscriber(config.stt.model, config.stt.language, config.stt.device)
            cached = True
        except Exception as e:
            checks.append(CheckResult("Whisper model download", False, detail=str(e)))
    checks.append(CheckResult(
        f"Whisper model ({config.stt.model})", cached,
        detail="cached" if cached else "not downloaded yet",
        fix="npc doctor   # downloads it (needs internet once)",
    ))

    # 4. Piper voice
    voice_path = config.tts.voice_path
    checks.append(CheckResult(
        f"Piper voice ({config.tts.voice})", voice_path.exists(), detail=str(voice_path),
        fix=download_hint(config.tts.voice, voice_path.parent),
    ))

    # 5. Audio devices
    try:
        import sounddevice as sd

        for kind in ("input", "output"):
            try:
                dev = sd.query_devices(kind=kind)
                checks.append(CheckResult(f"Audio {kind}", True, detail=dev["name"]))
            except Exception as e:
                checks.append(CheckResult(f"Audio {kind}", False, detail=str(e)))
    except Exception as e:
        checks.append(CheckResult(
            "Audio subsystem", False, detail=str(e),
            fix="sudo apt install libportaudio2",
        ))

    # 6. Push-to-talk key readable
    from .hotkey import HotkeyUnavailable, find_ptt_devices, keycode_from_name

    try:
        keycode = keycode_from_name(config.hotkey.key)
        devices = find_ptt_devices(keycode, config.hotkey.device)
        names = ", ".join(d.name for d in devices[:3])
        for d in devices:
            d.close()
        checks.append(CheckResult(f"Push-to-talk ({config.hotkey.key})", True, detail=names))
    except HotkeyUnavailable as e:
        checks.append(CheckResult(
            f"Push-to-talk ({config.hotkey.key})", False, detail=str(e),
            fix="sudo usermod -aG input $USER   # then log out and back in",
        ))

    return checks


def print_report(checks: list[CheckResult], out=print) -> bool:
    """Prints PASS/FAIL lines; returns True if no hard check failed."""
    ok = True
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        out(f"  [{mark}] {c.name}" + (f" — {c.detail}" if c.detail else ""))
        if not c.ok:
            if c.fix:
                out(f"         fix: {c.fix}")
            if c.hard:
                ok = False
    return ok
