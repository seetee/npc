"""Campaign configuration: config.toml loaded over dataclass defaults."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_VOICES_DIR = Path.home() / ".local" / "share" / "npc" / "voices"


@dataclass
class LlmConfig:
    backend: str = "ollama"  # or "openai" for Jan, LM Studio, llama.cpp, vLLM, …
    model: str = "qwen2.5:7b-instruct"
    host: str = "http://localhost:11434"
    timeout_seconds: float = 60.0  # per-request cap; a hung server becomes an error, not a stuck session
    retries: int = 1  # extra attempts after connection-level failures (never after HTTP errors)


@dataclass
class SttConfig:
    model: str = "small"
    language: str = "auto"  # "auto" detects per utterance (Swedish and English both work)
    device: str = "auto"
    silence_threshold_db: float = -45.0  # clips quieter than this never reach whisper


@dataclass
class TtsConfig:
    voice: str = "en_GB-alba-medium"
    voices_dir: str = str(DEFAULT_VOICES_DIR)

    @property
    def voice_path(self) -> Path:
        return Path(self.voices_dir).expanduser() / f"{self.voice}.onnx"


@dataclass
class HotkeyConfig:
    key: str = "KEY_SPACE"
    device: str = ""  # optional pin, e.g. /dev/input/by-id/usb-...-event-kbd
    grab: bool = False  # grab the device exclusively (only for a dedicated button!)


@dataclass
class Config:
    campaign_dir: Path
    llm: LlmConfig = field(default_factory=LlmConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    history_limit: int = 30
    logbook_sessions_in_prompt: int = 3
    checkpoint_every_turns: int = 20
    min_clip_seconds: float = 0.25

    @property
    def character_file(self) -> Path:
        return self.campaign_dir / "character.md"

    @property
    def adventure_file(self) -> Path:
        return self.campaign_dir / "adventure.md"

    @property
    def logbook_file(self) -> Path:
        return self.campaign_dir / "logbook.md"

    @property
    def sessions_dir(self) -> Path:
        return self.campaign_dir / "sessions"


class ConfigError(Exception):
    pass


def load_config(campaign_dir: Path) -> Config:
    """Load <campaign_dir>/config.toml; every key is optional."""
    campaign_dir = campaign_dir.expanduser().resolve()
    data: dict = {}
    toml_path = campaign_dir / "config.toml"
    if toml_path.exists():
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{toml_path}: {e}") from e

    def section(cls, name):
        try:
            return cls(**data.get(name, {}))
        except TypeError as e:
            valid = ", ".join(cls.__dataclass_fields__)
            raise ConfigError(f"{toml_path}: bad [{name}] section ({e}); valid keys: {valid}") from e

    top = {
        k: data[k]
        for k in ("history_limit", "logbook_sessions_in_prompt",
                  "checkpoint_every_turns", "min_clip_seconds")
        if k in data
    }
    return Config(
        campaign_dir=campaign_dir,
        llm=section(LlmConfig, "llm"),
        stt=section(SttConfig, "stt"),
        tts=section(TtsConfig, "tts"),
        hotkey=section(HotkeyConfig, "hotkey"),
        **top,
    )
