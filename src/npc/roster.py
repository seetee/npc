"""Campaign character roster: discovery, per-NPC state, and name resolution.

A campaign hosts one NPC (the legacy character.md) or many (characters/*.md).
Each roster entry is a CharacterSlot carrying ALL per-NPC state — sheet text,
conversation history, standing GM notes, its own logbook, and the turn buffer
that feeds per-NPC session summaries — so switching NPCs means swapping one
slot. Kept import-light (config + session only) so doctor.py can reuse the
discovery without pulling in the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .session.history import ConversationHistory
from .session.logbook import Logbook


@dataclass(frozen=True)
class CharacterFile:
    stem: str
    path: Path
    logbook_path: Path
    legacy: bool  # the campaign-root character.md (keeps using logbook.md)


def discover_character_files(campaign_dir: Path) -> list[CharacterFile]:
    """Legacy character.md first (if present), then characters/*.md sorted by
    stem. characters/character.md is skipped when the legacy file exists —
    the stems would collide (doctor warns about that)."""
    found: list[CharacterFile] = []
    legacy = campaign_dir / "character.md"
    if legacy.exists():
        found.append(CharacterFile("character", legacy,
                                   campaign_dir / "logbook.md", legacy=True))
    characters_dir = campaign_dir / "characters"
    if characters_dir.is_dir():
        for path in sorted(characters_dir.glob("*.md")):
            if path.stem == "character" and found:
                continue  # stem collision with the legacy character.md
            found.append(CharacterFile(
                path.stem, path,
                campaign_dir / "logbooks" / f"{path.stem}.md", legacy=False))
    return found


def read_display_name(text: str, fallback: str) -> str:
    """The first '# ' heading of a character sheet, else the fallback."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


@dataclass
class CharacterSlot:
    """Everything that is per-NPC. The app swaps its `active` slot on /npc."""

    stem: str
    path: Path
    name: str
    character: str  # sheet text, rebuilt into the system prompt every turn
    logbook: Logbook
    history: ConversationHistory
    voice: str | None = None  # piper voice name; None = campaign default
    ooc_notes: list[str] = field(default_factory=list)
    turns: list[tuple[str, str]] = field(default_factory=list)
    player_turns: int = 0
    dirty: bool = False  # has turns not yet summarized into the logbook

    def refresh(self, config: Config) -> None:
        """Re-read the character file (/reload). Conversation state is
        deliberately preserved — only text, name, and voice are replaced."""
        self.character = self.path.read_text(encoding="utf-8")
        self.name = read_display_name(self.character, self.stem)
        self.voice = config.tts.voices.get(self.stem)


def load_slot(ref: CharacterFile, config: Config) -> CharacterSlot:
    text = ref.path.read_text(encoding="utf-8")
    return CharacterSlot(
        stem=ref.stem,
        path=ref.path,
        name=read_display_name(text, ref.stem),
        character=text,
        logbook=Logbook(ref.logbook_path),
        history=ConversationHistory(limit=config.history_limit),
        voice=config.tts.voices.get(ref.stem),
    )


def resolve_npc(roster: dict[str, CharacterSlot],
                query: str) -> CharacterSlot | list[CharacterSlot]:
    """Match by stem or display name, case-insensitive; an exact match wins,
    otherwise a unique prefix. Returns the slot, or the candidate list
    ([] = unknown, 2+ = ambiguous) so the caller can render a helpful line."""
    wanted = query.strip().lower()
    exact = [slot for slot in roster.values()
             if wanted in (slot.stem.lower(), slot.name.lower())]
    if len(exact) == 1:
        return exact[0]
    prefixed = [slot for slot in roster.values()
                if slot.stem.lower().startswith(wanted)
                or slot.name.lower().startswith(wanted)]
    if len(prefixed) == 1:
        return prefixed[0]
    return exact or prefixed


def render_turns(turns: list[tuple[str, str]]) -> str:
    """Per-NPC summarizer input, in the same shape as the session transcript."""
    return "\n\n".join(f"**{speaker}:** {text}" for speaker, text in turns)
