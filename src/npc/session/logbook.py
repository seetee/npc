"""Rolling campaign logbook plus crash-safe raw session transcripts."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

_SESSION_HEADING = re.compile(r"^## Session (\d+)\b", re.MULTILINE)


class Logbook:
    """One rolling markdown file with `## Session N — date` sections.

    Sections are upserted: re-summarizing the same session (checkpoints,
    /save, /end) replaces its section instead of appending duplicates.
    """

    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> str:
        if self.path.exists():
            return self.path.read_text(encoding="utf-8")
        return ""

    def next_session_number(self) -> int:
        numbers = [int(m) for m in _SESSION_HEADING.findall(self._read())]
        return max(numbers, default=0) + 1

    def tail(self, n_sessions: int) -> str:
        """The last n `## Session` sections, verbatim."""
        text = self._read()
        starts = [m.start() for m in _SESSION_HEADING.finditer(text)]
        if not starts:
            return ""
        return text[starts[max(0, len(starts) - n_sessions)]:].strip()

    def upsert_entry(self, session_no: int, date: str, body: str) -> None:
        heading = f"## Session {session_no} — {date}"
        section = f"{heading}\n\n{body.strip()}\n"
        text = self._read()

        pattern = re.compile(
            rf"^## Session {session_no}\b.*?(?=^## Session \d+\b|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        if pattern.search(text):
            text = pattern.sub(section, text, count=1)
        else:
            if text and not text.endswith("\n"):
                text += "\n"
            text += ("\n" if text else "") + section
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text, encoding="utf-8")


class Transcript:
    """Raw per-turn log for the current session; appended on every turn."""

    def __init__(self, sessions_dir: Path, now: datetime | None = None):
        stamp = (now or datetime.now()).strftime("%Y-%m-%d-%H%M")
        self.path = sessions_dir / f"{stamp}-transcript.md"

    def append_turn(self, speaker: str, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"**{speaker}:** {text}\n\n")

    def read(self) -> str:
        if self.path.exists():
            return self.path.read_text(encoding="utf-8")
        return ""
