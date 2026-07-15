"""Rolling campaign logbook plus crash-safe raw session transcripts."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

_SESSION_HEADING = re.compile(r"^## Session (\d+)\b", re.MULTILINE)


def _atomic_write(path: Path, text: str) -> None:
    """Write via temp file + rename so a crash can never truncate the file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
        # demote any "## Session N" line the LLM produced inside the body —
        # it would corrupt section parsing (numbering, tail, future upserts)
        body = _SESSION_HEADING.sub(r"### Session \1", body)
        heading = f"## Session {session_no} — {date}"
        section = f"{heading}\n\n{body.strip()}\n"
        text = self._read()

        pattern = re.compile(
            rf"^## Session {session_no}\b.*?(?=^## Session \d+\b|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        if pattern.search(text):
            # lambda: section is literal text, not a replacement template
            # (a "\1" or stray backslash in LLM output must not be expanded)
            text = pattern.sub(lambda _match: section, text, count=1)
        else:
            if text and not text.endswith("\n"):
                text += "\n"
            text += ("\n" if text else "") + section
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, text)


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
