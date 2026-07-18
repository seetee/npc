"""GM-gated secrets: clues the NPC may only reveal after DM approval.

A secrets file holds `## <id>` sections with a required `hint:` line, an
optional `mode:` (hesitate|deflect), an optional `revealed:` line (written
back on approval), and the secret body. THE BODY NEVER ENTERS THE PROMPT
UNTIL REVEALED — the LLM sees only the hints, so a leak is impossible even
if the model ignores every instruction. When a locked topic comes up, the
model ends its stalling line with `[CHECK:<id>]`; strip_decoration's
bracketed-span rule already removes the marker from everything spoken or
recorded, so detection must run on the RAW reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .logbook import _atomic_write

MODES = ("hesitate", "deflect")

MARKER_RE = re.compile(r"\[\s*CHECK\s*:\s*([A-Za-z0-9-]+)\s*\]", re.IGNORECASE)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_HEADING_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)
_META_RE = re.compile(r"^(hint|mode|revealed):\s*(.*)$")

FALLBACK_STALL = "Hm. Give me a moment — I need to think about that."


class SecretsError(Exception):
    """A secrets file that cannot be parsed (doctor and /reload surface it)."""


@dataclass
class Secret:
    id: str
    hint: str  # one line: what it concerns / when sharing might be appropriate
    body: str  # the actual clue — never in the prompt until revealed
    mode: str = "hesitate"
    revealed: str | None = None  # e.g. "session 3" once the GM approved


@dataclass
class SecretsSheet:
    """One NPC's secrets file, round-trippable so reveals can be written back
    without losing the preamble above the first `## id` heading."""

    preamble: str = ""
    entries: list[Secret] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> SecretsSheet:
        headings = list(_HEADING_RE.finditer(text))
        preamble = text[: headings[0].start()].strip() if headings else text.strip()
        entries: list[Secret] = []
        for i, m in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            entries.append(_parse_entry(m.group(1), text[m.end():end]))
        seen: set[str] = set()
        for entry in entries:
            if entry.id in seen:
                raise SecretsError(f"duplicate secret id '{entry.id}'")
            seen.add(entry.id)
        return cls(preamble=preamble, entries=entries)

    @classmethod
    def load(cls, path: Path) -> SecretsSheet:
        if not path.exists():
            return cls()
        return cls.parse(path.read_text(encoding="utf-8"))

    def render(self) -> str:
        parts = [self.preamble.strip()] if self.preamble.strip() else []
        for s in self.entries:
            lines = [f"## {s.id}", f"hint: {s.hint}"]
            if s.mode != "hesitate":
                lines.append(f"mode: {s.mode}")
            if s.revealed:
                lines.append(f"revealed: {s.revealed}")
            parts.append("\n".join(lines) + (f"\n\n{s.body}" if s.body else ""))
        return "\n\n".join(parts) + "\n"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, self.render())

    def get(self, secret_id: str) -> Secret | None:
        return next((s for s in self.entries if s.id == secret_id), None)

    def locked(self) -> list[Secret]:
        return [s for s in self.entries if not s.revealed]

    def revealed(self) -> list[Secret]:
        return [s for s in self.entries if s.revealed]


def _parse_entry(heading: str, section: str) -> Secret:
    secret_id = heading.strip().lower()
    if not _ID_RE.match(secret_id):
        raise SecretsError(
            f"'## {heading}' is not a valid secret id "
            "(use lowercase letters, digits, and dashes, e.g. 'duke-tomb')")
    meta = {"mode": "hesitate", "revealed": None, "hint": None}
    lines = section.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = _META_RE.match(line)
        if m:
            meta[m.group(1)] = m.group(2).strip()
        elif line:
            break  # first non-metadata content line starts the body
        i += 1
    body = "\n".join(lines[i:]).strip()
    if not meta["hint"]:
        raise SecretsError(f"secret '{secret_id}' is missing its 'hint:' line")
    if meta["mode"] not in MODES:
        raise SecretsError(
            f"secret '{secret_id}' has mode '{meta['mode']}' "
            f"(must be one of: {', '.join(MODES)})")
    if not body:
        raise SecretsError(f"secret '{secret_id}' has no body to reveal")
    return Secret(id=secret_id, hint=meta["hint"], body=body,
                  mode=meta["mode"], revealed=meta["revealed"] or None)


def find_markers(raw: str) -> list[str]:
    """All [CHECK:id] markers in a raw LLM reply, lowercased, deduped, in order."""
    seen: list[str] = []
    for m in MARKER_RE.finditer(raw):
        secret_id = m.group(1).lower()
        if secret_id not in seen:
            seen.append(secret_id)
    return seen


def strip_markers(text: str) -> str:
    """Remove [CHECK:id] markers. strip_decoration's bracketed-span rule
    already eats them on the main path — this covers raw-text fallbacks
    (extract_dialogue's de-asterisk fallback would otherwise SPEAK them)."""
    return MARKER_RE.sub("", text)


class MarkerScrubber:
    """Removes [CHECK:id] markers from a token stream before it reaches the
    overlay (NpcReplyChunk shows the raw feed on the table screen). A marker
    can be split across chunks, so text from an unclosed '[' is held back
    until its ']' arrives or the stream ends."""

    def __init__(self) -> None:
        self._held = ""

    def feed(self, chunk: str) -> str:
        text = MARKER_RE.sub("", self._held + chunk)
        cut = text.rfind("[")
        if cut != -1 and "]" not in text[cut:]:
            self._held = text[cut:]
            return text[:cut]
        self._held = ""
        return text

    def flush(self) -> str:
        held, self._held = self._held, ""
        return held


def revealed_block(secrets: list[Secret]) -> str:
    sections = "\n\n".join(f"## {s.id}\n{s.body}" for s in secrets)
    return (
        "# Knowledge you may now share\n"
        "\n"
        "The GM has unlocked the following. Your character knows these details\n"
        "fully and may share them when the conversation makes it natural\n"
        "(standing GM instructions may add conditions):\n"
        "\n" + sections
    )


def locked_block(secrets: list[Secret]) -> str:
    """The gating instructions. The model never sees a body here — only hints.

    Wording is probe-tuned for 7B models (scripts/probe_secrets.py):
    per-topic `marker:` lines stop the model copying whichever id it saw
    last, and the explicit NO branch (plus its WRONG/RIGHT pair) stops
    markers on unrelated questions. Re-validate any change with the probe."""
    topics = "\n\n".join(
        f"- topic ({s.id}): {s.hint}\n"
        f"  handling: {s.mode}\n"
        f"  marker: [CHECK:{s.id}]"
        for s in secrets)
    example = secrets[0].id
    # recency bias: when a model staples a spurious marker onto an unrelated
    # reply it picks the LAST-listed id — so that id stars in the WRONG example
    example_wrong = secrets[-1].id
    return f"""# Locked knowledge (GM-gated)

Your character knows more about the topics listed below, but the actual
details are LOCKED: they have been withheld from you, and you do not know
them. You cannot reveal what you were never told.

{topics}

Decide EVERY reply like this:

1. Did the player's words ask about or touch one of the topics above?
   NO  → reply normally and output NO marker. Most replies are this case;
         marker replies are rare. If you are not sure, choose NO.
   YES → reply with ONE short spoken line in that topic's handling style,
         then end the reply with that topic's marker line, copied exactly
         from the list above.

Handling styles:
- hesitate: pause out loud, in character — you are visibly deciding whether
  to trust them.
- deflect: brush the question off naturally, as if you truly knew nothing
  worth telling.

Hard rules:
- Never invent, guess, embellish, or hint at what the locked details might
  be — you do not know them, so any specifics you give would be false.
- The marker is read by a machine and is never heard by the players. Never
  mention markers, the GM, permission, locks, or secrets in your spoken
  words.
- Only ever copy a marker from the topic list, and only the marker of the
  one topic that was touched.
- A marker may only end a reply that stalls or deflects. A reply that
  actually answers the player's question must never contain a marker.

WRONG: The Duke? He lies in the sea-cave beneath the old lighthouse.
  (invented locked details)
WRONG: I am not permitted to reveal that information yet.
  (broke character, mentioned permission)
WRONG: Safe travels — the east road is clear. [CHECK:{example_wrong}]
  (marker although no listed topic was touched)
RIGHT (a locked topic was touched): Hm. Give me a moment — that is not a
  thing I speak of lightly. [CHECK:{example}]
RIGHT (no listed topic touched, no marker): The east road is clear this
  week; keep to it before dusk."""


def delivery_instruction(secret: Secret, rider: str = "") -> str:
    """One-shot GM message for the reveal turn — never a standing note."""
    extra = f"\n\nGM adds: {rider}" if rider else ""
    return (
        "GM NOTE (out-of-character): The GM has decided you WILL share the\n"
        f"locked topic ({secret.id}) now. Here are the true details, which\n"
        "your character has known all along:\n"
        "\n"
        f"{secret.body}\n"
        "\n"
        "Tell the player now, in your own words and voice, staying fully in\n"
        "character. Convey the substance faithfully; do not add invented\n"
        f"specifics beyond it.{extra}"
    )


def deny_note(secret: Secret, rider: str = "") -> str:
    """Standing OOC note added on /no — survives history trimming. Deliberately
    references the HINT and never the id: the id is what a 7B model needs to
    reconstruct a [CHECK:…] marker after the topic is delisted (probe-verified),
    and the word 'marker' itself primes one."""
    extra = f" GM adds: {rider}" if rider else ""
    return (
        f"About {secret.hint} — your character truly knows nothing and has "
        "nothing to tell. If it comes up again, brush it off naturally, in "
        f"character.{extra}"
    )
