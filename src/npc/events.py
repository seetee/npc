"""Structured events emitted by NPCApp.

The app never prints. It emits Event objects to a single on_event callback;
the terminal UI renders them with format_event(). A future subscriber — an
OBS overlay, a web remote, a stream deck — receives the same events with
full payloads instead of parsing bracket strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class State(StrEnum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for all app events."""


@dataclass(frozen=True, slots=True)
class StateChanged(Event):
    """Every state transition — the heartbeat an overlay would visualize."""
    state: State


@dataclass(frozen=True, slots=True)
class RecordingStarted(Event):
    auto_stop: bool = False  # True when a VAD recorder will end it on silence


@dataclass(frozen=True, slots=True)
class RecordingDiscarded(Event):
    reason: str


@dataclass(frozen=True, slots=True)
class HeardNothing(Event):
    pass


@dataclass(frozen=True, slots=True)
class PlayerSpoke(Event):
    """A voice utterance was transcribed (always in-character)."""
    text: str


@dataclass(frozen=True, slots=True)
class NpcReplied(Event):
    npc_name: str
    text: str


@dataclass(frozen=True, slots=True)
class NpcReplyChunk(Event):
    """A fragment of the reply as it streams from the LLM. Overlay material —
    the CLI waits for the complete NpcReplied instead."""
    text: str


@dataclass(frozen=True, slots=True)
class GmNoteAdded(Event):
    text: str


@dataclass(frozen=True, slots=True)
class Busy(Event):
    pass


@dataclass(frozen=True, slots=True)
class VoiceUnavailable(Event):
    pass


@dataclass(frozen=True, slots=True)
class MicrophoneError(Event):
    message: str


@dataclass(frozen=True, slots=True)
class ErrorOccurred(Event):
    message: str


@dataclass(frozen=True, slots=True)
class TurnCompleted(Event):
    """Per-stage timings for one player turn, measured from the clip landing
    in the worker (or the /say line) until the reply finished. On the
    streaming path the LLM and TTS stages overlap, so stages don't sum to
    total. Hidden in the CLI by default — `npc run --timings` prints them."""
    stt_seconds: float | None            # None for typed /say turns
    llm_first_token_seconds: float | None  # None on the non-streaming path
    llm_seconds: float
    speak_seconds: float | None          # None when spoken replies are disabled
    total_seconds: float


@dataclass(frozen=True, slots=True)
class SessionEnding(Event):
    pass


@dataclass(frozen=True, slots=True)
class LogbookWritten(Event):
    path: str
    kind: Literal["save", "checkpoint", "end"]


@dataclass(frozen=True, slots=True)
class ConfigReloaded(Event):
    applied: tuple[str, ...]
    restart_needed: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StatusReport(Event):
    state: State
    npc_name: str
    model: str
    session_no: int
    player_turns: int
    gm_notes: int
    last_turn_seconds: float | None = None
    avg_turn_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class Info(Event):
    """Free-text output with no structured payload (help text, usage hints)."""
    message: str


def format_event(event: Event) -> str | None:
    """Render an event for the terminal; None means 'not shown in the CLI'."""
    match event:
        case StateChanged():
            return None  # overlay material — too chatty for the terminal
        case TurnCompleted():
            return None  # shown only via `npc run --timings` (format_timings)
        case RecordingStarted(auto_stop=auto):
            return "[recording… pause to send]" if auto else "[recording… release to send]"
        case RecordingDiscarded(reason=reason):
            return f"[{reason} — discarded]"
        case HeardNothing():
            return "[heard nothing]"
        case PlayerSpoke(text=text):
            return f"[player] {text}"
        case NpcReplied(npc_name=name, text=text):
            return f"[{name}] {text}"
        case NpcReplyChunk():
            return None  # the CLI prints the finished NpcReplied line
        case GmNoteAdded():
            return "[noted — will shape the NPC's behavior]"
        case Busy():
            return "[busy — still working on the previous line]"
        case VoiceUnavailable():
            return "[voice input unavailable — run `npc doctor`]"
        case MicrophoneError(message=message):
            return f"[microphone error: {message}]"
        case ErrorOccurred(message=message):
            return f"[error: {message}]"
        case SessionEnding():
            return "[summarizing session into logbook…]"
        case LogbookWritten(path=path, kind=kind):
            return {"save": f"[logbook saved: {path}]",
                    "checkpoint": "[logbook checkpoint written]",
                    "end": f"[logbook updated: {path}]"}[kind]
        case ConfigReloaded(applied=applied, restart_needed=restart_needed):
            text = f"[reloaded: {', '.join(applied)}]"
            if restart_needed:
                text += f"\n[changed {', '.join(restart_needed)} — takes effect after restart]"
            return text
        case StatusReport() as s:
            timing = ""
            if s.last_turn_seconds is not None:
                timing = (f" | last turn {s.last_turn_seconds:.1f}s, "
                          f"avg {s.avg_turn_seconds:.1f}s")
            return (f"[state: {s.state} | NPC: {s.npc_name} | model: {s.model} | "
                    f"session {s.session_no}, {s.player_turns} player turns | "
                    f"{s.gm_notes} standing GM notes{timing}]")
        case Info(message=message):
            return message
        case _:
            return repr(event)


def format_timings(event: TurnCompleted) -> str:
    """Render TurnCompleted for `npc run --timings`."""
    parts = []
    if event.stt_seconds is not None:
        parts.append(f"stt {event.stt_seconds:.1f}s")
    llm = f"llm {event.llm_seconds:.1f}s"
    if event.llm_first_token_seconds is not None:
        llm += f" (first token {event.llm_first_token_seconds:.2f}s)"
    parts.append(llm)
    if event.speak_seconds is not None:
        parts.append(f"speak {event.speak_seconds:.1f}s")
    return f"[turn {event.total_seconds:.1f}s: {' | '.join(parts)}]"


def print_event(event: Event) -> None:
    """Default subscriber: print what format_event renders."""
    text = format_event(event)
    if text is not None:
        print(text)
