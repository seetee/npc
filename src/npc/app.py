"""NPCApp: state machine, worker thread, and command dispatch.

Voice input (push-to-talk) is ALWAYS in-character player dialogue.
Typed lines are out-of-character GM instructions; /commands control the app.
All output happens as structured events (see events.py).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from datetime import date

from .audio.recorder import AudioClip
from .config import Config
from .events import (
    Busy,
    ConfigReloaded,
    ErrorOccurred,
    Event,
    GmNoteAdded,
    HeardNothing,
    Info,
    LogbookWritten,
    MicrophoneError,
    NpcReplied,
    PlayerSpoke,
    RecordingDiscarded,
    RecordingStarted,
    SessionEnding,
    State,
    StateChanged,
    StatusReport,
    VoiceUnavailable,
    print_event,
)
from .session.history import ConversationHistory
from .session.logbook import Logbook, Transcript
from .session.prompt import build_system_prompt, extract_dialogue

HELP = """\
Voice (hold the push-to-talk key)  in-character player dialogue
<typed text>                       out-of-character instruction to the LLM
/say <text>                        typed in-character player line
/save                              summarize session into the logbook now
/reload                            re-read character.md / adventure.md / config.toml
/status                            show state, model, session info
/end                               summarize into logbook and exit
/quit                              exit WITHOUT saving a summary
/help                              this text"""


class NPCApp:
    def __init__(self, config: Config, *, llm, transcriber=None, recorder=None,
                 speaker=None, on_event: Callable[[Event], None] | None = None):
        self.config = config
        self.llm = llm
        self.transcriber = transcriber
        self.recorder = recorder
        self.speaker = speaker
        self._on_event = on_event or print_event

        self.history = ConversationHistory(limit=config.history_limit)
        self.logbook = Logbook(config.logbook_file)
        self.transcript = Transcript(config.sessions_dir)
        self.session_no = self.logbook.next_session_number()
        self.ooc_notes: list[str] = []

        self._queue: queue.Queue = queue.Queue()
        self._state = State.IDLE
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._player_turns = 0

        self.character = ""
        self.adventure = ""
        self.npc_name = "NPC"
        self._load_files()

    # ---------- lifecycle ----------

    def _load_files(self) -> None:
        self.character = self.config.character_file.read_text(encoding="utf-8")
        if self.config.adventure_file.exists():
            self.adventure = self.config.adventure_file.read_text(encoding="utf-8")
        for line in self.character.splitlines():
            if line.startswith("# "):
                self.npc_name = line[2:].strip()
                break

    def _reload_config(self) -> None:
        """Re-read config.toml mid-session. The LLM model applies immediately
        (it's per-request); STT/TTS/hotkey are loaded at startup and need a
        restart."""
        from .config import load_config

        new = load_config(self.config.campaign_dir)
        applied = [f"character & adventure notes (NPC: {self.npc_name})"]
        if new.llm.model != self.llm.model:
            self.llm.model = new.llm.model
            applied.append(f"LLM model → {new.llm.model}")
        restart_needed = tuple(
            name for name, changed in (
                ("llm.backend", new.llm.backend != self.config.llm.backend),
                ("llm.host", new.llm.host != self.config.llm.host),
                ("[stt]", new.stt != self.config.stt),
                ("[tts]", new.tts != self.config.tts),
                ("[hotkey]", new.hotkey != self.config.hotkey),
            ) if changed
        )
        self.history.limit = new.history_limit
        self.config = new
        self._emit(ConfigReloaded(applied=tuple(applied), restart_needed=restart_needed))

    def start(self) -> None:
        self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                        name="npc-worker")
        self._worker.start()

    def shutdown(self, summarize: bool) -> None:
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=120)
            if self._worker.is_alive():
                # a turn is stuck (hung LLM call?) — don't also write the
                # logbook from this thread while the worker might
                self._emit(ErrorOccurred(
                    "a turn is still running; skipping the final summary — "
                    "run /save next session or edit logbook.md by hand"))
                return
        if summarize:
            if self._player_turns > 0:
                self._emit(SessionEnding())
                self._write_logbook_entry()
                self._emit(LogbookWritten(str(self.logbook.path), "end"))
            else:
                self._emit(Info("[nothing happened this session — logbook unchanged]"))

    # ---------- events & state ----------

    def _emit(self, event: Event) -> None:
        try:
            self._on_event(event)
        except Exception:  # a broken subscriber must not kill the session
            pass

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def _set_state_if(self, expected: set[State], new: State) -> bool:
        with self._lock:
            changed = self._state in expected
            if changed:
                self._state = new
        if changed:
            self._emit(StateChanged(new))
        return changed

    # ---------- push-to-talk callbacks (hotkey thread) ----------

    def on_ptt_press(self) -> None:
        if self.recorder is None or self.transcriber is None:
            self._emit(VoiceUnavailable())
            return
        with self._lock:
            prior = self._state
            if prior not in (State.RECORDING, State.PROCESSING):
                self._state = State.RECORDING
        if prior in (State.RECORDING, State.PROCESSING):
            self._emit(Busy())
            return
        if prior is State.SPEAKING and self.speaker is not None:
            self.speaker.stop()  # barge-in
        self._emit(StateChanged(State.RECORDING))
        try:
            self.recorder.start()
        except Exception as e:
            self._set_state_if({State.RECORDING}, State.IDLE)
            self._emit(MicrophoneError(str(e)))
            return
        self._emit(RecordingStarted())

    def on_ptt_release(self) -> None:
        if not self._set_state_if({State.RECORDING}, State.PROCESSING):
            return
        try:
            clip = self.recorder.stop()
        except Exception as e:
            self._set_state_if({State.PROCESSING}, State.IDLE)
            self._emit(MicrophoneError(str(e)))
            return
        if clip.duration < self.config.min_clip_seconds:
            self._set_state_if({State.PROCESSING}, State.IDLE)
            self._emit(RecordingDiscarded("too short"))
            return
        self._queue.put(("utterance", clip))

    # ---------- typed input (main thread) ----------

    def handle_line(self, line: str) -> str:
        """Returns "ok", "end" (quit + summarize) or "quit" (no summary)."""
        line = line.strip()
        if not line:
            return "ok"
        if line.startswith("/"):
            return self._handle_command(line)
        self.ooc_notes.append(line)
        self.history.add_ooc(line)
        self.transcript.append_turn("GM", line)
        self._emit(GmNoteAdded(line))
        return "ok"

    def _handle_command(self, line: str) -> str:
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        match cmd:
            case "/say":
                if not arg:
                    self._emit(Info("[usage: /say <what the player says>]"))
                else:
                    self._queue.put(("say", arg))
            case "/save":
                self._queue.put(("save", None))
            case "/reload":
                self._load_files()
                try:
                    self._reload_config()
                except Exception as e:
                    self._emit(ErrorOccurred(f"config.toml not reloaded: {e}"))
            case "/status":
                self._emit(StatusReport(
                    state=self.state, npc_name=self.npc_name,
                    model=self.llm.model, session_no=self.session_no,
                    player_turns=self._player_turns, gm_notes=len(self.ooc_notes),
                ))
            case "/help":
                self._emit(Info(HELP))
            case "/end":
                return "end"
            case "/quit":
                return "quit"
            case _:
                self._emit(Info(f"[unknown command {cmd} — /help lists commands]"))
        return "ok"

    # ---------- worker thread ----------

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                if kind == "utterance":
                    self._handle_utterance(payload)
                elif kind == "say":
                    self._respond_to_player(payload)
                elif kind == "save":
                    if self._player_turns > 0:
                        self._write_logbook_entry()
                        self._emit(LogbookWritten(str(self.logbook.path), "save"))
                    else:
                        self._emit(Info("[nothing to save yet]"))
            except Exception as e:
                self._emit(ErrorOccurred(str(e)))
            finally:
                self._set_state_if({State.PROCESSING, State.SPEAKING}, State.IDLE)
                self._queue.task_done()

    def _handle_utterance(self, clip: AudioClip) -> None:
        text = self.transcriber.transcribe(clip)
        if not text:
            self._emit(HeardNothing())
            return
        self._emit(PlayerSpoke(text))
        self._respond_to_player(text)

    def _respond_to_player(self, text: str) -> None:
        self._set_state_if({State.IDLE}, State.PROCESSING)
        self.history.add_player(text)
        self.transcript.append_turn("PLAYER", text)
        system = build_system_prompt(
            self.character,
            self.adventure,
            self.logbook.tail(self.config.logbook_sessions_in_prompt),
            self.ooc_notes,
        )
        reply = extract_dialogue(self.llm.chat(system, self.history.as_messages()),
                                 self.npc_name)
        self.history.add_npc(reply)
        self.transcript.append_turn("NPC", reply)
        self._emit(NpcReplied(self.npc_name, reply))
        self._player_turns += 1

        if self.speaker is not None and self._set_state_if({State.PROCESSING}, State.SPEAKING):
            self.speaker.say(reply)

        if (self.config.checkpoint_every_turns > 0
                and self._player_turns % self.config.checkpoint_every_turns == 0):
            self._write_logbook_entry()
            self._emit(LogbookWritten(str(self.logbook.path), "checkpoint"))

    def _write_logbook_entry(self) -> None:
        body = self.llm.summarize_session(
            self.transcript.read(),
            self.logbook.tail(self.config.logbook_sessions_in_prompt),
        )
        self.logbook.upsert_entry(self.session_no, date.today().isoformat(), body)
