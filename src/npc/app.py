"""NPCApp: state machine, worker thread, and command dispatch.

Voice input (push-to-talk) is ALWAYS in-character player dialogue.
Typed lines are out-of-character GM instructions; /commands control the app.
"""

from __future__ import annotations

import queue
import threading
from datetime import date
from enum import Enum
from typing import Callable

from .audio.recorder import AudioClip
from .config import Config
from .session.history import ConversationHistory
from .session.logbook import Logbook, Transcript
from .session.prompt import build_system_prompt

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


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class NPCApp:
    def __init__(self, config: Config, *, llm, transcriber=None, recorder=None,
                 speaker=None, out: Callable[[str], None] = print):
        self.config = config
        self.llm = llm
        self.transcriber = transcriber
        self.recorder = recorder
        self.speaker = speaker
        self.out = out

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
        """Re-read config.toml mid-session. The LLM model/settings apply
        immediately (it's per-request); STT/TTS/hotkey are loaded into memory
        at startup and need a restart."""
        from .config import load_config

        new = load_config(self.config.campaign_dir)
        applied = [f"character & adventure notes (NPC: {self.npc_name})"]
        if new.llm.model != self.llm.model:
            self.llm.model = new.llm.model
            applied.append(f"LLM model → {new.llm.model}")
        restart_needed = [
            name for name, changed in (
                ("llm.host", new.llm.host != self.config.llm.host),
                ("[stt]", new.stt != self.config.stt),
                ("[tts]", new.tts != self.config.tts),
                ("[hotkey]", new.hotkey != self.config.hotkey),
            ) if changed
        ]
        self.history.limit = new.history_limit
        self.config = new
        self.out(f"[reloaded: {', '.join(applied)}]")
        if restart_needed:
            self.out(f"[changed {', '.join(restart_needed)} — takes effect after restart]")

    def start(self) -> None:
        self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                        name="npc-worker")
        self._worker.start()

    def shutdown(self, summarize: bool) -> None:
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=120)
        if summarize:
            if self._player_turns > 0:
                self.out("[summarizing session into logbook…]")
                self._write_logbook_entry()
                self.out(f"[logbook updated: {self.logbook.path}]")
            else:
                self.out("[nothing happened this session — logbook unchanged]")

    # ---------- state helpers ----------

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def _set_state_if(self, expected: set[State], new: State) -> bool:
        with self._lock:
            if self._state in expected:
                self._state = new
                return True
            return False

    # ---------- push-to-talk callbacks (hotkey thread) ----------

    def on_ptt_press(self) -> None:
        if self.recorder is None or self.transcriber is None:
            self.out("[voice input unavailable — run `npc doctor`]")
            return
        with self._lock:
            if self._state in (State.RECORDING, State.PROCESSING):
                self.out("[busy — still working on the previous line]")
                return
            if self._state is State.SPEAKING and self.speaker is not None:
                self.speaker.stop()  # barge-in
            self._state = State.RECORDING
        try:
            self.recorder.start()
        except Exception as e:
            self._set_state_if({State.RECORDING}, State.IDLE)
            self.out(f"[microphone error: {e}]")
            return
        self.out("[recording… release to send]")

    def on_ptt_release(self) -> None:
        if not self._set_state_if({State.RECORDING}, State.PROCESSING):
            return
        try:
            clip = self.recorder.stop()
        except Exception as e:
            self._set_state_if({State.PROCESSING}, State.IDLE)
            self.out(f"[microphone error: {e}]")
            return
        if clip.duration < self.config.min_clip_seconds:
            self._set_state_if({State.PROCESSING}, State.IDLE)
            self.out("[too short — discarded]")
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
        self.out("[noted — will shape the NPC's behavior]")
        return "ok"

    def _handle_command(self, line: str) -> str:
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd == "/say":
            if not arg:
                self.out("[usage: /say <what the player says>]")
            else:
                self._queue.put(("say", arg))
        elif cmd == "/save":
            self._queue.put(("save", None))
        elif cmd == "/reload":
            self._load_files()
            try:
                self._reload_config()
            except Exception as e:
                self.out(f"[config.toml not reloaded: {e}]")
        elif cmd == "/status":
            self.out(f"[state: {self.state.value} | NPC: {self.npc_name} | "
                     f"model: {self.llm.model} | session {self.session_no}, "
                     f"{self._player_turns} player turns | "
                     f"{len(self.ooc_notes)} standing GM notes]")
        elif cmd == "/help":
            self.out(HELP)
        elif cmd == "/end":
            return "end"
        elif cmd == "/quit":
            return "quit"
        else:
            self.out(f"[unknown command {cmd} — /help lists commands]")
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
                        self.out(f"[logbook saved: {self.logbook.path}]")
                    else:
                        self.out("[nothing to save yet]")
            except Exception as e:
                self.out(f"[error: {e}]")
            finally:
                self._set_state_if({State.PROCESSING, State.SPEAKING}, State.IDLE)
                self._queue.task_done()

    def _handle_utterance(self, clip: AudioClip) -> None:
        text = self.transcriber.transcribe(clip)
        if not text:
            self.out("[heard nothing]")
            return
        self.out(f"[player] {text}")
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
        reply = self.llm.chat(system, self.history.as_messages())
        self.history.add_npc(reply)
        self.transcript.append_turn("NPC", reply)
        self.out(f"[{self.npc_name}] {reply}")
        self._player_turns += 1

        if self.speaker is not None and self._set_state_if({State.PROCESSING}, State.SPEAKING):
            self.speaker.say(reply)

        if (self.config.checkpoint_every_turns > 0
                and self._player_turns % self.config.checkpoint_every_turns == 0):
            self._write_logbook_entry()
            self.out("[logbook checkpoint written]")

    def _write_logbook_entry(self) -> None:
        body = self.llm.summarize_session(
            self.transcript.read(),
            self.logbook.tail(self.config.logbook_sessions_in_prompt),
        )
        self.logbook.upsert_entry(self.session_no, date.today().isoformat(), body)
