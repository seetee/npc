"""NPCApp: state machine, worker thread, and command dispatch.

Voice input (push-to-talk) is ALWAYS in-character player dialogue.
Typed lines are out-of-character GM instructions; /commands control the app.
All output happens as structured events (see events.py).
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
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
    NpcReplyChunk,
    PlayerSpoke,
    RecordingDiscarded,
    RecordingStarted,
    SessionEnding,
    State,
    StateChanged,
    StatusReport,
    TurnCompleted,
    VoiceUnavailable,
    print_event,
)
from .llm import StreamingNotSupported
from .session.history import ConversationHistory
from .session.logbook import Logbook, Transcript
from .session.prompt import build_system_prompt, extract_dialogue, strip_decoration
from .session.sentences import iter_sentences
from .stt import looks_like_hallucination

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
        self._last_turn: TurnCompleted | None = None
        self._turn_totals: deque[float] = deque(maxlen=20)

        self.character = ""
        self.adventure = ""
        self.npc_name = "NPC"
        self._load_files()

        if self.recorder is not None:
            # only a VAD recorder ever fires this; harmless for push-to-talk
            self.recorder.on_auto_stop = self.on_auto_stop

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
                ("llm.timeout/retries",
                 (new.llm.timeout_seconds, new.llm.retries)
                 != (self.config.llm.timeout_seconds, self.config.llm.retries)),
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
        self._emit(RecordingStarted(
            auto_stop=getattr(self.recorder, "is_auto_stop", False)))

    def on_ptt_release(self) -> None:
        if not self._set_state_if({State.RECORDING}, State.PROCESSING):
            return
        try:
            clip = self.recorder.stop()
        except Exception as e:
            self._set_state_if({State.PROCESSING}, State.IDLE)
            self._emit(MicrophoneError(str(e)))
            return
        self._enqueue_clip(clip)

    def on_auto_stop(self, clip: AudioClip) -> None:
        """Fired by a VAD recorder (from its finalizer thread) when silence
        ends the recording; a manual stop that won the state race makes this
        a no-op — exactly one of the two paths enqueues the clip."""
        if not self._set_state_if({State.RECORDING}, State.PROCESSING):
            return
        self._enqueue_clip(clip)

    def _enqueue_clip(self, clip: AudioClip) -> None:
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
                totals = self._turn_totals
                self._emit(StatusReport(
                    state=self.state, npc_name=self.npc_name,
                    model=self.llm.model, session_no=self.session_no,
                    player_turns=self._player_turns, gm_notes=len(self.ooc_notes),
                    last_turn_seconds=(self._last_turn.total_seconds
                                       if self._last_turn else None),
                    avg_turn_seconds=sum(totals) / len(totals) if totals else None,
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
        if clip.dbfs() < self.config.stt.silence_threshold_db:
            self._emit(RecordingDiscarded("silence"))
            return
        t0 = time.perf_counter()
        text = self.transcriber.transcribe(clip)
        stt_seconds = time.perf_counter() - t0
        if not text:
            self._emit(HeardNothing())
            return
        if looks_like_hallucination(text):
            self._emit(RecordingDiscarded(f"whisper hallucination {text!r}"))
            return
        self._emit(PlayerSpoke(text))
        self._respond_to_player(text, stt_seconds=stt_seconds)

    def _respond_to_player(self, text: str, stt_seconds: float | None = None) -> None:
        turn_start = time.perf_counter()
        self._set_state_if({State.IDLE}, State.PROCESSING)
        self.history.add_player(text)
        self.transcript.append_turn("PLAYER", text)
        system = build_system_prompt(
            self.character,
            self.adventure,
            self.logbook.tail(self.config.logbook_sessions_in_prompt),
            self.ooc_notes,
        )
        messages = self.history.as_messages()

        timings: dict[str, float] = {}
        reply = (self._stream_reply(system, messages, timings)
                 if self._can_stream() else None)
        if reply is None:  # streaming off, unsupported, or rejected by the server
            timings.clear()
            t = time.perf_counter()
            reply = extract_dialogue(self.llm.chat(system, messages), self.npc_name)
            timings["llm"] = time.perf_counter() - t
            self._record_npc_reply(reply)
            if self.speaker is not None and self._set_state_if({State.PROCESSING},
                                                               State.SPEAKING):
                t = time.perf_counter()
                self.speaker.say(reply)
                timings["speak"] = time.perf_counter() - t
        elif reply:  # streaming already spoke; "" = barged in before any audio
            self._record_npc_reply(reply)
        self._player_turns += 1

        turn = TurnCompleted(
            stt_seconds=stt_seconds,
            llm_first_token_seconds=timings.get("first_token"),
            llm_seconds=timings.get("llm", 0.0),
            speak_seconds=timings.get("speak"),
            total_seconds=time.perf_counter() - turn_start + (stt_seconds or 0.0),
        )
        self._last_turn = turn
        self._turn_totals.append(turn.total_seconds)
        self._emit(turn)

        if (self.config.checkpoint_every_turns > 0
                and self._player_turns % self.config.checkpoint_every_turns == 0):
            self._write_logbook_entry()
            self._emit(LogbookWritten(str(self.logbook.path), "checkpoint"))

    def _can_stream(self) -> bool:
        return (self.config.llm.stream
                and self.speaker is not None
                and hasattr(self.llm, "chat_stream")
                and hasattr(self.speaker, "say_stream"))

    def _stream_reply(self, system: str, messages: list[dict[str, str]],
                      timings: dict[str, float]) -> str | None:
        """Speak sentence-by-sentence while the LLM generates. Returns the
        text to record — the whole reply, or only what was heard if barged
        in — or None when the server rejects streaming (caller falls back).
        SPEAKING covers the entire stream, so barge-in also cancels a reply
        whose audio hasn't started yet."""
        raw: list[str] = []

        def cleaned_sentences():
            def chunks():
                t0 = time.perf_counter()
                for chunk in self.llm.chat_stream(system, messages):
                    timings.setdefault("first_token", time.perf_counter() - t0)
                    raw.append(chunk)
                    self._emit(NpcReplyChunk(chunk))
                    timings["llm"] = time.perf_counter() - t0
                    yield chunk

            for sentence in iter_sentences(chunks()):
                cleaned = strip_decoration(sentence, self.npc_name)
                if cleaned:  # a pure stage direction is never spoken
                    yield cleaned

        self._set_state_if({State.PROCESSING}, State.SPEAKING)
        t_speak = time.perf_counter()
        try:
            spoken, cancelled = self.speaker.say_stream(cleaned_sentences())
        except StreamingNotSupported:
            self._set_state_if({State.SPEAKING}, State.PROCESSING)
            return None
        timings["speak"] = time.perf_counter() - t_speak
        if cancelled:
            return " ".join(spoken)
        return extract_dialogue("".join(raw), self.npc_name)

    def _record_npc_reply(self, reply: str) -> None:
        self.history.add_npc(reply)
        self.transcript.append_turn("NPC", reply)
        self._emit(NpcReplied(self.npc_name, reply))

    def _write_logbook_entry(self) -> None:
        body = self.llm.summarize_session(
            self.transcript.read(),
            self.logbook.tail(self.config.logbook_sessions_in_prompt),
        )
        self.logbook.upsert_entry(self.session_no, date.today().isoformat(), body)
