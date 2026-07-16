# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`npc` — a fully-offline terminal voice agent that role-plays a TTRPG NPC at the game table. Hold a push-to-talk key → mic records → faster-whisper transcribes (Swedish/English auto-detect) → a local LLM via Ollama replies in character → Piper speaks it aloud (British English, `en_GB-alba-medium`). Voice input is ALWAYS in-character player dialogue; typed terminal lines are out-of-character GM instructions to the LLM. Design doc: `/home/kenneth/.claude/plans/the-project-is-as-squishy-pond.md`.

## Commands

```bash
uv sync --group dev --extra cuda   # install (cuda extra = GPU whisper; this machine has an RTX 3060)
uv run pytest                      # unit tests — hardware-free, fakes injected
uv run pytest -m integration      # real whisper/piper/ollama smoke tests (auto-skip if missing)
uv run pytest tests/test_app_pipeline.py::test_voice_turn_end_to_end   # single test
uv run ruff check .
uv run npc init|doctor|run <campaign-dir>     # scaffold / setup-check / play
uv run npc say "text" <dir>  /  npc transcribe file.wav <dir>   # debug TTS / STT
```

## Architecture

The pipeline is orchestrated by `src/npc/app.py` (`NPCApp`) — a state machine (`IDLE → RECORDING → PROCESSING → SPEAKING`) with four threads: the prompt_toolkit REPL (main), an evdev hotkey listener (`hotkey.py`), a single worker that serializes STT → LLM → TTS turns via one `queue.Queue`, and sounddevice's callback thread. Push-to-talk during SPEAKING is barge-in (stops playback, starts recording); during PROCESSING it reports busy.

The app never prints: it emits typed events (`events.py` — `PlayerSpoke`, `NpcReplied`, `StateChanged`, `LogbookWritten`, …) to a single `on_event` callback; the CLI renders them with `format_event`, tests assert on event objects, and a future OBS overlay/web remote subscribes to the same stream (`StateChanged` is deliberately not rendered in the terminal). Emit events outside `self._lock`; `_emit` swallows subscriber exceptions so a broken subscriber can't kill a session.

Key seams (all Protocol-typed, faked in `tests/test_app_pipeline.py`):
- `audio/recorder.py` — `Recorder` protocol; v1 `PushToTalkRecorder`. A future VAD recorder (tap to start, silence stops) implements the same protocol and fires `on_auto_stop`; nothing downstream changes.
- `stt.py` / `tts.py` / `llm.py` — `Transcriber`, `Speaker`, and the LLM clients. `llm.py:make_llm_client` picks the backend from `[llm].backend`: native `OllamaClient` (default) or `OpenAICompatClient` for Jan/LM Studio/llama.cpp/vLLM (tested via httpx MockTransport in `test_llm.py`). `stt.py` preloads pip-installed CUDA libs and falls back to CPU if CUDA breaks; resamples non-16kHz input. Whisper hallucinates YouTube outros/subtitle credits on near-silence, so three guards protect the LLM: an energy gate in `app.py:_handle_utterance` (`AudioClip.dbfs()` vs `stt.silence_threshold_db`, skips whisper entirely), a per-segment `no_speech_prob` filter (`stt.py:join_segments`), and the `PHANTOM_PHRASES` blocklist (`looks_like_hallucination`, en+sv, fires only when the WHOLE transcript is phantom).

Prompt assembly (`session/prompt.py`): system prompt is rebuilt every turn from `character.md` + `adventure.md` + logbook tail + accumulated OOC notes. OOC lines appear twice on purpose — inline in history (timing) and in the standing-instructions block (survives history trimming, `session/history.py`). The logbook (`session/logbook.py`) upserts one `## Session N — date` section per session (checkpoints/`/save`/`/end` re-summarize and replace, never duplicate); raw turns are appended crash-safe to `sessions/*-transcript.md`. Logbook writes are hardened against three failure modes — keep them that way: atomic temp-file+`os.replace` (crash can't truncate), `re.sub` with a lambda (backslashes in LLM output must stay literal), and demotion of `## Session N` lines inside summary bodies (would corrupt section parsing).

A campaign directory (see `src/npc/templates/`) is the unit of play; `config.py` loads its optional `config.toml` over dataclass defaults. The NPC's display name is the first `# heading` of `character.md`. `/reload` re-reads the markdown files AND `config.toml`: the LLM model applies live (it's per-request), while `[stt]`/`[tts]`/`[hotkey]` changes are flagged as needing a restart (`app.py:_reload_config`).

## Gotchas

- Terminals never deliver key-up events — that's why hotkeys use evdev (works on Wayland, needs the user in the `input` group). `evdev.list_devices()` silently hides devices you lack permission for; `find_ptt_devices` compensates.
- A held spacebar still types spaces into the REPL; `cli.py:_ptt_callbacks` snapshots/restores the prompt buffer around each press, and treats a press as typing (no recording) when the line already has text — so spaces typed inside `/say …` or an OOC note never trigger PTT. Both mitigations apply only to typing keys (`_key_types_text`); F12/right-ctrl/a grabbed USB button still allow PTT mid-line. A dedicated USB button should set `hotkey.grab = true`.
- In `config.toml` templates, top-level keys must stay above `[sections]` (TOML semantics).
- `sounddevice` needs the system package `libportaudio2` on Linux.
- Replies must always be English (Alba voice) even for Swedish input — enforced in `ROLE_FRAMING`, tested in `test_prompt.py`.
- Replies must be voice-only: `ROLE_FRAMING` forbids narration/stage directions/assistant behavior, and `session/prompt.py:extract_dialogue` strips what small models slip through anyway (`*actions*`, `(parentheticals)`, speaker labels, quote wrapping) before the reply reaches history/transcript/TTS.

License is AGPL-3.0-or-later — new dependencies must be compatible (Piper is GPL-3.0, fine).
