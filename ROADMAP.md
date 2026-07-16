# Roadmap

## Shipped

### v1.x

- **LLM timeouts + retry** ✓ — a hung or hiccuping LLM server degrades into a
  friendly error, never a stuck session.
- **Whisper hallucination guard** ✓ — energy gate on near-silent clips, a
  no-speech segment filter, and a blocklist for whisper's known phantom
  phrases (English and Swedish).
- **Streaming replies** ✓ — sentence-by-sentence LLM → TTS → playback;
  time-to-first-sound measured 3.4× faster (1.5 s → 0.4 s on an RTX 3060
  with qwen2.5:7b).

### v2.0 stage 1

- **Voice-only replies hardened** ✓ — narration ban with a few-shot example in
  the prompt plus quoted-dialogue extraction in the sanitizer
  (`scripts/probe_narration.py` re-checks any prompt/model change).
- **Tap-to-talk with voice-activity detection** ✓ — `[hotkey] mode = "tap"`:
  press once, trailing silence ends the recording (energy-based `VadRecorder`).
- **OBS / table overlay** ✓ — `[overlay]` config or `npc run --overlay`: a
  localhost WebSocket broadcasting the event stream plus a bundled HTML page.
- **`npc doctor --fix`** ✓ — interactive Ollama model pull and Piper voice
  download; sudo-level fixes stay copy-paste-only.
- **Latency instrumentation** ✓ — `TurnCompleted` per-stage timings,
  `npc run --timings`, last/avg in `/status`.
- **`api_key`** ✓ for the OpenAI-compatible backend (`NPC_LLM_API_KEY`).

## v2.x

- **Multiple NPCs per campaign** — a `characters/` directory, `/npc <name>` to
  switch, optionally a different Piper voice per character.
- **Docs that show, not tell** — demo GIF, audio samples, a gallery of
  ready-made campaign folders, `ARCHITECTURE.md`.
- **Release hygiene** — version tags, `CHANGELOG.md`, PyPI packaging,
  `CONTRIBUTING.md`.
- **Silero VAD upgrade** — drop-in replacement for the energy VAD inside
  `VadRecorder` if rooms prove too noisy for a dBFS threshold.
- **Streaming peek-ahead narration filter** — only if the probe starts showing
  quoteless leading narration again (prompt currently holds it at ~1 in 7,
  all quote-marked and caught by the sanitizer).
- **LAN overlay binding** — a table-display mode with explicit opt-in and its
  own security thinking; the overlay deliberately binds 127.0.0.1 only today.
