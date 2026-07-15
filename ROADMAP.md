# Roadmap

## v1.x — next up

- **Streaming replies**: sentence-by-sentence LLM → TTS → playback, cutting
  time-to-first-sound roughly in half.
- **Whisper hallucination guard**: energy gate on near-silent clips plus a
  filter for whisper's known phantom phrases.
- **LLM timeouts + retry**: a hung or hiccuping LLM server degrades into a
  friendly error, never a stuck session.

## v2.0

- **Multiple NPCs per campaign** — a `characters/` directory, `/npc <name>` to
  switch, optionally a different Piper voice per character.
- **Tap-to-talk with voice-activity detection** — press once, silence ends the
  recording (a `VadRecorder` behind the existing `Recorder` protocol).
- **OBS / table overlay** — `npc serve`: a local WebSocket broadcasting the
  structured event stream plus a bundled HTML overlay page.
- **`npc doctor --fix`** — interactive downloads/pulls instead of printed
  commands.
- **Latency instrumentation** — per-stage timings (`TurnCompleted` event,
  shown in `/status`).
- **Docs that show, not tell** — demo GIF, audio samples, a gallery of
  ready-made campaign folders, `ARCHITECTURE.md`.
- **Release hygiene** — version tags, `CHANGELOG.md`, PyPI packaging,
  `CONTRIBUTING.md`.
- **`api_key` support** for the OpenAI-compatible backend.
