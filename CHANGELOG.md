# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Install docs: `evdev` publishes no wheels, so **every** install path compiles
  it and needs `python3-dev` + a compiler — not just `pipx`. The README now
  lists them among the system packages, and no longer claims `uv tool install`
  avoids the problem (uv prefers an existing system Python over downloading a
  managed one; `--managed-python` is the no-sudo way out). ([#2])

[#2]: https://github.com/seetee/npc/issues/2

## [1.0.0] — 2026-07-19

First public release, published to PyPI as
[`ttrpg-npc`](https://pypi.org/project/ttrpg-npc/) (the installed command is
`npc`). Everything below works fully offline.

### Voice pipeline

- Push-to-talk (hold) and tap-to-talk (voice-activity detection) recording via
  evdev — works on Wayland, barge-in interrupts a speaking NPC.
- Speech-to-text with faster-whisper, Swedish/English auto-detect, with a
  three-layer guard against whisper hallucinations on near-silence (energy
  gate, no-speech segment filter, phantom-phrase blocklist).
- Local LLM replies via Ollama (native) or any OpenAI-compatible server
  (Jan, LM Studio, llama.cpp, vLLM), with timeouts and retry.
- Text-to-speech with Piper (British English `en_GB-alba-medium` by default),
  streamed sentence-by-sentence for ~3.4× faster time-to-first-sound.
- Replies locked to in-character, voice-only English: narration/stage-direction
  sanitizer and a foreign-language re-ask that survives direct baiting.

### At the table

- Multiple NPCs per campaign (`characters/`, `/npc <name>` switching), each
  with its own Piper voice, memory, logbook, and session summaries.
- GM-gated secrets: locked clue bodies never enter the prompt until the GM
  approves at the console (`/yes`, `/no`, `/later`, `/reveal`, `/secrets`) —
  the model mechanically cannot leak text it never saw.
- Lore documents (`lore/`, .md/.txt/.pdf) injected as established fact, with
  `[llm] num_ctx` support and doctor measuring the real prompt size.
- OBS/table overlay over WebSocket with an opt-in LAN mode for a tablet at
  the table; DM-only events never cross the network.
- Crash-safe session transcripts and per-session logbook summaries.

### Tooling & onboarding

- `npc init` guided campaign wizard, campaign-aware quick-start on `npc run`,
  grouped `/help`, did-you-mean command suggestions.
- `npc doctor [--fix]` checks (and safely sets up) models, voices, audio, and
  permissions; prints the exact `num_ctx` to configure.
- `npc run --timings` per-stage latency instrumentation; last/avg in `/status`.
- Debug helpers: `npc say`, `npc transcribe`.
- Example campaign gallery (`examples/`), recorded demo, `ARCHITECTURE.md`.

### Pre-release history

Development milestones before this release were tracked internally as
v1.x (timeouts, hallucination guard, streaming), v2.0 (voice-only hardening,
tap-to-talk, overlay, doctor --fix, timings), v2.1 (multiple NPCs, gated
secrets, UTF-8 + onboarding, lore), and v2.2 (LAN overlay, docs) — see
`ROADMAP.md` for the map.

[1.0.0]: https://github.com/seetee/npc/releases/tag/v1.0.0
