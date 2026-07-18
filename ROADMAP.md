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

### v2.1

- **Multiple NPCs per campaign** ✓ — `characters/` directory, `/npc <name>`
  switching, per-character Piper voices, strictly per-NPC memory and logbooks.
- **GM-gated secrets** ✓ — clues in `secrets.md` the NPC only reveals after
  asking the GM at the console (`/yes`, `/no`, `/later`, `/reveal`,
  `/secrets`); the locked text never enters the prompt until approved, so it
  mechanically cannot leak (`scripts/probe_secrets.py` pins the behavior).
- **UTF-8 everywhere + guided onboarding** ✓ — `npc init` wizard, campaign-
  aware quick-start on `npc run`, grouped `/help`, did-you-mean commands,
  Swedish-safe streams and overlay JSON.
- **Lore (tier 1)** ✓ — per-NPC reference documents (`lore/`, .txt/.md/.pdf)
  injected as established fact; `[llm] num_ctx` with doctor measuring the
  real prompt and printing the exact value to set.

### v2.2

- **LAN overlay opt-in** ✓ — `[overlay] listen` with printed tablet URL and
  warning; beyond loopback only table-safe play events cross the network.
- **Docs that show** ✓ — recorded demo GIF, audio sample, overlay
  screenshot, the `examples/` campaign gallery, `ARCHITECTURE.md`.

## v2.x
- **Release hygiene** — version tags, `CHANGELOG.md`, PyPI packaging,
  `CONTRIBUTING.md`.
- **Silero VAD upgrade** — drop-in replacement for the energy VAD inside
  `VadRecorder` if rooms prove too noisy for a dBFS threshold.
- **Streaming peek-ahead narration filter** — only if the probe starts showing
  quoteless leading narration again (prompt currently holds it at ~1 in 7,
  all quote-marked and caught by the sanitizer).
- **Lore tier 2 (retrieval)** — chunk + embed big documents (Ollama
  embeddings, offline) and inject only relevant passages; deferred until a
  real campaign outgrows a ~16k context window.
