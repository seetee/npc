# Contributing

Thanks for your interest! This is a small project, but contributions are
welcome — bug reports, campaign examples, and code alike.

## Development setup

Requirements: Linux, Python 3.11+, [uv](https://docs.astral.sh/uv/), and the
system package `libportaudio2` (for sounddevice). A local
[Ollama](https://ollama.com) install is only needed for integration tests and
actual play.

```bash
git clone https://github.com/seetee/npc.git
cd npc
uv sync --group dev            # add `--extra cuda` for GPU whisper
uv run npc --version
```

## Tests

```bash
uv run pytest                  # unit tests — hardware-free, fakes injected
uv run pytest -m integration   # real whisper/piper/ollama smoke tests
                               # (auto-skip when the services are missing)
uv run ruff check .            # lint
```

Unit tests must stay hardware-free: the STT/LLM/TTS/recorder seams are all
Protocol-typed and faked in `tests/test_app_pipeline.py` — follow that pattern
for new features.

## Prompt changes need probing

The system prompt is only as good as its measured behavior against a real
model. If you touch prompt wording, re-run the relevant probe against live
Ollama and include the numbers in your PR:

- `uv run python scripts/probe_narration.py` — after changes to
  `ROLE_FRAMING` / narration handling in `src/npc/session/prompt.py`.
- `uv run python scripts/probe_secrets.py` — after changes to
  `locked_block` / `deny_note` in `src/npc/session/secrets.py`.

## Architecture in one paragraph

The app never prints: `NPCApp` (`src/npc/app.py`) emits typed events
(`src/npc/events.py`) to a single callback; the CLI renders them, tests assert
on them, and the overlay broadcasts them. One worker thread serializes
STT → LLM → TTS turns. Read `ARCHITECTURE.md` for the full picture and
`CLAUDE.md` for the invariants that must not break (per-NPC memory isolation,
the secrets leak guarantee, the single shared `AudioPlayer`, DM-only event
tiers on the overlay).

## Pull requests

- Keep `uv run pytest` and `uv run ruff check .` green.
- New dependencies must be AGPL-3.0-or-later compatible.
- By contributing you agree your work is licensed under the project license,
  AGPL-3.0-or-later (see `LICENSE`).
