"""Thin client around the local Ollama server."""

from __future__ import annotations

SUMMARIZER_SYSTEM = """\
You are the archivist for a tabletop RPG campaign. You will be given the raw
transcript of the current session (PLAYER lines, GM notes, NPC replies) and
the tail of the existing campaign logbook.

Write a concise session summary in markdown with exactly these bold labels:

**Location:** where the players are now / where the scene takes place.
**NPC state:** the NPC's current attitude toward the players and anything the
NPC learned, revealed, or promised.
**Highlights:** 2-6 bullet points of what actually happened.
**Open threads:** unresolved hooks, promises, or dangers.

Be factual — only include things that happened in the transcript. Reply with
ONLY the summary body, no heading, no preamble.
"""


class OllamaClient:
    def __init__(self, host: str, model: str):
        import ollama

        self.model = model
        self._client = ollama.Client(host=host)

    def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return response["message"]["content"].strip()

    def summarize_session(self, transcript: str, logbook_tail: str) -> str:
        prompt = ""
        if logbook_tail.strip():
            prompt += f"Existing logbook tail:\n\n{logbook_tail.strip()}\n\n---\n\n"
        prompt += f"Session transcript:\n\n{transcript.strip()}"
        return self.chat(SUMMARIZER_SYSTEM, [{"role": "user", "content": prompt}])

    # --- helpers for doctor ---

    def is_up(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def available_models(self) -> list[str]:
        return [m.model for m in self._client.list().models]

    def has_model(self) -> bool:
        wanted = self.model if ":" in self.model else self.model + ":latest"
        return any(name == wanted or name == self.model for name in self.available_models())
