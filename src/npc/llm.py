"""LLM backends: native Ollama (default) or any OpenAI-compatible server
(Jan, LM Studio, llama.cpp server, vLLM, …), selected via [llm].backend."""

from __future__ import annotations

from .config import ConfigError

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

# every alias points at the same OpenAI-compatible client
_BACKEND_ALIASES = {
    "openai": "openai-compatible",
    "openai-compatible": "openai-compatible",
    "jan": "openai-compatible",
    "lmstudio": "openai-compatible",
    "llamacpp": "openai-compatible",
    "vllm": "openai-compatible",
    "ollama": "ollama",
}


class _ChatClient:
    """Shared behavior; subclasses implement chat() and the doctor helpers."""

    model: str

    def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def summarize_session(self, transcript: str, logbook_tail: str) -> str:
        prompt = ""
        if logbook_tail.strip():
            prompt += f"Existing logbook tail:\n\n{logbook_tail.strip()}\n\n---\n\n"
        prompt += f"Session transcript:\n\n{transcript.strip()}"
        return self.chat(SUMMARIZER_SYSTEM, [{"role": "user", "content": prompt}])

    def is_up(self) -> bool:
        try:
            self.available_models()
            return True
        except Exception:
            return False

    def available_models(self) -> list[str]:
        raise NotImplementedError

    def has_model(self) -> bool:
        raise NotImplementedError


class OllamaClient(_ChatClient):
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

    def available_models(self) -> list[str]:
        return [m.model for m in self._client.list().models]

    def has_model(self) -> bool:
        wanted = self.model if ":" in self.model else self.model + ":latest"
        return any(name == wanted or name == self.model for name in self.available_models())


class OpenAICompatClient(_ChatClient):
    """Speaks the OpenAI chat-completions dialect served by Jan, LM Studio,
    llama.cpp server, vLLM — and Ollama itself at :11434/v1."""

    def __init__(self, host: str, model: str, transport=None):
        import httpx

        base = host.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.base_url = base
        self.model = model
        self._http = httpx.Client(base_url=base, timeout=120.0, transport=transport)

    def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        response = self._http.post("/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def available_models(self) -> list[str]:
        response = self._http.get("/models")
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]

    def has_model(self) -> bool:
        return self.model in self.available_models()


def make_llm_client(llm_config) -> _ChatClient:
    backend = _BACKEND_ALIASES.get(llm_config.backend.lower().strip())
    if backend == "ollama":
        return OllamaClient(llm_config.host, llm_config.model)
    if backend == "openai-compatible":
        return OpenAICompatClient(llm_config.host, llm_config.model)
    raise ConfigError(
        f"unknown llm backend {llm_config.backend!r} — use \"ollama\" or "
        f"\"openai\" (aliases: {', '.join(sorted(set(_BACKEND_ALIASES) - {'ollama'}))})"
    )
