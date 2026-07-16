"""LLM backends: native Ollama (default) or any OpenAI-compatible server
(Jan, LM Studio, llama.cpp server, vLLM, …), selected via [llm].backend."""

from __future__ import annotations

from .config import ConfigError

SUMMARIZER_SYSTEM = """\
You are the archivist for a tabletop RPG campaign. You will be given the raw
transcript of the CURRENT session (PLAYER lines, GM notes, NPC replies), and
possibly the previous logbook entries as background.

Write a concise summary of the CURRENT session in markdown with exactly these
bold labels:

**Location:** where the players are now / where the scene takes place.
**NPC state:** the NPC's current attitude toward the players and anything the
NPC learned, revealed, or promised THIS session.
**Highlights:** 2-6 bullet points of what actually happened THIS session.
**Open threads:** unresolved hooks, promises, or dangers from THIS session.

HARD RULE: the previous entries are background ONLY, so you understand
references — never copy, repeat, or re-summarize anything from them. Every
statement in your summary must be grounded in the CURRENT session transcript.
A short session gets a short summary; never pad it with old material. Reply
with ONLY the summary body, no heading, no preamble.
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


class LlmError(Exception):
    """An LLM failure with a friendly, actionable message for the GM."""


class StreamingNotSupported(Exception):
    """The server rejected the streaming request; retry without streaming."""


class _ChatClient:
    """Shared behavior; subclasses implement _chat()/_chat_stream() and the
    doctor helpers."""

    model: str
    host: str = ""
    timeout_seconds: float = 60.0
    retries: int = 1
    _server_hint = "is it running?"

    def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        return self._with_retry(lambda: self._chat(system, messages))

    def chat_stream(self, system: str, messages: list[dict[str, str]]):
        """Yield reply chunks as the model generates them. Connection-level
        failures are retried only before the first chunk arrives — once audio
        may be playing there is nothing safe to retry. A server that rejects
        streaming raises StreamingNotSupported (callers fall back to chat())."""
        import httpx

        last: Exception | None = None
        for _ in range(self.retries + 1):
            started = False
            try:
                for chunk in self._chat_stream(system, messages):
                    started = True
                    yield chunk
                return
            except (httpx.TransportError, ConnectionError) as e:
                if started:
                    raise LlmError("the LLM connection dropped mid-reply — try again") from e
                last = e
        raise self._connection_error(last) from last

    def _chat(self, system: str, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def _chat_stream(self, system: str, messages: list[dict[str, str]]):
        raise NotImplementedError

    def _with_retry(self, call):
        """Retry connection-level failures ([llm].retries extra attempts);
        HTTP errors raise immediately. Everything surfaces as LlmError."""
        import httpx

        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                return call()
            except (httpx.TransportError, ConnectionError) as e:
                last = e
        raise self._connection_error(last) from last

    def _connection_error(self, cause: Exception | None) -> LlmError:
        import httpx

        if isinstance(cause, httpx.TimeoutException):
            return LlmError(
                f"the LLM gave no answer within {self.timeout_seconds:g}s — a large "
                f"model may still be loading; try again, or raise timeout_seconds "
                f"under [llm] in config.toml")
        return LlmError(
            f"cannot reach the LLM server at {self.host} — {self._server_hint}")

    def summarize_session(self, transcript: str, logbook_tail: str) -> str:
        prompt = ""
        if logbook_tail.strip():
            prompt += ("Previous logbook entries (background ONLY — do not "
                       f"repeat these):\n\n{logbook_tail.strip()}\n\n---\n\n")
        prompt += ("CURRENT session transcript (summarize ONLY this):\n\n"
                   f"{transcript.strip()}")
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
    _server_hint = "is Ollama running? (`ollama serve`, check with `ollama ps`)"

    def __init__(self, host: str, model: str, timeout_seconds: float = 60.0,
                 retries: int = 1, transport=None):
        import ollama

        self.host = host
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        # extra kwargs are forwarded to the underlying httpx.Client
        self._client = ollama.Client(host=host, timeout=timeout_seconds,
                                     transport=transport)

    def _chat(self, system: str, messages: list[dict[str, str]]) -> str:
        import ollama

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
            )
        except ollama.ResponseError as e:
            raise self._response_error(e) from e
        return response["message"]["content"].strip()

    def _chat_stream(self, system: str, messages: list[dict[str, str]]):
        import ollama

        try:
            parts = self._client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                stream=True,
            )
            for part in parts:
                content = part["message"]["content"]
                if content:
                    yield content
        except ollama.ResponseError as e:
            raise self._response_error(e) from e

    def _response_error(self, e) -> LlmError:
        message = f"Ollama error: {e.error}"
        if e.status_code == 404:
            message += f" — pull the model with `ollama pull {self.model}`"
        return LlmError(message)

    def pull_model(self, progress=lambda status: None) -> None:
        """Download self.model via the Ollama server (used by doctor --fix);
        progress receives human-readable status lines."""
        for part in self._client.pull(self.model, stream=True):
            status = part.status or ""
            if part.total:
                status += f" {100 * (part.completed or 0) / part.total:.0f}%"
            progress(status)

    def available_models(self) -> list[str]:
        return [m.model for m in self._client.list().models]

    def has_model(self) -> bool:
        wanted = self.model if ":" in self.model else self.model + ":latest"
        return any(name == wanted or name == self.model for name in self.available_models())


class OpenAICompatClient(_ChatClient):
    """Speaks the OpenAI chat-completions dialect served by Jan, LM Studio,
    llama.cpp server, vLLM — and Ollama itself at :11434/v1."""

    _server_hint = "is your server (Jan / LM Studio / llama.cpp / vLLM) running?"

    def __init__(self, host: str, model: str, timeout_seconds: float = 60.0,
                 retries: int = 1, api_key: str = "", transport=None):
        import httpx

        base = host.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.base_url = base
        self.host = host
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        # one client, so the header covers chat, chat_stream, and /models alike
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._http = httpx.Client(base_url=base, timeout=timeout_seconds,
                                  headers=headers, transport=transport)

    def _chat(self, system: str, messages: list[dict[str, str]]) -> str:
        import httpx

        response = self._http.post("/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
        })
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = e.response.text.strip()[:200]
            raise LlmError(f"LLM server returned {e.response.status_code} "
                           f"for model {self.model!r}: {body}") from e
        return response.json()["choices"][0]["message"]["content"].strip()

    def _chat_stream(self, system: str, messages: list[dict[str, str]]):
        import json

        with self._http.stream("POST", "/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
        }) as response:
            if response.status_code >= 400:
                response.read()
                raise StreamingNotSupported(
                    f"{response.status_code}: {response.text.strip()[:200]}")
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or []
                if not choices:
                    continue  # e.g. the final usage-only chunk
                content = choices[0].get("delta", {}).get("content")
                if content:
                    yield content

    def available_models(self) -> list[str]:
        response = self._http.get("/models")
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]

    def has_model(self) -> bool:
        return self.model in self.available_models()


def make_llm_client(llm_config) -> _ChatClient:
    backend = _BACKEND_ALIASES.get(llm_config.backend.lower().strip())
    if backend == "ollama":
        return OllamaClient(llm_config.host, llm_config.model,
                            llm_config.timeout_seconds, llm_config.retries)
    if backend == "openai-compatible":
        return OpenAICompatClient(llm_config.host, llm_config.model,
                                  llm_config.timeout_seconds, llm_config.retries,
                                  api_key=llm_config.api_key)
    raise ConfigError(
        f"unknown llm backend {llm_config.backend!r} — use \"ollama\" or "
        f"\"openai\" (aliases: {', '.join(sorted(set(_BACKEND_ALIASES) - {'ollama'}))})"
    )
