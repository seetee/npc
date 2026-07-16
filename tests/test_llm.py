import json

import httpx
import pytest

from npc.config import ConfigError, LlmConfig
from npc.llm import (
    LlmError,
    OllamaClient,
    OpenAICompatClient,
    StreamingNotSupported,
    make_llm_client,
)


def make_client(handler, host="http://localhost:1337", model="qwen2.5-7b"):
    return OpenAICompatClient(host, model, transport=httpx.MockTransport(handler))


def test_host_gets_v1_suffix():
    client = make_client(lambda r: httpx.Response(200))
    assert client.base_url == "http://localhost:1337/v1"
    client2 = OpenAICompatClient("http://localhost:1234/v1/", "m",
                                 transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert client2.base_url == "http://localhost:1234/v1"


def test_chat_payload_and_reply():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant",
                                     "content": "  Greetings, traveler.  "}}],
        })

    client = make_client(handler)
    reply = client.chat("SYSTEM", [{"role": "user", "content": "hi"}])
    assert reply == "Greetings, traveler."
    assert seen["path"] == "/v1/chat/completions"
    assert seen["json"]["model"] == "qwen2.5-7b"
    assert seen["json"]["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert seen["json"]["messages"][1] == {"role": "user", "content": "hi"}


def test_models_listing_and_is_up():
    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "qwen2.5-7b"},
                                                  {"id": "llama-3.1-8b"}]})

    client = make_client(handler)
    assert client.is_up()
    assert client.has_model()
    assert client.available_models() == ["qwen2.5-7b", "llama-3.1-8b"]

    down = make_client(lambda r: httpx.Response(503))
    assert not down.is_up()


def reply_ok(text="Hello."):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}}],
    })


def test_timeout_once_is_retried_transparently():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow model")
        return reply_ok()

    client = make_client(handler)
    assert client.chat("S", []) == "Hello."
    assert calls["n"] == 2


def test_timeout_always_raises_friendly_error_after_retries():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("slow model")

    client = make_client(handler)
    with pytest.raises(LlmError, match="timeout_seconds"):
        client.chat("S", [])
    assert calls["n"] == 2  # retries=1 → two attempts, then give up


def test_connect_error_names_the_host():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = make_client(handler)
    with pytest.raises(LlmError, match="localhost:1337"):
        client.chat("S", [])


def test_http_error_is_never_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client = make_client(handler)
    with pytest.raises(LlmError, match="400"):
        client.chat("S", [])
    assert calls["n"] == 1


def test_ollama_missing_model_suggests_pull():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'qwen' not found"})

    client = OllamaClient("http://localhost:11434", "qwen",
                          transport=httpx.MockTransport(handler))
    with pytest.raises(LlmError, match="ollama pull qwen"):
        client.chat("S", [])


def test_ollama_connection_error_is_friendly():
    def handler(request):
        raise httpx.ConnectError("refused")  # ollama wraps this in ConnectionError

    client = OllamaClient("http://localhost:11434", "qwen",
                          transport=httpx.MockTransport(handler))
    with pytest.raises(LlmError, match="ollama serve"):
        client.chat("S", [])


def sse(*chunks):
    lines = [f'data: {json.dumps({"choices": [{"delta": {"content": c}}]})}'
             for c in chunks]
    lines.append('data: {"choices": []}')  # usage-only final chunk some servers send
    lines.append("data: [DONE]")
    return httpx.Response(200, content="\n".join(lines).encode(),
                          headers={"content-type": "text/event-stream"})


def test_chat_stream_parses_sse_until_done():
    seen = {}

    def handler(request):
        seen["json"] = json.loads(request.content)
        return sse("Hel", "lo.")

    client = make_client(handler)
    assert list(client.chat_stream("S", [])) == ["Hel", "lo."]
    assert seen["json"]["stream"] is True


def test_chat_stream_rejection_raises_streaming_not_supported():
    client = make_client(lambda r: httpx.Response(400, text="streaming not allowed"))
    with pytest.raises(StreamingNotSupported):
        list(client.chat_stream("S", []))


def test_chat_stream_retries_before_first_token():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused")
        return sse("Hi.")

    client = make_client(handler)
    assert list(client.chat_stream("S", [])) == ["Hi."]
    assert calls["n"] == 2


def test_chat_stream_dead_server_raises_friendly_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = make_client(handler)
    with pytest.raises(LlmError, match="localhost:1337"):
        list(client.chat_stream("S", []))


def test_ollama_chat_stream_yields_content():
    body = (b'{"message":{"role":"assistant","content":"Hel"},"done":false}\n'
            b'{"message":{"role":"assistant","content":"lo."},"done":true}\n')
    client = OllamaClient(
        "http://localhost:11434", "m",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    assert list(client.chat_stream("S", [])) == ["Hel", "lo."]


def test_factory_passes_timeout_and_retries():
    config = LlmConfig(timeout_seconds=5.0, retries=2)
    client = make_llm_client(config)
    assert (client.timeout_seconds, client.retries) == (5.0, 2)
    compat = make_llm_client(LlmConfig(backend="openai", timeout_seconds=5.0, retries=2))
    assert (compat.timeout_seconds, compat.retries) == (5.0, 2)
    assert compat._http.timeout == httpx.Timeout(5.0)


def test_factory_backends():
    assert isinstance(make_llm_client(LlmConfig()), OllamaClient)
    for alias in ("openai", "openai-compatible", "jan", "lmstudio", "OpenAI"):
        client = make_llm_client(LlmConfig(backend=alias, host="http://x:1337"))
        assert isinstance(client, OpenAICompatClient)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ConfigError, match="unknown llm backend"):
        make_llm_client(LlmConfig(backend="skynet"))
