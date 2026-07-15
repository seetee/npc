import httpx
import pytest

from npc.config import ConfigError, LlmConfig
from npc.llm import OllamaClient, OpenAICompatClient, make_llm_client


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


def test_factory_backends():
    assert isinstance(make_llm_client(LlmConfig()), OllamaClient)
    for alias in ("openai", "openai-compatible", "jan", "lmstudio", "OpenAI"):
        client = make_llm_client(LlmConfig(backend=alias, host="http://x:1337"))
        assert isinstance(client, OpenAICompatClient)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ConfigError, match="unknown llm backend"):
        make_llm_client(LlmConfig(backend="skynet"))
