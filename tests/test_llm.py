import httpx
import pytest

from ragcheck.llm import GroqClient, build_client


def groq_transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def make_client(monkeypatch, handler):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return GroqClient(http_client=groq_transport(handler), max_retries=2)


def test_groq_client_parses_response_and_tracks_usage(monkeypatch):
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "model": "llama-3.3-70b-versatile",
                "choices": [{"message": {"content": "SUPPORTED"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    client = make_client(monkeypatch, handler)
    result = client.complete("is it supported?")
    assert result.text == "SUPPORTED"
    assert (result.input_tokens, result.output_tokens) == (12, 3)
    assert client.total_input_tokens == 12


def test_groq_client_retries_on_429(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    monkeypatch.setattr("time.sleep", lambda s: None)
    client = make_client(monkeypatch, handler)
    assert client.complete("q").text == "ok"
    assert calls["n"] == 2


def test_groq_client_does_not_retry_client_errors(monkeypatch):
    def handler(request):
        return httpx.Response(400, json={"error": "bad request"})

    client = make_client(monkeypatch, handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("q")


def test_groq_client_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqClient()


def test_build_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_client("openai")
