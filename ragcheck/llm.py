"""LLM client protocol and provider implementations (Anthropic, Groq)."""

from __future__ import annotations

import contextlib
import os
import random
import time
from typing import Protocol

import anthropic
import httpx
from pydantic import BaseModel

DEFAULT_JUDGE_MODEL = "claude-opus-4-8"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMResponse(BaseModel):
    """A completed LLM call with token accounting."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient(Protocol):
    """Minimal completion interface so non-Anthropic judges can be added later."""

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 1024
    ) -> LLMResponse:
        """Run a single-turn completion and return text plus token usage."""
        ...


class AnthropicClient:
    """LLMClient backed by the Anthropic Messages API.

    Reads the API key from the ``ANTHROPIC_API_KEY`` environment variable.
    Rate limits and transient server errors are retried with exponential
    backoff by the SDK (``max_retries``). Cumulative token usage is tracked
    on ``total_input_tokens`` / ``total_output_tokens``.
    """

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, max_retries: int = 4) -> None:
        """Create a client for ``model`` with SDK-managed retry/backoff."""
        self._client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 1024
    ) -> LLMResponse:
        """Run a single-turn completion against the configured model."""
        messages: list[anthropic.types.MessageParam] = [{"role": "user", "content": prompt}]
        if system is None:
            response = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, messages=messages
            )
        else:
            response = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, messages=messages
            )
        text = "".join(block.text for block in response.content if block.type == "text")
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )


class GroqClient:
    """LLMClient backed by Groq's OpenAI-compatible chat completions API.

    Serves open-weight models (Llama, etc.). Reads the API key from the
    ``GROQ_API_KEY`` environment variable. Judged calls run at temperature 0.
    Retries 429 and 5xx responses with exponential backoff.
    """

    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        max_retries: int = 8,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Create a client for ``model``; ``http_client`` is injectable for tests."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        self._http = http_client or httpx.Client(timeout=120.0)
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._max_retries = max_retries
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 1024
    ) -> LLMResponse:
        """Run a single-turn completion against the configured model."""
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": messages,
        }
        data = self._post_with_retry(payload)
        usage = data.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=data.get("model", self.model),
        )

    def _post_with_retry(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            delay = min(2**attempt + random.random(), 30.0)
            try:
                response = self._http.post(
                    f"{GROQ_BASE_URL}/chat/completions", json=payload, headers=self._headers
                )
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    return dict(response.json())
                if response.status_code not in (429,) and response.status_code < 500:
                    response.raise_for_status()
                if response.status_code == 429:
                    # Groq's retry-after reflects the TPM window; trust it over backoff.
                    retry_after = response.headers.get("retry-after")
                    if retry_after is not None:
                        with contextlib.suppress(ValueError):
                            delay = min(float(retry_after) + random.random(), 120.0)
                last_error = httpx.HTTPStatusError(
                    f"Groq returned {response.status_code}: {response.text[:300]}",
                    request=response.request,
                    response=response,
                )
            if attempt < self._max_retries:
                time.sleep(delay)
        raise last_error if last_error else RuntimeError("Groq request failed")


def build_client(provider: str, model: str | None = None) -> LLMClient:
    """Instantiate an LLMClient by provider name (``anthropic`` or ``groq``)."""
    if provider == "anthropic":
        return AnthropicClient(model=model or DEFAULT_JUDGE_MODEL)
    if provider == "groq":
        return GroqClient(model=model or DEFAULT_GROQ_MODEL)
    raise ValueError(f"Unknown LLM provider {provider!r} (use 'anthropic' or 'groq')")


def default_client(model: str | None = None) -> LLMClient:
    """Pick a provider from the environment: Anthropic if its key is set, else Groq.

    ``RAGCHECK_MODEL`` overrides the provider's default model when ``model``
    is not given (useful when a free-tier model's daily budget is exhausted).
    """
    model = model or os.environ.get("RAGCHECK_MODEL")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return build_client("anthropic", model)
    if os.environ.get("GROQ_API_KEY"):
        return build_client("groq", model)
    raise RuntimeError("Set ANTHROPIC_API_KEY or GROQ_API_KEY to use LLM-backed components")
