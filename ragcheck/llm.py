"""LLM client protocol and the Anthropic implementation."""

from __future__ import annotations

from typing import Protocol

import anthropic
from pydantic import BaseModel

DEFAULT_JUDGE_MODEL = "claude-opus-4-8"


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
