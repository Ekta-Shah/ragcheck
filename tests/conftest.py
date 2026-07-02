"""Shared fixtures: mocked LLM clients so no test hits the network."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from ragcheck.adapters.base import RAGResponse, RetrievedChunk
from ragcheck.datasets.models import EvalSample, QAPair
from ragcheck.llm import LLMResponse


class MockLLMClient:
    """LLMClient whose responses come from a prompt -> text responder."""

    def __init__(self, responder: Callable[[str], str], model: str = "mock-judge-1"):
        self.responder = responder
        self.model = model
        self.calls: list[str] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024):
        self.calls.append(prompt)
        self.total_input_tokens += 10
        self.total_output_tokens += 5
        return LLMResponse(
            text=self.responder(prompt), input_tokens=10, output_tokens=5, model=self.model
        )


def faithfulness_responder(claims_by_answer: dict[str, list[str]], supported: set[str]):
    """Build a responder that scripts decompose + verify judge calls.

    ``claims_by_answer`` maps an answer substring to its decomposed claims;
    ``supported`` is the set of claims to verdict as SUPPORTED.
    """

    def respond(prompt: str) -> str:
        if "atomic factual claims" in prompt:
            for answer_key, claims in claims_by_answer.items():
                if answer_key in prompt:
                    return json.dumps(claims)
            return "[]"
        claim = prompt.split("Claim:")[1].split("Respond with")[0].strip()
        return "SUPPORTED" if claim in supported else "UNSUPPORTED"

    return respond


@pytest.fixture
def mock_llm_factory():
    return MockLLMClient


def make_sample(
    question: str = "What is X?",
    answer: str = "X is Y.",
    chunk_ids: list[str] | None = None,
    relevant_ids: list[str] | None = None,
    **response_kwargs,
) -> EvalSample:
    chunks = [
        RetrievedChunk(content=f"content of {cid}", source_id=cid) for cid in (chunk_ids or [])
    ]
    return EvalSample(
        qa=QAPair(question=question, relevant_source_ids=relevant_ids or []),
        response=RAGResponse(answer=answer, retrieved_chunks=chunks, **response_kwargs),
    )
