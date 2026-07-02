"""agentic_rag: LLM query decomposition -> iterative retrieval -> synthesized answer.

Capped at 3 retrieval rounds. Decomposition/sufficiency calls use the same
generator model as every other pipeline; their tokens and latency are charged
to this pipeline's cost (that is the architecture's trade-off).
"""

from __future__ import annotations

import json
import re
import time

from common import Chunk, DenseIndex, generate_answer

from ragcheck.adapters.base import RAGAdapter, RAGResponse
from ragcheck.llm import LLMClient

MAX_ROUNDS = 3
PER_QUERY_K = 3
MAX_CONTEXT_CHUNKS = 8

DECOMPOSE_PROMPT = """Break the question below into 1-3 self-contained search queries that together gather the information needed to answer it. If the question is already atomic, return it as a single query.

Question: {question}

Respond with ONLY a JSON array of query strings."""

SUFFICIENCY_PROMPT = """You are gathering evidence to answer a question.

Question: {question}

Snippets gathered so far:
{snippets}

If the snippets contain enough information to answer the question, respond with exactly: SUFFICIENT
Otherwise respond with ONE additional search query (a short phrase, no other text) that would find the missing information."""


class AgenticRAG(RAGAdapter):
    """Query decomposition with iterative dense retrieval, same generator as naive."""

    name = "agentic_rag"

    def __init__(self, chunks: list[Chunk], llm: LLMClient, dense: DenseIndex | None = None):
        self.dense = dense or DenseIndex(chunks)
        self.llm = llm

    def query(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        extra_in = extra_out = 0
        gathered: dict[str, tuple[Chunk, float]] = {}

        completion = self.llm.complete(
            DECOMPOSE_PROMPT.format(question=question), max_tokens=256
        )
        extra_in += completion.input_tokens
        extra_out += completion.output_tokens
        queries = self._parse_queries(completion.text) or [question]

        for query in queries[:3]:
            self._retrieve_into(gathered, query)

        rounds = 1
        while rounds < MAX_ROUNDS:
            snippets = "\n".join(
                f"[{cid}] {chunk.text[:300]}" for cid, (chunk, _) in gathered.items()
            )
            completion = self.llm.complete(
                SUFFICIENCY_PROMPT.format(question=question, snippets=snippets or "(none)"),
                max_tokens=64,
            )
            extra_in += completion.input_tokens
            extra_out += completion.output_tokens
            verdict = completion.text.strip()
            if "SUFFICIENT" in verdict.upper() or not verdict:
                break
            self._retrieve_into(gathered, verdict.splitlines()[0])
            rounds += 1

        retrieved = sorted(gathered.values(), key=lambda item: item[1], reverse=True)
        retrieved = retrieved[:MAX_CONTEXT_CHUNKS]
        retrieval_ms = (time.perf_counter() - t0) * 1000

        response = generate_answer(self.llm, question, retrieved, retrieval_ms)
        response.token_usage["input_tokens"] = (
            response.token_usage.get("input_tokens", 0) + extra_in
        )
        response.token_usage["output_tokens"] = (
            response.token_usage.get("output_tokens", 0) + extra_out
        )
        return response

    def _retrieve_into(self, gathered: dict[str, tuple[Chunk, float]], query: str) -> None:
        for chunk, score in self.dense.query(query, PER_QUERY_K):
            existing = gathered.get(chunk.id)
            if existing is None or score > existing[1]:
                gathered[chunk.id] = (chunk, score)

    @staticmethod
    def _parse_queries(raw: str) -> list[str]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return [str(q) for q in parsed if str(q).strip()]
