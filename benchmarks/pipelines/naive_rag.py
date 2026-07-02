"""naive_rag: single dense retrieval -> top-k -> generate."""

from __future__ import annotations

import time

from common import TOP_K, Chunk, DenseIndex, generate_answer

from ragcheck.adapters.base import RAGAdapter, RAGResponse
from ragcheck.llm import LLMClient


class NaiveRAG(RAGAdapter):
    """Dense retrieval (all-MiniLM-L6-v2 cosine) with no reranking or fusion."""

    name = "naive_rag"

    def __init__(self, chunks: list[Chunk], llm: LLMClient, dense: DenseIndex | None = None):
        self.dense = dense or DenseIndex(chunks)
        self.llm = llm

    def query(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        retrieved = self.dense.query(question, TOP_K)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        return generate_answer(self.llm, question, retrieved, retrieval_ms)
