"""reranked_rag: dense top-20 -> cross-encoder rerank -> top-5 -> generate."""

from __future__ import annotations

import time

from common import TOP_K, Chunk, DenseIndex, generate_answer
from sentence_transformers import CrossEncoder

from ragcheck.adapters.base import RAGAdapter, RAGResponse
from ragcheck.llm import LLMClient

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20


class RerankedRAG(RAGAdapter):
    """Dense candidate retrieval reranked by a cross-encoder, same generator as naive."""

    name = "reranked_rag"

    def __init__(self, chunks: list[Chunk], llm: LLMClient, dense: DenseIndex | None = None):
        self.dense = dense or DenseIndex(chunks)
        self.cross = CrossEncoder(RERANK_MODEL)
        self.llm = llm

    def query(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        candidates = self.dense.query(question, CANDIDATES)
        scores = self.cross.predict([(question, chunk.text) for chunk, _ in candidates])
        reranked = sorted(
            zip(candidates, scores, strict=True), key=lambda item: float(item[1]), reverse=True
        )
        retrieved = [(chunk, float(score)) for (chunk, _), score in reranked[:TOP_K]]
        retrieval_ms = (time.perf_counter() - t0) * 1000
        return generate_answer(self.llm, question, retrieved, retrieval_ms)
