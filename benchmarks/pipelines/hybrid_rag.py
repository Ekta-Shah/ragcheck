"""hybrid_rag: BM25 + dense retrieval fused with reciprocal rank fusion."""

from __future__ import annotations

import time

from common import TOP_K, BM25Index, Chunk, DenseIndex, generate_answer

from ragcheck.adapters.base import RAGAdapter, RAGResponse
from ragcheck.llm import LLMClient

RRF_K = 60
CANDIDATES = 20


class HybridRAG(RAGAdapter):
    """Reciprocal-rank fusion of BM25 and dense rankings, same generator as naive."""

    name = "hybrid_rag"

    def __init__(self, chunks: list[Chunk], llm: LLMClient, dense: DenseIndex | None = None):
        self.dense = dense or DenseIndex(chunks)
        self.bm25 = BM25Index(chunks)
        self.llm = llm

    def query(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        fused: dict[str, tuple[Chunk, float]] = {}
        for ranking in (
            self.dense.query(question, CANDIDATES),
            self.bm25.query(question, CANDIDATES),
        ):
            for rank, (chunk, _) in enumerate(ranking):
                score = 1.0 / (RRF_K + rank + 1)
                if chunk.id in fused:
                    fused[chunk.id] = (chunk, fused[chunk.id][1] + score)
                else:
                    fused[chunk.id] = (chunk, score)
        retrieved = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:TOP_K]
        retrieval_ms = (time.perf_counter() - t0) * 1000
        return generate_answer(self.llm, question, retrieved, retrieval_ms)
