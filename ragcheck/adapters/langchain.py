"""Adapter for LangChain retriever + chain pipelines.

LangChain is not a dependency of ragcheck; objects are duck-typed. The
retriever must expose ``invoke(question) -> list[Document]`` (LangChain's
standard retriever interface) and the chain ``invoke(inputs) -> str | Message``.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from ragcheck.adapters.base import RAGAdapter, RAGResponse, RetrievedChunk


class _Invokable(Protocol):
    def invoke(self, __input: Any) -> Any: ...


class LangChainAdapter(RAGAdapter):
    """Wrap a LangChain retriever + generation chain as a RAGAdapter.

    The chain is invoked with ``{"question": ..., "context": ...}`` where
    context is the concatenated retrieved documents. Document source ids come
    from ``doc.metadata[source_key]`` (falling back to a positional id).
    """

    def __init__(
        self,
        retriever: _Invokable,
        chain: _Invokable,
        source_key: str = "source",
    ) -> None:
        """Bind a retriever and a chain; ``source_key`` names the id metadata field."""
        self.retriever = retriever
        self.chain = chain
        self.source_key = source_key

    def query(self, question: str) -> RAGResponse:
        """Retrieve documents, run the chain, and normalize into a RAGResponse."""
        t0 = time.perf_counter()
        documents = self.retriever.invoke(question)
        t1 = time.perf_counter()
        chunks = [
            RetrievedChunk(
                content=getattr(doc, "page_content", str(doc)),
                source_id=str(getattr(doc, "metadata", {}).get(self.source_key, f"doc_{i}")),
            )
            for i, doc in enumerate(documents)
        ]
        context = "\n\n".join(f"[{c.source_id}] {c.content}" for c in chunks)
        raw = self.chain.invoke({"question": question, "context": context})
        t2 = time.perf_counter()
        answer = str(getattr(raw, "content", raw)).strip()
        return RAGResponse(
            answer=answer,
            retrieved_chunks=chunks,
            latencies_ms={"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000},
        )
