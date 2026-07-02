"""Core adapter interface: wrap any RAG pipeline so RAGCheck can evaluate it."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One chunk returned by the retrieval stage."""

    content: str
    source_id: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Everything a pipeline produced for one question."""

    answer: str
    retrieved_chunks: list[RetrievedChunk]
    latencies_ms: dict[str, float] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    refused: bool = False


class RAGAdapter(ABC):
    """Wrap any RAG pipeline so RAGCheck can evaluate it."""

    @abstractmethod
    def query(self, question: str) -> RAGResponse:
        """Answer a single question, returning the answer plus retrieval trace."""
        ...

    def batch_query(self, questions: list[str]) -> list[RAGResponse]:
        """Override for pipelines with native batching. Default: sequential."""
        return [self.query(q) for q in questions]
