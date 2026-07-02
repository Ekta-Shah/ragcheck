"""Adapter that wraps a plain Python callable."""

from __future__ import annotations

from collections.abc import Callable

from ragcheck.adapters.base import RAGAdapter, RAGResponse


class FunctionAdapter(RAGAdapter):
    """Wrap any ``(question: str) -> RAGResponse`` callable as a RAGAdapter."""

    def __init__(self, fn: Callable[[str], RAGResponse]) -> None:
        """Store the pipeline function to delegate queries to."""
        self._fn = fn

    def query(self, question: str) -> RAGResponse:
        """Delegate the question to the wrapped function."""
        return self._fn(question)
