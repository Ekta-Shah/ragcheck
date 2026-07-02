"""Metric interface, result model, and shared scoring helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from pydantic import BaseModel, Field

from ragcheck.datasets.models import EvalSample

T = TypeVar("T")
R = TypeVar("R")


class MetricResult(BaseModel):
    """Aggregate and per-sample scores for one metric over one eval run."""

    metric_name: str
    score: float
    per_sample_scores: list[float]
    details: dict = Field(default_factory=dict)
    judge_model: str | None = None
    prompt_version: str | None = None


class Metric(ABC):
    """A scoring function over evaluated samples, normalized to 0.0-1.0."""

    name: str
    requires_llm: bool

    @abstractmethod
    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score ``samples`` and return an aggregate MetricResult."""
        ...


def mean_ignoring_nan(values: Sequence[float]) -> tuple[float, int]:
    """Mean over non-NaN values plus how many were scored; (0.0, 0) if none."""
    scored = [v for v in values if v == v]
    return (sum(scored) / len(scored) if scored else 0.0), len(scored)


def parallel_map(fn: Callable[[T], R], items: Sequence[T], concurrency: int) -> list[R]:
    """Order-preserving map, threaded when ``concurrency > 1``.

    Used by LLM-judged metrics to overlap judge calls across samples; the
    cache and LLM clients are thread-safe.
    """
    if concurrency <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(fn, items))
