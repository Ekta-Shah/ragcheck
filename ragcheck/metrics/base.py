"""Metric interface and result model."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ragcheck.datasets.models import EvalSample


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
