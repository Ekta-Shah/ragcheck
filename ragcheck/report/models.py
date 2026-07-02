"""Eval report schema."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ragcheck.metrics.base import MetricResult


class LatencySummary(BaseModel):
    """Per-stage latency percentiles across all queries, in milliseconds."""

    stage: str
    p50_ms: float
    p95_ms: float


class ReportSample(BaseModel):
    """One evaluated sample, embedded in the report for failure inspection."""

    question: str
    answer: str
    refused: bool
    answerable: bool
    difficulty: str
    contexts: list[str]  # "[source_id] <truncated content>"
    scores: dict[str, float | None]  # metric name -> per-sample score (None = skipped)


class EvalReport(BaseModel):
    """The full output of one eval run - serialized to JSON in the output dir."""

    run_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset: str
    adapter: str
    n_samples: int
    metrics: list[MetricResult]
    latency: list[LatencySummary] = Field(default_factory=list)
    pipeline_token_usage: dict[str, int] = Field(default_factory=dict)
    judge_token_usage: dict[str, int] = Field(default_factory=dict)
    cache_stats: dict[str, int] = Field(default_factory=dict)
    samples: list[ReportSample] = Field(default_factory=list)
    judge_validation: dict | None = None  # embedded ValidationReport, if configured


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]
