"""Metric registry."""

from __future__ import annotations

from collections.abc import Callable

from ragcheck.judge.judge import Judge
from ragcheck.metrics.base import Metric, MetricResult
from ragcheck.metrics.generation.citation_accuracy import CitationAccuracy
from ragcheck.metrics.generation.faithfulness import Faithfulness
from ragcheck.metrics.generation.relevance import AnswerRelevance
from ragcheck.metrics.retrieval.context_precision import ContextPrecision
from ragcheck.metrics.retrieval.context_recall import ContextRecall
from ragcheck.metrics.retrieval.hit_rate import HitRate
from ragcheck.metrics.retrieval.mrr import MRR
from ragcheck.metrics.robustness.paraphrase_consistency import ParaphraseConsistency
from ragcheck.metrics.robustness.refusal_calibration import RefusalCalibration

__all__ = [
    "MRR",
    "AnswerRelevance",
    "CitationAccuracy",
    "ContextPrecision",
    "ContextRecall",
    "Faithfulness",
    "HitRate",
    "Metric",
    "MetricResult",
    "ParaphraseConsistency",
    "RefusalCalibration",
    "build_metric",
]

_DETERMINISTIC: dict[str, Callable[..., Metric]] = {"hit_rate": HitRate, "mrr": MRR}
_LLM_JUDGED: dict[str, Callable[..., Metric]] = {
    "faithfulness": Faithfulness,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "answer_relevance": AnswerRelevance,
    "citation_accuracy": CitationAccuracy,
    "refusal_calibration": RefusalCalibration,
    "paraphrase_consistency": ParaphraseConsistency,
}


def build_metric(name: str, params: dict, judge: Judge | None, concurrency: int = 1) -> Metric:
    """Instantiate a metric by config name.

    LLM-judged metrics receive the shared ``judge`` and ``concurrency``;
    deterministic metrics only get their ``params``.
    """
    if name in _DETERMINISTIC:
        return _DETERMINISTIC[name](**params)
    if name in _LLM_JUDGED:
        if judge is None:
            raise ValueError(f"Metric {name!r} requires an LLM judge")
        return _LLM_JUDGED[name](judge, concurrency=concurrency, **params)
    known = sorted(_DETERMINISTIC | _LLM_JUDGED)
    raise ValueError(f"Unknown metric {name!r}. Available: {known}")
