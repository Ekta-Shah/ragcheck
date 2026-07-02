"""Metric registry."""

from __future__ import annotations

from collections.abc import Callable

from ragcheck.judge.judge import Judge
from ragcheck.metrics.base import Metric, MetricResult
from ragcheck.metrics.generation.faithfulness import Faithfulness
from ragcheck.metrics.retrieval.hit_rate import HitRate

__all__ = ["Faithfulness", "HitRate", "Metric", "MetricResult", "build_metric"]

_DETERMINISTIC: dict[str, Callable[..., Metric]] = {"hit_rate": HitRate}
_LLM_JUDGED: dict[str, Callable[..., Metric]] = {"faithfulness": Faithfulness}


def build_metric(name: str, params: dict, judge: Judge | None) -> Metric:
    """Instantiate a metric by config name.

    LLM-judged metrics receive the shared ``judge``; deterministic metrics
    only get their ``params``.
    """
    if name in _DETERMINISTIC:
        return _DETERMINISTIC[name](**params)
    if name in _LLM_JUDGED:
        if judge is None:
            raise ValueError(f"Metric {name!r} requires an LLM judge")
        return _LLM_JUDGED[name](judge, **params)
    known = sorted(_DETERMINISTIC | _LLM_JUDGED)
    raise ValueError(f"Unknown metric {name!r}. Available: {known}")
