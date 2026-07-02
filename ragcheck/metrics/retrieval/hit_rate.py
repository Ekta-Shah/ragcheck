"""Hit rate @ k: did any relevant source appear in the top-k retrieved chunks?"""

from __future__ import annotations

from ragcheck.datasets.models import EvalSample
from ragcheck.metrics.base import Metric, MetricResult


class HitRate(Metric):
    """Fraction of questions where a relevant source was retrieved in the top k.

    Samples without ``relevant_source_ids`` labels are excluded from the
    aggregate (reported in ``details.skipped``).
    """

    requires_llm = False

    def __init__(self, k: int = 5) -> None:
        """Configure the cutoff ``k`` for the retrieval window."""
        self.k = k
        self.name = f"hit_rate@{k}"

    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score each labeled sample 1.0 on a hit in the top k, else 0.0."""
        per_sample: list[float] = []
        skipped = 0
        for sample in samples:
            relevant = set(sample.qa.relevant_source_ids)
            if not relevant:
                skipped += 1
                per_sample.append(float("nan"))
                continue
            top_k = {c.source_id for c in sample.response.retrieved_chunks[: self.k]}
            per_sample.append(1.0 if top_k & relevant else 0.0)
        scored = [s for s in per_sample if s == s]  # drop NaN placeholders
        score = sum(scored) / len(scored) if scored else 0.0
        return MetricResult(
            metric_name=self.name,
            score=score,
            per_sample_scores=per_sample,
            details={"k": self.k, "skipped": skipped, "scored": len(scored)},
        )
