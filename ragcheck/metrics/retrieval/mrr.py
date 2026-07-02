"""Mean reciprocal rank of the first relevant retrieved chunk."""

from __future__ import annotations

from ragcheck.datasets.models import EvalSample
from ragcheck.metrics.base import Metric, MetricResult, mean_ignoring_nan


class MRR(Metric):
    """Reciprocal rank of the first relevant chunk, averaged over labeled samples.

    Unlabeled samples (no ``relevant_source_ids``) are excluded from the
    aggregate and reported in ``details.skipped``.
    """

    name = "mrr"
    requires_llm = False

    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score each labeled sample 1/rank of its first relevant chunk (0 if absent)."""
        per_sample: list[float] = []
        skipped = 0
        for sample in samples:
            relevant = set(sample.qa.relevant_source_ids)
            if not relevant:
                skipped += 1
                per_sample.append(float("nan"))
                continue
            rr = 0.0
            for rank, chunk in enumerate(sample.response.retrieved_chunks, start=1):
                if chunk.source_id in relevant:
                    rr = 1.0 / rank
                    break
            per_sample.append(rr)
        score, scored = mean_ignoring_nan(per_sample)
        return MetricResult(
            metric_name=self.name,
            score=score,
            per_sample_scores=per_sample,
            details={"skipped": skipped, "scored": scored},
        )
