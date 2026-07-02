"""Context recall: does the retrieved context cover the ground-truth answer?"""

from __future__ import annotations

from ragcheck.datasets.models import EvalSample
from ragcheck.judge.judge import Judge, load_prompt
from ragcheck.metrics.base import Metric, MetricResult, mean_ignoring_nan, parallel_map
from ragcheck.metrics.generation.faithfulness import _parse_claims


class ContextRecall(Metric):
    """LLM-judged recall: fraction of ground-truth claims attributable to the context.

    The ground-truth answer is decomposed into atomic claims (same prompt as
    faithfulness); each claim is verified against the retrieved chunks. Samples
    without a ``ground_truth_answer`` are skipped.
    """

    name = "context_recall"
    requires_llm = True

    def __init__(self, judge: Judge, concurrency: int = 1) -> None:
        """Bind the judge; reuses the faithfulness decompose/verify prompt pair."""
        self.judge = judge
        self.concurrency = concurrency
        self.decompose_prompt = load_prompt("faithfulness_decompose")
        self.verify_prompt = load_prompt("faithfulness_verify")

    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score all samples; details carry the claims missing from context."""
        outcomes = parallel_map(self._score_sample, samples, self.concurrency)
        per_sample = [score for score, _ in outcomes]
        missing = [m for _, misses in outcomes for m in misses]
        score, scored = mean_ignoring_nan(per_sample)
        return MetricResult(
            metric_name=self.name,
            score=score,
            per_sample_scores=per_sample,
            details={
                "scored": scored,
                "skipped": len(samples) - scored,
                "missing_claims": missing,
            },
            judge_model=self.judge.model,
            prompt_version=self.verify_prompt.version,
        )

    def _score_sample(self, sample: EvalSample) -> tuple[float, list[dict[str, str]]]:
        ground_truth = sample.qa.ground_truth_answer
        if not ground_truth:
            return float("nan"), []
        question = sample.qa.question
        context = "\n\n".join(
            f"[{c.source_id}] {c.content}" for c in sample.response.retrieved_chunks
        )
        raw = self.judge.ask(
            self.decompose_prompt,
            metric_name=self.name,
            key_parts=(question, ground_truth),
            question=question,
            answer=ground_truth,
        )
        claims = _parse_claims(raw)
        if not claims:
            return float("nan"), []
        missing: list[dict[str, str]] = []
        supported = 0
        for claim in claims:
            verdict = self.judge.ask(
                self.verify_prompt,
                metric_name=self.name,
                key_parts=(question, context, claim),
                max_tokens=16,
                context=context,
                claim=claim,
            ).upper()
            if "SUPPORTED" in verdict and "UNSUPPORTED" not in verdict:
                supported += 1
            else:
                missing.append({"question": question, "claim": claim})
        return supported / len(claims), missing
