"""Context precision: are the retrieved chunks actually relevant to the question?"""

from __future__ import annotations

from ragcheck.datasets.models import EvalSample
from ragcheck.judge.judge import Judge, load_prompt
from ragcheck.metrics.base import Metric, MetricResult, mean_ignoring_nan, parallel_map


class ContextPrecision(Metric):
    """LLM-judged, rank-aware precision of the top-k retrieved chunks.

    Each chunk is judged RELEVANT/IRRELEVANT to the question; the sample score
    is the mean of precision@i at each relevant position i (rewards ranking
    relevant chunks first). Samples with no retrieved chunks are skipped.
    """

    name = "context_precision"
    requires_llm = True

    def __init__(self, judge: Judge, k: int = 5, concurrency: int = 1) -> None:
        """Judge the top ``k`` chunks per sample with ``concurrency`` parallel samples."""
        self.judge = judge
        self.k = k
        self.concurrency = concurrency
        self.prompt = load_prompt("chunk_relevance")

    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score all samples; details carry the per-position relevance flags."""
        outcomes = parallel_map(self._score_sample, samples, self.concurrency)
        per_sample = [score for score, _ in outcomes]
        score, scored = mean_ignoring_nan(per_sample)
        return MetricResult(
            metric_name=self.name,
            score=score,
            per_sample_scores=per_sample,
            details={
                "k": self.k,
                "scored": scored,
                "skipped": len(samples) - scored,
                "relevance_flags": [flags for _, flags in outcomes],
            },
            judge_model=self.judge.model,
            prompt_version=self.prompt.version,
        )

    def _score_sample(self, sample: EvalSample) -> tuple[float, list[bool]]:
        chunks = sample.response.retrieved_chunks[: self.k]
        if not chunks:
            return float("nan"), []
        flags = [self._is_relevant(sample.qa.question, c.content, c.source_id) for c in chunks]
        relevant_seen = 0
        precision_sum = 0.0
        for position, flag in enumerate(flags, start=1):
            if flag:
                relevant_seen += 1
                precision_sum += relevant_seen / position
        return (precision_sum / relevant_seen if relevant_seen else 0.0), flags

    def _is_relevant(self, question: str, chunk: str, source_id: str) -> bool:
        raw = self.judge.ask(
            self.prompt,
            metric_name=self.name,
            key_parts=(question, source_id, chunk),
            max_tokens=16,
            question=question,
            chunk=chunk,
        )
        verdict = raw.upper()
        return "IRRELEVANT" not in verdict and "RELEVANT" in verdict
