"""Citation accuracy: do [source_id]-style citations actually support their sentences?"""

from __future__ import annotations

import re

from ragcheck.datasets.models import EvalSample
from ragcheck.judge.judge import Judge, load_prompt
from ragcheck.metrics.base import Metric, MetricResult, mean_ignoring_nan, parallel_map

CITATION_RE = re.compile(r"\[([A-Za-z0-9][\w.\-]*)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def extract_citations(answer: str) -> list[tuple[str, str]]:
    """Return (sentence, cited_source_id) pairs found in the answer."""
    pairs: list[tuple[str, str]] = []
    for sentence in SENTENCE_SPLIT_RE.split(answer):
        sentence = sentence.strip()
        for source_id in CITATION_RE.findall(sentence):
            pairs.append((sentence, source_id))
    return pairs


class CitationAccuracy(Metric):
    """LLM-judged support of each cited sentence by its cited chunk.

    A citation of a source_id that was never retrieved counts as inaccurate.
    Samples whose answers contain no citations are skipped (reported in
    ``details.skipped``) - use faithfulness for uncited answers.
    """

    name = "citation_accuracy"
    requires_llm = True

    def __init__(self, judge: Judge, concurrency: int = 1) -> None:
        """Bind the judge and load the citation-support prompt."""
        self.judge = judge
        self.concurrency = concurrency
        self.prompt = load_prompt("citation_support")

    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score all samples; details carry every failed citation."""
        outcomes = parallel_map(self._score_sample, samples, self.concurrency)
        per_sample = [score for score, _ in outcomes]
        failed = [f for _, failures in outcomes for f in failures]
        score, scored = mean_ignoring_nan(per_sample)
        return MetricResult(
            metric_name=self.name,
            score=score,
            per_sample_scores=per_sample,
            details={
                "scored": scored,
                "skipped": len(samples) - scored,
                "failed_citations": failed,
            },
            judge_model=self.judge.model,
            prompt_version=self.prompt.version,
        )

    def _score_sample(self, sample: EvalSample) -> tuple[float, list[dict[str, str]]]:
        citations = extract_citations(sample.response.answer)
        if not citations:
            return float("nan"), []
        chunks = {c.source_id: c.content for c in sample.response.retrieved_chunks}
        failures: list[dict[str, str]] = []
        supported = 0
        for sentence, source_id in citations:
            chunk = chunks.get(source_id)
            if chunk is not None and self._supports(sentence, source_id, chunk):
                supported += 1
            else:
                reason = "source not retrieved" if chunk is None else "chunk does not support"
                failures.append({"sentence": sentence, "source_id": source_id, "reason": reason})
        return supported / len(citations), failures

    def _supports(self, sentence: str, source_id: str, chunk: str) -> bool:
        raw = self.judge.ask(
            self.prompt,
            metric_name=self.name,
            key_parts=(sentence, source_id, chunk),
            max_tokens=16,
            sentence=sentence,
            chunk=chunk,
        )
        verdict = raw.upper()
        return "SUPPORTED" in verdict and "UNSUPPORTED" not in verdict
