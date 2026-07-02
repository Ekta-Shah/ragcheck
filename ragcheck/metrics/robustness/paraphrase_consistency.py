"""Paraphrase consistency: does rephrasing the question change the answer?"""

from __future__ import annotations

from itertools import combinations

from ragcheck.datasets.models import EvalSample
from ragcheck.judge.judge import Judge, load_prompt
from ragcheck.metrics.base import Metric, MetricResult, parallel_map


class ParaphraseConsistency(Metric):
    """Mean pairwise semantic agreement of answers within each paraphrase group.

    Samples sharing a ``paraphrase_group`` id form a group; every answer pair
    in a group is judged EQUIVALENT/DIFFERENT. A group's score is the fraction
    of equivalent pairs; the metric score is the mean over groups. Samples
    outside any group (or in singleton groups) are excluded (NaN).
    """

    name = "paraphrase_consistency"
    requires_llm = True

    def __init__(self, judge: Judge, concurrency: int = 1) -> None:
        """Bind the judge and load the equivalence prompt."""
        self.judge = judge
        self.concurrency = concurrency
        self.prompt = load_prompt("answers_equivalent")

    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score all paraphrase groups; details carry the inconsistent pairs."""
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            if sample.qa.paraphrase_group:
                groups.setdefault(sample.qa.paraphrase_group, []).append(index)
        scorable = {gid: idxs for gid, idxs in groups.items() if len(idxs) >= 2}

        outcomes = parallel_map(
            lambda item: self._score_group(samples, item[1]),
            sorted(scorable.items()),
            self.concurrency,
        )

        per_sample = [float("nan")] * len(samples)
        inconsistent: list[dict[str, str]] = []
        group_scores: list[float] = []
        for (gid, idxs), (group_score, disagreements) in zip(
            sorted(scorable.items()), outcomes, strict=True
        ):
            group_scores.append(group_score)
            inconsistent.extend({"group": gid, **d} for d in disagreements)
            for index in idxs:
                per_sample[index] = group_score

        score = sum(group_scores) / len(group_scores) if group_scores else 0.0
        return MetricResult(
            metric_name=self.name,
            score=score,
            per_sample_scores=per_sample,
            details={
                "n_groups": len(scorable),
                "ungrouped_samples": len(samples) - sum(len(v) for v in scorable.values()),
                "inconsistent_pairs": inconsistent,
            },
            judge_model=self.judge.model,
            prompt_version=self.prompt.version,
        )

    def _score_group(
        self, samples: list[EvalSample], idxs: list[int]
    ) -> tuple[float, list[dict[str, str]]]:
        question = samples[idxs[0]].qa.question  # reference phrasing for the judge
        disagreements: list[dict[str, str]] = []
        agree = 0
        pairs = list(combinations(idxs, 2))
        for a, b in pairs:
            answer_a = samples[a].response.answer
            answer_b = samples[b].response.answer
            if self._equivalent(question, answer_a, answer_b):
                agree += 1
            else:
                disagreements.append({"answer_a": answer_a, "answer_b": answer_b})
        return agree / len(pairs), disagreements

    def _equivalent(self, question: str, answer_a: str, answer_b: str) -> bool:
        raw = self.judge.ask(
            self.prompt,
            metric_name=self.name,
            key_parts=(question, answer_a, answer_b),
            max_tokens=8,
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
        )
        verdict = raw.upper()
        return "EQUIVALENT" in verdict and "DIFFERENT" not in verdict
