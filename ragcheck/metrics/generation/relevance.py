"""Answer relevance: does the answer actually address the question?"""

from __future__ import annotations

import re

from ragcheck.datasets.models import EvalSample
from ragcheck.judge.judge import Judge, load_prompt
from ragcheck.metrics.base import Metric, MetricResult, mean_ignoring_nan, parallel_map


class AnswerRelevance(Metric):
    """LLM-judged relevance on a 1-5 rubric, normalized to 0.0-1.0.

    Refusals on answerable questions are penalized by the rubric itself;
    unparseable judge output scores 0.0 and is counted in details.
    """

    name = "answer_relevance"
    requires_llm = True

    def __init__(self, judge: Judge, concurrency: int = 1) -> None:
        """Bind the judge and load the rubric prompt."""
        self.judge = judge
        self.concurrency = concurrency
        self.prompt = load_prompt("answer_relevance")

    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score every sample on the rubric."""
        outcomes = parallel_map(self._score_sample, samples, self.concurrency)
        per_sample = [score for score, _ in outcomes]
        unparseable = sum(1 for _, parsed in outcomes if not parsed)
        score, _ = mean_ignoring_nan(per_sample)
        return MetricResult(
            metric_name=self.name,
            score=score,
            per_sample_scores=per_sample,
            details={"unparseable_verdicts": unparseable},
            judge_model=self.judge.model,
            prompt_version=self.prompt.version,
        )

    def _score_sample(self, sample: EvalSample) -> tuple[float, bool]:
        raw = self.judge.ask(
            self.prompt,
            metric_name=self.name,
            key_parts=(sample.qa.question, sample.response.answer),
            max_tokens=8,
            question=sample.qa.question,
            answer=sample.response.answer,
        )
        match = re.search(r"[1-5]", raw)
        if not match:
            return 0.0, False
        return (int(match.group(0)) - 1) / 4, True
