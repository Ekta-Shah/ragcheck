"""Refusal calibration: does the system say "I don't know" exactly when it should?"""

from __future__ import annotations

from ragcheck.datasets.models import EvalSample
from ragcheck.judge.judge import Judge, load_prompt
from ragcheck.metrics.base import Metric, MetricResult, parallel_map


class RefusalCalibration(Metric):
    """Correct-refusal accuracy over answerable and unanswerable questions.

    A sample is correct when: answerable and answered, or unanswerable and
    refused. Refusals are detected via the pipeline's ``refused`` flag or,
    failing that, an LLM-judged refusal classification of the answer text.

    Details report the two failure modes separately:
    ``false_answer_rate`` (hallucinated answers to unanswerable questions)
    and ``over_refusal_rate`` (refused answerable questions).
    """

    name = "refusal_calibration"
    requires_llm = True

    def __init__(self, judge: Judge, concurrency: int = 1) -> None:
        """Bind the judge and load the refusal-detection prompt."""
        self.judge = judge
        self.concurrency = concurrency
        self.prompt = load_prompt("refusal_detection")

    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score refusal behavior; overall score is accuracy across all samples."""
        refusals = parallel_map(self._is_refusal, samples, self.concurrency)

        per_sample: list[float] = []
        false_answers: list[str] = []
        over_refusals: list[str] = []
        n_answerable = n_unanswerable = 0
        for sample, refused in zip(samples, refusals, strict=True):
            if sample.qa.answerable:
                n_answerable += 1
                if refused:
                    over_refusals.append(sample.qa.question)
                per_sample.append(0.0 if refused else 1.0)
            else:
                n_unanswerable += 1
                if not refused:
                    false_answers.append(sample.qa.question)
                per_sample.append(1.0 if refused else 0.0)

        score = sum(per_sample) / len(per_sample) if per_sample else 0.0
        return MetricResult(
            metric_name=self.name,
            score=score,
            per_sample_scores=per_sample,
            details={
                "n_answerable": n_answerable,
                "n_unanswerable": n_unanswerable,
                "false_answer_rate": (
                    len(false_answers) / n_unanswerable if n_unanswerable else 0.0
                ),
                "over_refusal_rate": len(over_refusals) / n_answerable if n_answerable else 0.0,
                "false_answers": false_answers,
                "over_refusals": over_refusals,
            },
            judge_model=self.judge.model,
            prompt_version=self.prompt.version,
        )

    def _is_refusal(self, sample: EvalSample) -> bool:
        if sample.response.refused:
            return True
        raw = self.judge.ask(
            self.prompt,
            metric_name=self.name,
            key_parts=(sample.qa.question, sample.response.answer),
            max_tokens=8,
            question=sample.qa.question,
            answer=sample.response.answer,
        )
        return "REFUSAL" in raw.upper()
