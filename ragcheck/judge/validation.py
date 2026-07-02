"""Judge validation: measure LLM-judge vs. human agreement before trusting judged metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from ragcheck.adapters.base import RAGResponse, RetrievedChunk
from ragcheck.datasets.models import EvalSample, QAPair
from ragcheck.judge.judge import Judge


class LabeledSample(BaseModel):
    """One human-labeled example: the judge is validated against ``human_label``."""

    question: str
    answer: str
    context: str | list[str]
    human_label: int  # 1 = pass (e.g. faithful), 0 = fail

    def to_eval_sample(self) -> EvalSample:
        """Convert to the EvalSample shape metrics consume."""
        contexts = [self.context] if isinstance(self.context, str) else self.context
        chunks = [
            RetrievedChunk(content=text, source_id=f"ctx_{i}")
            for i, text in enumerate(contexts)
        ]
        return EvalSample(
            qa=QAPair(question=self.question),
            response=RAGResponse(answer=self.answer, retrieved_chunks=chunks),
        )


class ValidationReport(BaseModel):
    """Judge-vs-human agreement statistics for one metric."""

    metric_name: str
    judge_model: str
    prompt_version: str | None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    n_samples: int
    threshold: float
    agreement: float
    kappa: float
    confusion: dict[str, int]  # tp/fp/fn/tn, judge vs. human ("positive" = pass)


def load_labels(path: str | Path) -> list[LabeledSample]:
    """Load human-labeled samples from a JSONL file."""
    return [
        LabeledSample.model_validate(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's kappa for two binary raters from a confusion matrix."""
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    observed = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def validate_judge(
    labels: list[LabeledSample],
    metric_name: str,
    judge: Judge,
    *,
    threshold: float = 0.5,
    concurrency: int = 1,
) -> ValidationReport:
    """Run the judged metric on labeled samples and compare against human labels.

    A sample's judge label is 1 when its per-sample metric score is
    >= ``threshold``. Returns agreement rate, Cohen's kappa, and the
    confusion matrix (judge vs. human).
    """
    from ragcheck.metrics import build_metric  # local import: avoids module cycle

    metric = build_metric(metric_name, {}, judge, concurrency=concurrency)
    if not metric.requires_llm:
        raise ValueError(f"Metric {metric_name!r} is deterministic; nothing to validate")

    samples = [label.to_eval_sample() for label in labels]
    result = metric.compute(samples)

    tp = fp = fn = tn = 0
    for label, score in zip(labels, result.per_sample_scores, strict=True):
        if score != score:  # NaN: metric skipped the sample
            continue
        judged = 1 if score >= threshold else 0
        if judged and label.human_label:
            tp += 1
        elif judged and not label.human_label:
            fp += 1
        elif not judged and label.human_label:
            fn += 1
        else:
            tn += 1

    n = tp + fp + fn + tn
    return ValidationReport(
        metric_name=metric_name,
        judge_model=judge.model,
        prompt_version=result.prompt_version,
        n_samples=n,
        threshold=threshold,
        agreement=(tp + tn) / n if n else 0.0,
        kappa=cohens_kappa(tp, fp, fn, tn),
        confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    )
