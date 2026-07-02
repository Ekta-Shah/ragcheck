"""Regression comparison between two eval reports, CI-friendly."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from ragcheck.report.models import EvalReport

FAIL_IF_RE = re.compile(r"^([\w@.]+)<(-?\d+(?:\.\d+)?)$")


class MetricDiff(BaseModel):
    """Score movement of one metric between two runs."""

    metric_name: str
    old: float
    new: float
    delta: float
    threshold: float | None = None  # fail when delta < threshold
    breached: bool = False


def parse_fail_if(specs: list[str]) -> dict[str, float]:
    """Parse ``metric<-0.05`` threshold specs into {metric: min_allowed_delta}."""
    thresholds: dict[str, float] = {}
    for spec in specs:
        match = FAIL_IF_RE.match(spec.strip())
        if not match:
            raise ValueError(
                f"Invalid --fail-if spec {spec!r}; expected e.g. 'faithfulness<-0.05'"
            )
        thresholds[match.group(1)] = float(match.group(2))
    return thresholds


def load_report(path: str | Path) -> EvalReport:
    """Load a report JSON produced by ``ragcheck run``."""
    return EvalReport.model_validate(json.loads(Path(path).read_text()))


def compare_reports(
    old: EvalReport, new: EvalReport, fail_if: dict[str, float] | None = None
) -> list[MetricDiff]:
    """Diff metrics present in both reports; flag threshold breaches."""
    fail_if = fail_if or {}
    old_scores = {m.metric_name: m.score for m in old.metrics}
    diffs: list[MetricDiff] = []
    for metric in new.metrics:
        if metric.metric_name not in old_scores:
            continue
        old_score = old_scores[metric.metric_name]
        delta = metric.score - old_score
        threshold = fail_if.get(metric.metric_name)
        diffs.append(
            MetricDiff(
                metric_name=metric.metric_name,
                old=old_score,
                new=metric.score,
                delta=delta,
                threshold=threshold,
                breached=threshold is not None and delta < threshold,
            )
        )
    unknown = set(fail_if) - {d.metric_name for d in diffs}
    if unknown:
        raise ValueError(f"--fail-if names metrics absent from both reports: {sorted(unknown)}")
    return diffs


def markdown_diff(diffs: list[MetricDiff], old_name: str, new_name: str) -> str:
    """Markdown summary suitable for PR comments."""
    lines = [
        f"### RAGCheck regression: `{old_name}` -> `{new_name}`",
        "",
        "| Metric | Old | New | Delta | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for d in diffs:
        if d.breached:
            status = f"FAIL (delta < {d.threshold})"
        elif d.delta < 0:
            status = "regressed"
        elif d.delta > 0:
            status = "improved"
        else:
            status = "unchanged"
        lines.append(
            f"| {d.metric_name} | {d.old:.3f} | {d.new:.3f} | {d.delta:+.3f} | {status} |"
        )
    return "\n".join(lines)
