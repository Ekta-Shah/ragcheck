from ragcheck.metrics.base import MetricResult
from ragcheck.report.html import render_html
from ragcheck.report.models import EvalReport, LatencySummary, ReportSample


def make_report(**overrides) -> EvalReport:
    defaults = dict(
        run_name="html-test",
        dataset="d.jsonl",
        adapter="mod:attr",
        n_samples=2,
        metrics=[
            MetricResult(
                metric_name="faithfulness",
                score=0.75,
                per_sample_scores=[0.5, 1.0],
                judge_model="mock-judge-1",
                prompt_version="v1",
            )
        ],
        latency=[LatencySummary(stage="retrieval", p50_ms=10, p95_ms=30)],
        samples=[
            ReportSample(
                question="What is <X>?",
                answer="X is Y & Z.",
                refused=False,
                answerable=True,
                difficulty="easy",
                contexts=["[d1] some context"],
                scores={"faithfulness": 0.5},
            ),
            ReportSample(
                question="q2",
                answer="a2",
                refused=False,
                answerable=True,
                difficulty="easy",
                contexts=[],
                scores={"faithfulness": 1.0},
            ),
        ],
    )
    defaults.update(overrides)
    return EvalReport(**defaults)


def test_render_html_scorecard_and_worst_samples():
    html = render_html(make_report())
    assert "<!DOCTYPE html>" in html
    assert "faithfulness" in html
    assert "0.750" in html
    assert "mock-judge-1" in html
    # worst-sample section shows only the failing sample, escaped
    assert "Worst samples - faithfulness" in html
    assert "What is &lt;X&gt;?" in html
    assert "X is Y &amp; Z." in html
    assert ">q2<" not in html  # score 1.0 is not a failure


def test_render_html_judge_validation_states():
    unvalidated = render_html(make_report())
    assert "not validated" in unvalidated

    validated = render_html(
        make_report(
            judge_validation={
                "metric_name": "faithfulness",
                "kappa": 0.83,
                "agreement": 0.9,
                "n_samples": 40,
                "threshold": 1.0,
                "judge_model": "mock-judge-1",
            }
        )
    )
    assert "validated at" in validated and "0.83" in validated


def test_render_html_is_self_contained():
    html = render_html(make_report())
    assert "http://" not in html and "https://" not in html
    assert "<script src" not in html and "<link" not in html
