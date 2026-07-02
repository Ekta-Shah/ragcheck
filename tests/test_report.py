from rich.console import Console

from ragcheck.metrics.base import MetricResult
from ragcheck.report.cli_summary import print_summary
from ragcheck.report.models import EvalReport, LatencySummary, percentile


def test_percentile():
    assert percentile([], 50) == 0.0
    assert percentile([10.0], 95) == 10.0
    values = [float(v) for v in range(1, 102)]  # 1..101, odd count
    assert percentile(values, 50) == 51.0
    assert percentile(values, 100) == 101.0
    assert percentile(values, 0) == 1.0


def test_print_summary_renders():
    report = EvalReport(
        run_name="test-run",
        dataset="d.jsonl",
        adapter="mod:attr",
        n_samples=2,
        metrics=[
            MetricResult(
                metric_name="hit_rate@3", score=0.5, per_sample_scores=[1.0, 0.0]
            ),
            MetricResult(
                metric_name="faithfulness",
                score=0.9,
                per_sample_scores=[0.8, 1.0],
                judge_model="mock-judge-1",
                prompt_version="v1",
            ),
        ],
        latency=[LatencySummary(stage="retrieval", p50_ms=10, p95_ms=20)],
        cache_stats={"hits": 3, "misses": 1},
    )
    console = Console(record=True, width=100)
    print_summary(report, console)
    output = console.export_text()
    assert "hit_rate@3" in output
    assert "0.900" in output
    assert "mock-judge-1" in output
    assert "3 hits" in output
