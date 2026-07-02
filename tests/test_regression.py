import pytest

from ragcheck.metrics.base import MetricResult
from ragcheck.report.models import EvalReport
from ragcheck.report.regression import compare_reports, markdown_diff, parse_fail_if


def make_report(scores: dict[str, float], name="run") -> EvalReport:
    return EvalReport(
        run_name=name,
        dataset="d",
        adapter="a",
        n_samples=2,
        metrics=[
            MetricResult(metric_name=m, score=s, per_sample_scores=[s, s])
            for m, s in scores.items()
        ],
    )


def test_parse_fail_if():
    assert parse_fail_if(["faithfulness<-0.05", "hit_rate@5<-0.1"]) == {
        "faithfulness": -0.05,
        "hit_rate@5": -0.1,
    }
    with pytest.raises(ValueError, match="Invalid --fail-if"):
        parse_fail_if(["faithfulness>0.5"])


def test_compare_detects_breach_and_improvement():
    old = make_report({"faithfulness": 0.90, "hit_rate@5": 0.70, "mrr": 0.5})
    new = make_report({"faithfulness": 0.80, "hit_rate@5": 0.75, "mrr": 0.5})
    diffs = compare_reports(old, new, {"faithfulness": -0.05, "hit_rate@5": -0.05})
    by_name = {d.metric_name: d for d in diffs}
    assert by_name["faithfulness"].breached  # dropped 0.10 < -0.05
    assert by_name["faithfulness"].delta == pytest.approx(-0.10)
    assert not by_name["hit_rate@5"].breached  # improved
    assert not by_name["mrr"].breached  # no threshold set


def test_compare_small_drop_within_threshold_passes():
    old = make_report({"faithfulness": 0.90})
    new = make_report({"faithfulness": 0.88})
    diffs = compare_reports(old, new, {"faithfulness": -0.05})
    assert not diffs[0].breached


def test_compare_rejects_unknown_fail_if_metric():
    old = make_report({"faithfulness": 0.9})
    new = make_report({"faithfulness": 0.9})
    with pytest.raises(ValueError, match="absent from both reports"):
        compare_reports(old, new, {"nope": -0.1})


def test_markdown_diff_contains_status():
    old = make_report({"faithfulness": 0.90})
    new = make_report({"faithfulness": 0.80})
    diffs = compare_reports(old, new, {"faithfulness": -0.05})
    md = markdown_diff(diffs, "old.json", "new.json")
    assert "| faithfulness | 0.900 | 0.800 | -0.100 | FAIL" in md


def test_cli_compare_exit_code(tmp_path):
    from typer.testing import CliRunner

    from ragcheck.cli import app

    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(make_report({"faithfulness": 0.9}).model_dump_json())
    new_path.write_text(make_report({"faithfulness": 0.7}).model_dump_json())

    runner = CliRunner()
    failing = runner.invoke(
        app, ["compare", str(old_path), str(new_path), "--fail-if", "faithfulness<-0.05"]
    )
    assert failing.exit_code == 1
    passing = runner.invoke(app, ["compare", str(old_path), str(new_path)])
    assert passing.exit_code == 0
