"""Rich terminal rendering of an EvalReport."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ragcheck.report.models import EvalReport


def print_summary(report: EvalReport, console: Console | None = None) -> None:
    """Print the metric scorecard and run stats to the terminal."""
    console = console or Console()

    table = Table(title=f"RAGCheck - {report.run_name}")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Judge", style="dim")
    for m in report.metrics:
        judge = f"{m.judge_model} ({m.prompt_version})" if m.judge_model else "-"
        table.add_row(m.metric_name, f"{m.score:.3f}", str(len(m.per_sample_scores)), judge)
    console.print(table)

    if report.latency:
        lat = Table(title="Latency (ms)")
        lat.add_column("Stage")
        lat.add_column("p50", justify="right")
        lat.add_column("p95", justify="right")
        for entry in report.latency:
            lat.add_row(entry.stage, f"{entry.p50_ms:.0f}", f"{entry.p95_ms:.0f}")
        console.print(lat)

    parts = []
    if report.pipeline_token_usage:
        parts.append(f"pipeline tokens: {report.pipeline_token_usage}")
    if report.judge_token_usage:
        parts.append(f"judge tokens: {report.judge_token_usage}")
    if report.cache_stats:
        parts.append(
            f"cache: {report.cache_stats.get('hits', 0)} hits / "
            f"{report.cache_stats.get('misses', 0)} misses"
        )
    if parts:
        console.print("[dim]" + "  |  ".join(parts) + "[/dim]")
