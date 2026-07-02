"""RAGCheck command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ragcheck.config import load_config
from ragcheck.report.cli_summary import print_summary
from ragcheck.runner import run_eval

app = typer.Typer(help="pytest for RAG systems.", no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """RAGCheck: evaluate RAG pipelines."""
    # Explicit callback keeps subcommand names (`ragcheck run ...`) even while
    # only one command exists; typer otherwise collapses single-command apps.


@app.command()
def run(config: Path = typer.Argument(..., help="Path to an eval config YAML.")) -> None:
    """Run an evaluation and print the scorecard."""
    eval_config = load_config(config)
    report, out_path = run_eval(eval_config)
    print_summary(report, console)
    console.print(f"[green]Report written to[/green] {out_path}")


@app.command("validate-judge")
def validate_judge_cmd(
    labels: Path = typer.Argument(..., help="JSONL of {question, answer, context, human_label}."),
    metric: str = typer.Option("faithfulness", help="LLM-judged metric to validate."),
    provider: str = typer.Option("anthropic", help="Judge provider: anthropic or groq."),
    model: str | None = typer.Option(None, help="Judge model (provider default if omitted)."),
    threshold: float = typer.Option(
        0.5, help="Per-sample metric score at or above which the judge label is 'pass'."
    ),
    concurrency: int = typer.Option(4, help="Parallel judge calls."),
    cache_path: Path = typer.Option(Path(".ragcheck_cache.sqlite"), help="Judgment cache."),
    output: Path = typer.Option(
        Path("ragcheck_output"), help="Directory for the validation report JSON."
    ),
) -> None:
    """Measure LLM-judge vs. human agreement (Cohen's kappa) on labeled samples."""
    from rich.table import Table

    from ragcheck.cache import JudgmentCache
    from ragcheck.judge.judge import Judge
    from ragcheck.judge.validation import load_labels, validate_judge
    from ragcheck.llm import build_client

    judge = Judge(build_client(provider, model), JudgmentCache(cache_path))
    report = validate_judge(
        load_labels(labels), metric, judge, threshold=threshold, concurrency=concurrency
    )

    table = Table(title=f"Judge validation - {report.metric_name}")
    table.add_column("Stat")
    table.add_column("Value", justify="right")
    table.add_row("judge model", f"{report.judge_model} ({report.prompt_version})")
    table.add_row("samples", str(report.n_samples))
    table.add_row("agreement", f"{report.agreement:.3f}")
    table.add_row("Cohen's kappa", f"{report.kappa:.3f}")
    c = report.confusion
    confusion = f"tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']}"
    table.add_row("confusion (judge vs human)", confusion)
    console.print(table)

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / f"judge_validation_{report.metric_name}.json"
    out_path.write_text(report.model_dump_json(indent=2))
    console.print(f"[green]Validation report written to[/green] {out_path}")


if __name__ == "__main__":
    app()
