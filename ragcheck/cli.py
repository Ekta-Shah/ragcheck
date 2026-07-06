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
def run(
    config: Path = typer.Argument(..., help="Path to an eval config YAML."),
    yes: bool = typer.Option(
        False, "--yes", help="Proceed without confirmation on large LLM-judged runs."
    ),
) -> None:
    """Run an evaluation and print the scorecard."""
    eval_config = load_config(config)
    if yes:
        eval_config.assume_yes = True
    report, out_path = run_eval(eval_config)
    print_summary(report, console)
    console.print(f"[green]Report written to[/green] {out_path}")
    if eval_config.html:
        console.print(f"[green]HTML report:[/green] {out_path.with_suffix('.html')}")


@app.command()
def compare(
    old: Path = typer.Argument(..., help="Baseline report JSON."),
    new: Path = typer.Argument(..., help="New report JSON."),
    fail_if: list[str] = typer.Option(
        [],
        "--fail-if",
        help="Exit 1 when a metric's delta drops below a threshold, e.g. 'faithfulness<-0.05'. "
        "Repeatable.",
    ),
) -> None:
    """Diff two eval reports; non-zero exit on threshold breach (CI-friendly)."""
    from ragcheck.report.regression import (
        compare_reports,
        load_report,
        markdown_diff,
        parse_fail_if,
    )

    diffs = compare_reports(load_report(old), load_report(new), parse_fail_if(fail_if))
    console.print(markdown_diff(diffs, old.name, new.name))
    breached = [d for d in diffs if d.breached]
    if breached:
        names = ", ".join(d.metric_name for d in breached)
        console.print(f"\n[red]Regression threshold breached:[/red] {names}")
        raise typer.Exit(code=1)


@app.command()
def demo(
    output: Path = typer.Option(Path("ragcheck_output"), help="Where to write the reports."),
) -> None:
    """Run the zero-key demo: canned pipeline, offline judge, full reports in ~5 seconds."""
    from ragcheck.config import EvalConfig, MetricSpec
    from ragcheck.demo import DemoPipeline, demo_dataset
    from ragcheck.runner import evaluate

    console.print(
        "[bold]RAGCheck demo[/bold] - no API key needed.\n"
        "[dim]Canned pipeline answers (two deliberately wrong) + a deterministic offline "
        "judge, so you can see the workflow. Real runs use Claude/Groq judges.[/dim]\n"
    )
    config = EvalConfig(
        dataset=Path("demo"),
        adapter="ragcheck.demo:DemoPipeline",
        metrics=[
            MetricSpec(name="hit_rate", params={"k": 3}),
            MetricSpec(name="mrr"),
            MetricSpec(name="faithfulness"),
            MetricSpec(name="answer_relevance"),
            MetricSpec(name="refusal_calibration"),
        ],
        judge_provider="offline-demo",
        output_dir=output,
        cache_path=output / "demo_cache.sqlite",
        run_name="demo",
    )
    report, out_path = evaluate(
        DemoPipeline(), demo_dataset(), config, adapter_name="demo_pipeline"
    )
    print_summary(report, console)
    faith = next(m for m in report.metrics if m.metric_name == "faithfulness")
    refusal = next(m for m in report.metrics if m.metric_name == "refusal_calibration")
    console.print(
        f"\n[bold]What just happened:[/bold] faithfulness caught the planted hallucination "
        f"({[c['claim'] for c in faith.details['failed_claims']][:1]}), and refusal "
        f"calibration flagged the invented answer "
        f"(false-answer rate: {refusal.details['false_answer_rate']:.2f}).\n"
        f"[green]Open the HTML report to see the failing samples with their contexts:[/green] "
        f"{out_path.with_suffix('.html')}\n"
        "[dim]Next: wrap your own pipeline (docs/quickstart.md) and set "
        "ANTHROPIC_API_KEY or GROQ_API_KEY for real judging.[/dim]"
    )


@app.command("generate-dataset")
def generate_dataset_cmd(
    corpus_dir: Path = typer.Argument(..., help="Directory of .txt/.md corpus files."),
    n: int = typer.Option(200, help="Base QA pairs to generate (before paraphrase expansion)."),
    unanswerable_frac: float = typer.Option(0.15, help="Fraction of n that is unanswerable."),
    paraphrase_groups: int = typer.Option(
        20, help="Answerable questions to expand with 4 paraphrases each."
    ),
    out: Path = typer.Option(Path("dataset.jsonl"), help="Output JSONL path."),
    provider: str = typer.Option("anthropic", help="Generator provider: anthropic or groq."),
    model: str | None = typer.Option(None, help="Generator model (provider default if omitted)."),
    seed: int = typer.Option(13, help="Sampling seed (keep fixed for resumability)."),
    cache_path: Path = typer.Option(
        Path(".ragcheck_cache.sqlite"), help="Generation cache (makes re-runs resumable)."
    ),
) -> None:
    """Generate an eval dataset (difficulty tiers, unanswerables, paraphrases) from a corpus."""
    from collections import Counter

    from ragcheck.cache import JudgmentCache
    from ragcheck.datasets.synthetic import SyntheticGenerator
    from ragcheck.judge.judge import Judge
    from ragcheck.llm import build_client

    cache = JudgmentCache(cache_path)
    judge = Judge(build_client(provider, model), cache)
    generator = SyntheticGenerator(judge, seed=seed)
    dataset = generator.generate(
        corpus_dir, n=n, unanswerable_frac=unanswerable_frac, paraphrase_groups=paraphrase_groups
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(p.model_dump_json(exclude_none=True) for p in dataset.pairs) + "\n")

    tiers = Counter(p.difficulty for p in dataset.pairs if p.answerable)
    n_groups = len({p.paraphrase_group for p in dataset.pairs if p.paraphrase_group})
    console.print(
        f"[green]Wrote {len(dataset.pairs)} pairs to[/green] {out}\n"
        f"tiers: {dict(tiers)}  |  unanswerable: "
        f"{sum(1 for p in dataset.pairs if not p.answerable)}  |  "
        f"paraphrase groups: {n_groups}  |  "
        f"cache: {cache.hits} hits / {cache.misses} misses"
    )


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
