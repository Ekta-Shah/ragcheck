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


if __name__ == "__main__":
    app()
