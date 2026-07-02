"""Run the architecture benchmark: same corpus, dataset, and generator; retrieval varies.

Rough early version: naive_rag vs hybrid_rag. The full 4-pipeline / 300-500 sample
version lands after Phases 2-3.

    python benchmarks/run_benchmark.py --quick   # subset of samples
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent / "pipelines"))

from common import DATA_DIR, DenseIndex, load_chunks  # noqa: E402
from hybrid_rag import HybridRAG  # noqa: E402
from naive_rag import NaiveRAG  # noqa: E402

from ragcheck.config import EvalConfig, MetricSpec  # noqa: E402
from ragcheck.datasets.loaders import load_dataset  # noqa: E402
from ragcheck.datasets.models import EvalDataset  # noqa: E402
from ragcheck.llm import default_client  # noqa: E402
from ragcheck.report.models import EvalReport  # noqa: E402
from ragcheck.runner import evaluate  # noqa: E402

BENCH_DIR = Path(__file__).parent
DATASET_PATH = BENCH_DIR / "benchmark_dataset.jsonl"


def run(quick: bool) -> None:
    console = Console()
    dataset = load_dataset(DATASET_PATH)
    if quick:
        dataset = EvalDataset(name=dataset.name, pairs=dataset.pairs[:8])
    console.print(f"Corpus: {DATA_DIR}  |  samples: {len(dataset.pairs)}")

    chunks = load_chunks()
    model = os.environ.get("RAGCHECK_BENCH_MODEL")  # None -> provider default
    llm = default_client(model)
    dense = DenseIndex(chunks)  # shared: identical embeddings for both pipelines
    pipelines = [
        NaiveRAG(chunks, llm, dense=dense),
        HybridRAG(chunks, llm, dense=dense),
    ]

    reports: list[EvalReport] = []
    for pipeline in pipelines:
        console.print(f"\n[bold]Evaluating {pipeline.name}...[/bold]")
        config = EvalConfig(
            dataset=DATASET_PATH,
            adapter=pipeline.name,
            metrics=[MetricSpec(name="hit_rate", params={"k": 5}), MetricSpec(name="faithfulness")],
            judge_provider="groq",
            judge_model=model,
            output_dir=BENCH_DIR / "results",
            cache_path=BENCH_DIR / "results" / "judge_cache.sqlite",
            run_name=pipeline.name,
        )
        report, path = evaluate(pipeline, dataset, config, adapter_name=pipeline.name)
        reports.append(report)
        console.print(f"  report: {path}")

    table = Table(title=f"Architecture comparison ({len(dataset.pairs)} samples)")
    table.add_column("Pipeline", style="bold")
    for metric in reports[0].metrics:
        table.add_column(metric.metric_name, justify="right")
    table.add_column("tokens/query", justify="right")
    table.add_column("retrieval p50 (ms)", justify="right")
    for report in reports:
        tokens = sum(report.pipeline_token_usage.values()) / max(report.n_samples, 1)
        retrieval_p50 = next(
            (entry.p50_ms for entry in report.latency if entry.stage == "retrieval"), 0.0
        )
        table.add_row(
            report.adapter,
            *(f"{m.score:.3f}" for m in report.metrics),
            f"{tokens:.0f}",
            f"{retrieval_p50:.0f}",
        )
    console.print()
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run on a small subset")
    args = parser.parse_args()
    run(quick=args.quick)
