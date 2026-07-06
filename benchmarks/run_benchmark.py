"""Run the architecture benchmark: 4 RAG pipelines, same corpus/generator, retrieval varies.

    python benchmarks/run_benchmark.py                 # full dataset, all pipelines
    python benchmarks/run_benchmark.py --quick         # stratified 20-sample subset
    python benchmarks/run_benchmark.py --n 4 --pipelines naive_rag,agentic_rag \
        --metrics hit_rate,mrr,faithfulness            # cheap smoke run

Outputs per-pipeline reports, a combined comparison (JSON + markdown), and the
cost-quality frontier chart under benchmarks/results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent / "pipelines"))

from agentic_rag import AgenticRAG  # noqa: E402
from common import DATA_DIR, DenseIndex, load_chunks  # noqa: E402
from hybrid_rag import HybridRAG  # noqa: E402
from naive_rag import NaiveRAG  # noqa: E402
from reranked_rag import RerankedRAG  # noqa: E402

from ragcheck.adapters.base import RAGAdapter, RAGResponse  # noqa: E402
from ragcheck.cache import JudgmentCache, make_key  # noqa: E402
from ragcheck.config import EvalConfig, MetricSpec  # noqa: E402
from ragcheck.datasets.loaders import load_dataset  # noqa: E402
from ragcheck.datasets.models import EvalDataset, QAPair  # noqa: E402
from ragcheck.llm import default_client  # noqa: E402
from ragcheck.report.models import EvalReport  # noqa: E402
from ragcheck.runner import evaluate  # noqa: E402

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
DEFAULT_DATASET = BENCH_DIR / "synthetic_dataset.jsonl"

PIPELINES = {
    "naive_rag": NaiveRAG,
    "hybrid_rag": HybridRAG,
    "reranked_rag": RerankedRAG,
    "agentic_rag": AgenticRAG,
}

FULL_METRICS = [
    MetricSpec(name="hit_rate", params={"k": 5}),
    MetricSpec(name="mrr"),
    MetricSpec(name="context_precision", params={"k": 5}),
    MetricSpec(name="context_recall"),
    MetricSpec(name="faithfulness"),
    MetricSpec(name="answer_relevance"),
    MetricSpec(name="citation_accuracy"),
    MetricSpec(name="refusal_calibration"),
    MetricSpec(name="paraphrase_consistency"),
]

# USD per 1M input/output tokens (Groq list prices; override if pricing changes).
PRICING = {
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


class CachedPipeline(RAGAdapter):
    """Cache pipeline responses so interrupted runs resume without re-generating.

    Keyed on (pipeline, generator model, question). Cached responses keep
    their original token_usage/latencies - they represent the pipeline's
    intrinsic cost, which is what the comparison reports. Stable answers
    across resumes also keep judge-cache hits intact.
    """

    def __init__(self, inner: RAGAdapter, cache: JudgmentCache, model_tag: str) -> None:
        self.inner = inner
        self.cache = cache
        self.model_tag = model_tag
        self.name = getattr(inner, "name", type(inner).__name__)

    def query(self, question: str) -> RAGResponse:
        key = make_key("pipeline_response", self.name, self.model_tag, question)
        cached = self.cache.get(key)
        if cached is not None:
            return RAGResponse.model_validate_json(cached)
        response = self.inner.query(question)
        self.cache.set(key, response.model_dump_json())
        return response


def quick_subset(pairs: list[QAPair], n: int) -> list[QAPair]:
    """Stratified subset: keep unanswerables and whole paraphrase groups, fill with base pairs."""
    unanswerable = [p for p in pairs if not p.answerable][:2]
    group_ids = list(dict.fromkeys(p.paraphrase_group for p in pairs if p.paraphrase_group))[:2]
    grouped = [p for p in pairs if p.paraphrase_group in group_ids]
    base = [p for p in pairs if p.answerable and not p.paraphrase_group]
    return (unanswerable + grouped + base)[:n]


def cost_per_query(report: EvalReport, model: str) -> float | None:
    """Pipeline (not judge) USD cost per query, if the model's price is known."""
    prices = PRICING.get(model)
    if prices is None or report.n_samples == 0:
        return None
    in_price, out_price = prices
    usage = report.pipeline_token_usage
    total = (
        usage.get("input_tokens", 0) * in_price + usage.get("output_tokens", 0) * out_price
    ) / 1e6
    return total / report.n_samples


def per_difficulty(report: EvalReport, dataset: EvalDataset, metric_name: str) -> dict[str, float]:
    """Mean per-sample score of one metric, split by question difficulty."""
    result = next((m for m in report.metrics if m.metric_name == metric_name), None)
    if result is None:
        return {}
    buckets: dict[str, list[float]] = {}
    for pair, score in zip(dataset.pairs, result.per_sample_scores, strict=True):
        if score == score:  # skip NaN
            buckets.setdefault(pair.difficulty, []).append(score)
    return {d: sum(v) / len(v) for d, v in sorted(buckets.items())}


def frontier_chart(rows: list[dict], path: Path) -> None:
    """Cost-quality frontier: x = cost/query (USD), y = faithfulness."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted = [r for r in rows if r.get("cost_per_query") is not None]
    if not plotted:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in plotted:
        ax.scatter(row["cost_per_query"], row["metrics"].get("faithfulness", 0.0), s=80)
        ax.annotate(
            row["pipeline"],
            (row["cost_per_query"], row["metrics"].get("faithfulness", 0.0)),
            xytext=(8, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Cost per query (USD)")
    ax.set_ylabel("Faithfulness")
    ax.set_title("Cost-quality frontier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def run(
    dataset_path: Path,
    pipeline_names: list[str],
    metrics: list[MetricSpec],
    judge_model: str | None,
    n: int | None,
    quick: bool,
) -> None:
    console = Console()
    dataset = load_dataset(dataset_path)
    if quick:
        dataset = EvalDataset(name=dataset.name, pairs=quick_subset(dataset.pairs, 20))
    elif n is not None:
        dataset = EvalDataset(name=dataset.name, pairs=quick_subset(dataset.pairs, n))
    console.print(
        f"Corpus: {DATA_DIR}  |  dataset: {dataset_path.name}  |  "
        f"samples: {len(dataset.pairs)}  |  pipelines: {', '.join(pipeline_names)}"
    )

    gen_model = os.environ.get("RAGCHECK_BENCH_MODEL")
    chunks = load_chunks()
    llm = default_client(gen_model)
    dense = DenseIndex(chunks)  # shared embeddings across all pipelines

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    response_cache = JudgmentCache(RESULTS_DIR / "response_cache.sqlite")
    rows: list[dict] = []
    for name in pipeline_names:
        console.print(f"\n[bold]Evaluating {name}...[/bold]")
        pipeline = CachedPipeline(
            PIPELINES[name](chunks, llm, dense=dense),
            response_cache,
            gen_model or getattr(llm, "model", "default"),
        )
        config = EvalConfig(
            dataset=dataset_path,
            adapter=name,
            metrics=metrics,
            judge_provider="groq",
            judge_model=judge_model or gen_model,
            concurrency=int(os.environ.get("RAGCHECK_CONCURRENCY", "2")),
            output_dir=RESULTS_DIR,
            cache_path=RESULTS_DIR / "judge_cache.sqlite",
            run_name=name,
        )
        report, path = evaluate(pipeline, dataset, config, adapter_name=name)
        console.print(
            f"  report: {path}  [dim](responses: {response_cache.hits} cached / "
            f"{response_cache.misses} fresh)[/dim]"
        )
        effective_model = getattr(llm, "model", gen_model or "unknown")
        rows.append(
            {
                "pipeline": name,
                "metrics": {m.metric_name: m.score for m in report.metrics},
                "tokens_per_query": (
                    sum(report.pipeline_token_usage.values()) / max(report.n_samples, 1)
                ),
                "cost_per_query": cost_per_query(report, effective_model),
                "retrieval_p50_ms": next(
                    (e.p50_ms for e in report.latency if e.stage == "retrieval"), 0.0
                ),
                "per_difficulty_faithfulness": per_difficulty(report, dataset, "faithfulness"),
                "per_difficulty_hit_rate": per_difficulty(
                    report, dataset, f"hit_rate@{metrics[0].params.get('k', 5)}"
                ),
            }
        )

    # combined outputs
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(rows, indent=2))
    frontier_chart(rows, RESULTS_DIR / "cost_quality_frontier.png")

    metric_names = list(rows[0]["metrics"]) if rows else []
    table = Table(title=f"Architecture comparison ({len(dataset.pairs)} samples)")
    table.add_column("Pipeline", style="bold")
    for name in metric_names:
        table.add_column(name.replace("_", "\n"), justify="right")
    table.add_column("tok/q", justify="right")
    table.add_column("$/q", justify="right")
    table.add_column("ret p50", justify="right")
    lines = ["| Pipeline | " + " | ".join(metric_names) + " | tok/q | $/q |", ""]
    lines[1] = "|" + "---|" * (len(metric_names) + 3)
    for row in rows:
        cost = f"{row['cost_per_query']:.5f}" if row["cost_per_query"] is not None else "-"
        scores = [f"{row['metrics'][m]:.3f}" for m in metric_names]
        table.add_row(
            row["pipeline"], *scores, f"{row['tokens_per_query']:.0f}", cost,
            f"{row['retrieval_p50_ms']:.0f}",
        )
        lines.append(f"| {row['pipeline']} | " + " | ".join(scores) + f" | {row['tokens_per_query']:.0f} | {cost} |")
    (RESULTS_DIR / "comparison.md").write_text("\n".join(lines) + "\n")
    console.print()
    console.print(table)

    diff_table = Table(title="Faithfulness by difficulty")
    diff_table.add_column("Pipeline", style="bold")
    for tier in ("easy", "medium", "hard"):
        diff_table.add_column(tier, justify="right")
    for row in rows:
        tiers = row["per_difficulty_faithfulness"]
        diff_table.add_row(
            row["pipeline"], *(f"{tiers[t]:.3f}" if t in tiers else "-" for t in ("easy", "medium", "hard"))
        )
    console.print(diff_table)
    console.print(f"[dim]combined report: {RESULTS_DIR / 'comparison.json'}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--quick", action="store_true", help="Stratified 20-sample subset")
    parser.add_argument("--n", type=int, default=None, help="Stratified subset of N samples")
    parser.add_argument(
        "--pipelines", default=",".join(PIPELINES), help="Comma-separated pipeline names"
    )
    parser.add_argument(
        "--metrics", default=None,
        help="Comma-separated metric names (default: full suite)",
    )
    parser.add_argument(
        "--judge-model", default=os.environ.get("RAGCHECK_JUDGE_MODEL"),
        help="Judge model (defaults to the generator model)",
    )
    args = parser.parse_args()

    names = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    unknown = [p for p in names if p not in PIPELINES]
    if unknown:
        parser.error(f"Unknown pipelines {unknown}; choose from {list(PIPELINES)}")
    if args.metrics:
        chosen = [
            MetricSpec(name=spec.name, params=spec.params)
            for spec in FULL_METRICS
            if spec.name in {m.strip() for m in args.metrics.split(",")}
        ]
    else:
        chosen = FULL_METRICS
    run(args.dataset, names, chosen, args.judge_model, args.n, args.quick)
