"""Eval orchestration: dataset -> adapter -> metrics -> report."""

from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ragcheck.adapters.base import RAGAdapter
from ragcheck.cache import JudgmentCache
from ragcheck.config import EvalConfig
from ragcheck.datasets.loaders import load_dataset
from ragcheck.datasets.models import EvalDataset, EvalSample
from ragcheck.judge.judge import Judge
from ragcheck.llm import LLMClient, build_client
from ragcheck.metrics import _LLM_JUDGED, build_metric
from ragcheck.report.models import EvalReport, LatencySummary, percentile


def load_adapter(spec: str) -> RAGAdapter:
    """Resolve ``"module.path:attribute"`` to a RAGAdapter.

    The attribute may be a RAGAdapter instance or a zero-argument factory
    returning one. The current working directory is importable, so example
    and project-local adapters resolve without installation.
    """
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"Adapter spec {spec!r} must look like 'module.path:attribute'")
    if "" not in sys.path and str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    if isinstance(obj, RAGAdapter):
        return obj
    if callable(obj):
        adapter = obj()
        if isinstance(adapter, RAGAdapter):
            return adapter
    raise TypeError(f"{spec!r} is not a RAGAdapter or a factory returning one")


def run_eval(config: EvalConfig) -> tuple[EvalReport, Path]:
    """Run the full eval described by ``config``; return the report and its path."""
    dataset = load_dataset(config.dataset)
    adapter = load_adapter(config.adapter)
    return evaluate(adapter, dataset, config, adapter_name=config.adapter)


def evaluate(
    adapter: RAGAdapter,
    dataset: EvalDataset,
    config: EvalConfig,
    *,
    adapter_name: str,
) -> tuple[EvalReport, Path]:
    """Evaluate an already-constructed adapter (programmatic entry point).

    ``config.dataset``/``config.adapter`` are ignored here; the caller supplies
    both objects directly. Used by benchmarks and library consumers.
    """
    responses = adapter.batch_query([qa.question for qa in dataset.pairs])
    samples = [
        EvalSample(qa=qa, response=r)
        for qa, r in zip(dataset.pairs, responses, strict=True)
    ]

    needs_judge = any(spec.name in _LLM_JUDGED for spec in config.metrics)
    cache: JudgmentCache | None = None
    judge: Judge | None = None
    llm: LLMClient | None = None
    if needs_judge:
        config.cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = JudgmentCache(config.cache_path)
        llm = build_client(config.judge_provider, config.judge_model)
        judge = Judge(llm, cache)

    results = [
        build_metric(spec.name, spec.params, judge).compute(samples) for spec in config.metrics
    ]

    pipeline_tokens: dict[str, int] = defaultdict(int)
    stage_latencies: dict[str, list[float]] = defaultdict(list)
    for response in responses:
        for key, count in response.token_usage.items():
            pipeline_tokens[key] += count
        for stage, ms in response.latencies_ms.items():
            stage_latencies[stage].append(ms)

    report = EvalReport(
        run_name=config.run_name or f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}",
        dataset=dataset.name,
        adapter=adapter_name,
        n_samples=len(samples),
        metrics=results,
        latency=[
            LatencySummary(stage=stage, p50_ms=percentile(v, 50), p95_ms=percentile(v, 95))
            for stage, v in sorted(stage_latencies.items())
        ],
        pipeline_token_usage=dict(pipeline_tokens),
        judge_token_usage=(
            {
                "input_tokens": getattr(llm, "total_input_tokens", 0),
                "output_tokens": getattr(llm, "total_output_tokens", 0),
            }
            if llm
            else {}
        ),
        cache_stats={"hits": cache.hits, "misses": cache.misses} if cache else {},
    )
    if cache is not None:
        cache.close()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / f"{report.run_name}.json"
    out_path.write_text(report.model_dump_json(indent=2))
    return report, out_path
