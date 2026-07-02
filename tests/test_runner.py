import json

import pytest

from ragcheck.adapters.base import RAGAdapter, RAGResponse, RetrievedChunk
from ragcheck.config import EvalConfig, MetricSpec
from ragcheck.runner import load_adapter, run_eval
from tests.conftest import MockLLMClient, faithfulness_responder

DATASET = (
    '{"question": "What color is the sky?", "relevant_source_ids": ["doc_sky"]}\n'
    '{"question": "What color is grass?", "relevant_source_ids": ["doc_grass"]}\n'
)


class ToyAdapter(RAGAdapter):
    def query(self, question: str) -> RAGResponse:
        doc = "doc_sky" if "sky" in question else "doc_other"
        return RAGResponse(
            answer="The sky is blue.",
            retrieved_chunks=[RetrievedChunk(content="The sky is blue.", source_id=doc)],
            latencies_ms={"retrieval": 3.0, "generation": 40.0},
            token_usage={"input_tokens": 50, "output_tokens": 10},
        )


def build_toy_adapter() -> ToyAdapter:
    return ToyAdapter()


def make_config(tmp_path, metrics):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(DATASET)
    return EvalConfig(
        dataset=dataset,
        adapter="tests.test_runner:build_toy_adapter",
        metrics=metrics,
        output_dir=tmp_path / "out",
        cache_path=tmp_path / "cache.sqlite",
        run_name="test-run",
    )


def test_load_adapter_factory_and_errors():
    adapter = load_adapter("tests.test_runner:build_toy_adapter")
    assert isinstance(adapter, ToyAdapter)
    with pytest.raises(ValueError, match="module.path:attribute"):
        load_adapter("tests.test_runner")
    with pytest.raises(TypeError):
        load_adapter("tests.test_runner:DATASET")


def test_run_eval_deterministic_only(tmp_path):
    config = make_config(tmp_path, [MetricSpec(name="hit_rate", params={"k": 1})])
    report, out_path = run_eval(config)
    assert report.n_samples == 2
    assert report.metrics[0].score == 0.5  # sky hit, grass miss
    assert report.pipeline_token_usage == {"input_tokens": 100, "output_tokens": 20}
    assert {entry.stage for entry in report.latency} == {"retrieval", "generation"}
    saved = json.loads(out_path.read_text())
    assert saved["run_name"] == "test-run"


def test_run_eval_with_judged_metric_uses_cache(tmp_path, monkeypatch):
    responder = faithfulness_responder(
        {"The sky is blue.": ["The sky is blue"]}, supported={"The sky is blue"}
    )
    monkeypatch.setattr(
        "ragcheck.runner.build_client", lambda provider, model: MockLLMClient(responder)
    )
    config = make_config(tmp_path, [MetricSpec(name="faithfulness")])

    report, _ = run_eval(config)
    assert report.metrics[0].score == 1.0
    assert report.cache_stats["misses"] > 0
    assert report.cache_stats["hits"] == 0

    # second run: every judgment served from cache
    report2, _ = run_eval(config)
    assert report2.cache_stats["misses"] == 0
    assert report2.cache_stats["hits"] == report.cache_stats["misses"]


def test_unknown_metric_raises(tmp_path):
    config = make_config(tmp_path, [MetricSpec(name="nope")])
    with pytest.raises(ValueError, match="Unknown metric"):
        run_eval(config)
