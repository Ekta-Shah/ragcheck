import math

from ragcheck.judge.judge import Judge
from ragcheck.metrics.retrieval.context_precision import ContextPrecision
from ragcheck.metrics.retrieval.context_recall import ContextRecall
from ragcheck.metrics.retrieval.hit_rate import HitRate
from ragcheck.metrics.retrieval.mrr import MRR
from tests.conftest import MockLLMClient, make_sample


def test_hit_rate_hit_and_miss():
    samples = [
        make_sample(chunk_ids=["a", "b"], relevant_ids=["b"]),  # hit
        make_sample(chunk_ids=["a", "b"], relevant_ids=["z"]),  # miss
    ]
    result = HitRate(k=5).compute(samples)
    assert result.score == 0.5
    assert result.per_sample_scores == [1.0, 0.0]
    assert result.metric_name == "hit_rate@5"
    assert result.judge_model is None


def test_hit_rate_respects_k_cutoff():
    samples = [make_sample(chunk_ids=["a", "b", "c"], relevant_ids=["c"])]
    assert HitRate(k=2).compute(samples).score == 0.0
    assert HitRate(k=3).compute(samples).score == 1.0


def test_hit_rate_skips_unlabeled_samples():
    samples = [
        make_sample(chunk_ids=["a"], relevant_ids=["a"]),
        make_sample(chunk_ids=["a"], relevant_ids=None),  # no labels
    ]
    result = HitRate(k=1).compute(samples)
    assert result.score == 1.0
    assert result.details["skipped"] == 1
    assert math.isnan(result.per_sample_scores[1])


# --- MRR ---


def test_mrr_known_ranks():
    samples = [
        make_sample(chunk_ids=["a", "b", "c"], relevant_ids=["a"]),  # rank 1 -> 1.0
        make_sample(chunk_ids=["a", "b", "c"], relevant_ids=["c"]),  # rank 3 -> 1/3
        make_sample(chunk_ids=["a", "b", "c"], relevant_ids=["z"]),  # absent -> 0.0
    ]
    result = MRR().compute(samples)
    assert result.per_sample_scores == [1.0, 1 / 3, 0.0]
    assert result.score == (1.0 + 1 / 3 + 0.0) / 3


def test_mrr_skips_unlabeled():
    result = MRR().compute([make_sample(chunk_ids=["a"], relevant_ids=None)])
    assert result.details["skipped"] == 1
    assert result.score == 0.0


# --- context_precision (LLM-judged) ---


def relevance_responder(relevant_markers: set[str]):
    """Chunk is RELEVANT when its content contains any marker string."""

    def respond(prompt: str) -> str:
        chunk = prompt.split("Chunk:")[1].split("Respond with")[0]
        return "RELEVANT" if any(m in chunk for m in relevant_markers) else "IRRELEVANT"

    return respond


def make_precision(relevant_markers, k=5):
    llm = MockLLMClient(relevance_responder(relevant_markers))
    return ContextPrecision(Judge(llm), k=k)


def test_context_precision_all_relevant():
    metric = make_precision({"content of"})  # every chunk matches
    result = metric.compute([make_sample(chunk_ids=["a", "b"])])
    assert result.score == 1.0
    assert result.judge_model == "mock-judge-1"
    assert result.prompt_version == "v1"


def test_context_precision_rank_aware():
    # chunks: a (relevant), b (irrelevant), c (relevant)
    # precision@1 = 1/1, precision@3 = 2/3 -> mean = (1 + 2/3)/2 = 5/6
    metric = make_precision({"content of a", "content of c"})
    result = metric.compute([make_sample(chunk_ids=["a", "b", "c"])])
    assert result.per_sample_scores[0] == (1.0 + 2 / 3) / 2
    assert result.details["relevance_flags"][0] == [True, False, True]


def test_context_precision_none_relevant_and_empty_retrieval():
    metric = make_precision(set())
    scored = metric.compute([make_sample(chunk_ids=["a", "b"])])
    assert scored.score == 0.0
    empty = metric.compute([make_sample(chunk_ids=[])])
    assert math.isnan(empty.per_sample_scores[0])
    assert empty.details["skipped"] == 1


# --- context_recall (LLM-judged) ---


def recall_responder(claims: list[str], supported: set[str]):
    import json

    def respond(prompt: str) -> str:
        if "atomic factual claims" in prompt:
            return json.dumps(claims)
        claim = prompt.split("Claim:")[1].split("Respond with")[0].strip()
        return "SUPPORTED" if claim in supported else "UNSUPPORTED"

    return respond


def make_recall(claims, supported):
    llm = MockLLMClient(recall_responder(claims, supported))
    return ContextRecall(Judge(llm))


def gt_sample(ground_truth="G is H and I.", chunk_ids=("d1",)):
    sample = make_sample(chunk_ids=list(chunk_ids))
    sample.qa.ground_truth_answer = ground_truth
    return sample


def test_context_recall_full_coverage():
    metric = make_recall(["G is H", "G is I"], supported={"G is H", "G is I"})
    result = metric.compute([gt_sample()])
    assert result.score == 1.0


def test_context_recall_partial_coverage_reports_missing():
    metric = make_recall(["G is H", "G is I"], supported={"G is H"})
    result = metric.compute([gt_sample()])
    assert result.score == 0.5
    assert result.details["missing_claims"][0]["claim"] == "G is I"


def test_context_recall_skips_without_ground_truth():
    metric = make_recall(["x"], supported=set())
    result = metric.compute([make_sample(chunk_ids=["d1"])])  # no ground truth
    assert math.isnan(result.per_sample_scores[0])
    assert result.details["skipped"] == 1
