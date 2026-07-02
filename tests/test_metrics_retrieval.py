import math

from ragcheck.metrics.retrieval.hit_rate import HitRate
from tests.conftest import make_sample


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
