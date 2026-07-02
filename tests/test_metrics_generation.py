from ragcheck.judge.judge import Judge
from ragcheck.metrics.generation.faithfulness import Faithfulness, _parse_claims
from tests.conftest import MockLLMClient, faithfulness_responder, make_sample


def build_metric(claims_by_answer, supported):
    llm = MockLLMClient(faithfulness_responder(claims_by_answer, supported))
    return Faithfulness(Judge(llm)), llm


def test_faithfulness_all_supported():
    metric, _ = build_metric({"X is Y.": ["X is Y"]}, supported={"X is Y"})
    result = metric.compute([make_sample(answer="X is Y.", chunk_ids=["d1"])])
    assert result.score == 1.0
    assert result.judge_model == "mock-judge-1"
    assert result.prompt_version == "v1"


def test_faithfulness_partial_support():
    metric, _ = build_metric(
        {"X is Y and Z.": ["X is Y", "X is Z"]}, supported={"X is Y"}
    )
    result = metric.compute([make_sample(answer="X is Y and Z.", chunk_ids=["d1"])])
    assert result.score == 0.5
    assert result.details["failed_claims"] == [{"question": "What is X?", "claim": "X is Z"}]


def test_faithfulness_no_claims_scores_one():
    metric, llm = build_metric({}, supported=set())
    result = metric.compute([make_sample(answer="I don't know.", chunk_ids=["d1"])])
    assert result.score == 1.0
    assert result.details["no_claim_samples"] == 1
    assert len(llm.calls) == 1  # decompose only, no verify calls


def test_parse_claims_tolerates_surrounding_prose():
    assert _parse_claims('Here you go:\n["a", "b"]\nDone.') == ["a", "b"]
    assert _parse_claims("no json here") == []
    assert _parse_claims("[not valid json") == []
