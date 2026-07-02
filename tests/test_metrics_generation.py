import math

from ragcheck.judge.judge import Judge
from ragcheck.metrics.generation.citation_accuracy import CitationAccuracy, extract_citations
from ragcheck.metrics.generation.faithfulness import Faithfulness, _parse_claims
from ragcheck.metrics.generation.relevance import AnswerRelevance
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


def test_faithfulness_parallel_matches_serial():
    claims = {"X is Y.": ["X is Y"]}
    serial = Faithfulness(Judge(MockLLMClient(faithfulness_responder(claims, {"X is Y"}))))
    threaded = Faithfulness(
        Judge(MockLLMClient(faithfulness_responder(claims, {"X is Y"}))), concurrency=4
    )
    samples = [make_sample(answer="X is Y.", chunk_ids=["d1"]) for _ in range(8)]
    assert serial.compute(samples).per_sample_scores == threaded.compute(samples).per_sample_scores


# --- answer_relevance ---


def rating_responder(ratings_by_answer: dict[str, str]):
    def respond(prompt: str) -> str:
        for answer_key, rating in ratings_by_answer.items():
            if answer_key in prompt:
                return rating
        return "3"

    return respond


def make_relevance(ratings):
    return AnswerRelevance(Judge(MockLLMClient(rating_responder(ratings))))


def test_answer_relevance_normalizes_ratings():
    metric = make_relevance({"perfect answer": "5", "partial answer": "3", "off topic": "1"})
    samples = [
        make_sample(answer="perfect answer"),
        make_sample(answer="partial answer"),
        make_sample(answer="off topic"),
    ]
    result = metric.compute(samples)
    assert result.per_sample_scores == [1.0, 0.5, 0.0]
    assert result.score == 0.5


def test_answer_relevance_parses_rating_from_prose():
    metric = make_relevance({"the answer": "Rating: 4 (mostly direct)"})
    result = metric.compute([make_sample(answer="the answer")])
    assert result.per_sample_scores == [0.75]


def test_answer_relevance_unparseable_scores_zero():
    metric = make_relevance({"the answer": "excellent"})
    result = metric.compute([make_sample(answer="the answer")])
    assert result.per_sample_scores == [0.0]
    assert result.details["unparseable_verdicts"] == 1


# --- citation_accuracy ---


def test_extract_citations():
    answer = "Revenue grew 10% [doc_a]. Margins fell [doc_b]. No citation here."
    assert extract_citations(answer) == [
        ("Revenue grew 10% [doc_a].", "doc_a"),
        ("Margins fell [doc_b].", "doc_b"),
    ]


def citation_responder(supported_ids: set[str]):
    def respond(prompt: str) -> str:
        chunk = prompt.split("Cited document:")[1].split("Sentence:")[0]
        return "SUPPORTED" if any(cid in chunk for cid in supported_ids) else "UNSUPPORTED"

    return respond


def make_citation(supported_ids):
    return CitationAccuracy(Judge(MockLLMClient(citation_responder(supported_ids))))


def test_citation_accuracy_all_supported():
    metric = make_citation({"doc_a", "doc_b"})
    sample = make_sample(answer="Fact one [doc_a]. Fact two [doc_b].", chunk_ids=["doc_a", "doc_b"])
    assert metric.compute([sample]).score == 1.0


def test_citation_accuracy_unsupported_and_unretrieved():
    metric = make_citation({"doc_a"})
    sample = make_sample(
        answer="Fact one [doc_a]. Fact two [doc_b]. Fact three [doc_z].",
        chunk_ids=["doc_a", "doc_b"],  # doc_z never retrieved
    )
    result = metric.compute([sample])
    assert result.per_sample_scores == [1 / 3]
    reasons = {f["source_id"]: f["reason"] for f in result.details["failed_citations"]}
    assert reasons == {"doc_b": "chunk does not support", "doc_z": "source not retrieved"}


def test_citation_accuracy_skips_uncited_answers():
    metric = make_citation(set())
    result = metric.compute([make_sample(answer="No citations at all.", chunk_ids=["doc_a"])])
    assert math.isnan(result.per_sample_scores[0])
    assert result.details["skipped"] == 1
