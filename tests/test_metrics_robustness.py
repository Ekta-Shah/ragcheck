import math

from ragcheck.adapters.base import RAGResponse, RetrievedChunk
from ragcheck.datasets.models import EvalSample, QAPair
from ragcheck.judge.judge import Judge
from ragcheck.metrics.robustness.paraphrase_consistency import ParaphraseConsistency
from ragcheck.metrics.robustness.refusal_calibration import RefusalCalibration
from tests.conftest import MockLLMClient

REFUSAL_TEXT = "The provided documents do not contain this information."


def sample(question, answer, *, answerable=True, refused=False, group=None):
    return EvalSample(
        qa=QAPair(question=question, answerable=answerable, paraphrase_group=group),
        response=RAGResponse(
            answer=answer,
            retrieved_chunks=[RetrievedChunk(content="ctx", source_id="d1")],
            refused=refused,
        ),
    )


def refusal_responder(prompt: str) -> str:
    answer = prompt.split("Response:")[1].split("Respond with")[0]
    return "REFUSAL" if "do not contain" in answer else "ANSWER"


def make_refusal_metric():
    return RefusalCalibration(Judge(MockLLMClient(refusal_responder)))


# --- refusal_calibration ---


def test_refusal_calibration_four_quadrants():
    samples = [
        sample("q1", "A real answer.", answerable=True),  # correct
        sample("q2", REFUSAL_TEXT, answerable=True),  # over-refusal
        sample("q3", REFUSAL_TEXT, answerable=False),  # correct refusal
        sample("q4", "A hallucinated answer.", answerable=False),  # false answer
    ]
    result = make_refusal_metric().compute(samples)
    assert result.per_sample_scores == [1.0, 0.0, 1.0, 0.0]
    assert result.score == 0.5
    assert result.details["false_answer_rate"] == 0.5
    assert result.details["over_refusal_rate"] == 0.5
    assert result.details["false_answers"] == ["q4"]
    assert result.details["over_refusals"] == ["q2"]


def test_refusal_flag_short_circuits_judge():
    llm = MockLLMClient(refusal_responder)
    metric = RefusalCalibration(Judge(llm))
    result = metric.compute([sample("q", "anything", answerable=False, refused=True)])
    assert result.score == 1.0
    assert llm.calls == []  # explicit flag needs no judge call


def test_refusal_calibration_separates_hallucinator_from_calibrated():
    """Integration: a always-answer pipeline vs one that refuses unanswerables."""
    questions = [
        QAPair(question=f"answerable {i}") for i in range(3)
    ] + [QAPair(question=f"unanswerable {i}", answerable=False) for i in range(3)]

    def run_pipeline(refuses_unanswerable: bool):
        samples = []
        for qa in questions:
            if not qa.answerable and refuses_unanswerable:
                answer = REFUSAL_TEXT
            else:
                answer = f"Confident answer to: {qa.question}"
            samples.append(
                EvalSample(
                    qa=qa,
                    response=RAGResponse(answer=answer, retrieved_chunks=[]),
                )
            )
        return make_refusal_metric().compute(samples)

    hallucinator = run_pipeline(refuses_unanswerable=False)
    calibrated = run_pipeline(refuses_unanswerable=True)
    assert hallucinator.details["false_answer_rate"] == 1.0
    assert calibrated.details["false_answer_rate"] == 0.0
    assert calibrated.score == 1.0
    assert hallucinator.score == 0.5
    assert calibrated.score > hallucinator.score


# --- paraphrase_consistency ---


def equivalence_responder(prompt: str) -> str:
    answer_a = prompt.split("Answer A:")[1].split("Answer B:")[0].strip()
    answer_b = prompt.split("Answer B:")[1].split("Respond with")[0].strip()
    return "EQUIVALENT" if answer_a.split()[0] == answer_b.split()[0] else "DIFFERENT"


def make_consistency_metric():
    return ParaphraseConsistency(Judge(MockLLMClient(equivalence_responder)))


def test_paraphrase_consistency_all_agree():
    samples = [
        sample(f"q{i}", "blue is the answer", group="pg_0") for i in range(3)
    ]
    result = make_consistency_metric().compute(samples)
    assert result.score == 1.0
    assert result.per_sample_scores == [1.0, 1.0, 1.0]


def test_paraphrase_consistency_one_outlier():
    samples = [
        sample("q0", "blue is the answer", group="pg_0"),
        sample("q1", "blue is right", group="pg_0"),
        sample("q2", "red actually", group="pg_0"),
    ]
    result = make_consistency_metric().compute(samples)
    assert result.per_sample_scores == [1 / 3, 1 / 3, 1 / 3]
    assert len(result.details["inconsistent_pairs"]) == 2


def test_paraphrase_consistency_groups_and_ungrouped():
    samples = [
        sample("q0", "blue one", group="pg_0"),
        sample("q1", "blue two", group="pg_0"),
        sample("q2", "red one", group="pg_1"),
        sample("q3", "green one", group="pg_1"),
        sample("q4", "no group answer"),  # ungrouped -> NaN
        sample("q5", "singleton", group="pg_2"),  # singleton -> NaN
    ]
    result = make_consistency_metric().compute(samples)
    assert result.score == 0.5  # mean of group scores (1.0 + 0.0) / 2
    assert math.isnan(result.per_sample_scores[4])
    assert math.isnan(result.per_sample_scores[5])
    assert result.details["n_groups"] == 2
    assert result.details["ungrouped_samples"] == 2
