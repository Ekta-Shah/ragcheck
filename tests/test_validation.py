import json

import pytest

from ragcheck.judge.judge import Judge
from ragcheck.judge.validation import (
    LabeledSample,
    cohens_kappa,
    load_labels,
    validate_judge,
)
from tests.conftest import MockLLMClient, faithfulness_responder


def test_cohens_kappa_known_values():
    assert cohens_kappa(tp=10, fp=0, fn=0, tn=10) == 1.0  # perfect agreement
    assert cohens_kappa(tp=5, fp=5, fn=5, tn=5) == 0.0  # chance-level
    # textbook 2x2: po=0.7, pe=0.5 -> kappa=0.4
    assert cohens_kappa(tp=20, fp=5, fn=10, tn=15) == pytest.approx(0.4)
    assert cohens_kappa(0, 0, 0, 0) == 0.0


def test_load_labels(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps(
            {"question": "q", "answer": "a", "context": "ctx", "human_label": 1}
        )
        + "\n"
        + json.dumps(
            {"question": "q2", "answer": "a2", "context": ["c1", "c2"], "human_label": 0}
        )
        + "\n"
    )
    labels = load_labels(path)
    assert len(labels) == 2
    sample = labels[1].to_eval_sample()
    assert [c.source_id for c in sample.response.retrieved_chunks] == ["ctx_0", "ctx_1"]


def make_labels():
    # judge will call: "grounded answer" -> claim supported (score 1.0),
    # "hallucinated answer" -> claim unsupported (score 0.0)
    return [
        LabeledSample(question="q1", answer="grounded answer", context="c", human_label=1),
        LabeledSample(question="q2", answer="hallucinated answer", context="c", human_label=0),
        LabeledSample(question="q3", answer="grounded answer", context="c", human_label=0),
    ]


def make_judge():
    responder = faithfulness_responder(
        {"grounded answer": ["good claim"], "hallucinated answer": ["bad claim"]},
        supported={"good claim"},
    )
    return Judge(MockLLMClient(responder))


def test_validate_judge_confusion_and_agreement():
    report = validate_judge(make_labels(), "faithfulness", make_judge())
    # judge: pass, fail, pass ; human: 1, 0, 0
    assert report.confusion == {"tp": 1, "fp": 1, "fn": 0, "tn": 1}
    assert report.agreement == pytest.approx(2 / 3)
    assert report.n_samples == 3
    assert report.judge_model == "mock-judge-1"
    assert report.prompt_version == "v1"


def test_validate_judge_rejects_deterministic_metric():
    with pytest.raises(ValueError, match="deterministic"):
        validate_judge(make_labels(), "hit_rate", make_judge())
