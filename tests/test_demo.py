from typer.testing import CliRunner

from ragcheck.cli import app
from ragcheck.demo import DEMO_CASES, DemoJudgeClient, DemoPipeline, demo_dataset


def test_demo_pipeline_answers_and_refusal_flag():
    pipeline = DemoPipeline()
    grounded = pipeline.query(DEMO_CASES[0][0])
    assert "24 days" in grounded.answer
    assert grounded.retrieved_chunks and not grounded.refused
    refusal = pipeline.query("What is the parental leave policy?")
    assert refusal.refused


def test_demo_judge_verifies_by_overlap():
    judge = DemoJudgeClient()
    supported = judge.complete(
        "Retrieved documents:\n[doc_leave] Employees receive 24 days of paid leave per year.\n"
        "Claim:\nEmployees receive 24 days of paid leave.\n"
        "Respond with exactly one word: SUPPORTED or UNSUPPORTED."
    )
    unsupported = judge.complete(
        "Retrieved documents:\n[doc_leave] Employees receive 24 days of paid leave per year.\n"
        "Claim:\nEmployees get unlimited sick leave on request.\n"
        "Respond with exactly one word: SUPPORTED or UNSUPPORTED."
    )
    assert supported.text == "SUPPORTED"
    assert unsupported.text == "UNSUPPORTED"


def test_demo_cli_runs_offline_and_catches_planted_failures(tmp_path, monkeypatch):
    # Prove no key is needed
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = CliRunner().invoke(app, ["demo", "--output", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "unlimited sick leave" in result.output  # hallucination surfaced
    assert "false-answer rate: 0.50" in result.output
    assert (tmp_path / "demo.json").exists()
    assert (tmp_path / "demo.html").exists()


def test_demo_dataset_shape():
    dataset = demo_dataset()
    assert len(dataset.pairs) == 6
    assert sum(1 for p in dataset.pairs if not p.answerable) == 2
