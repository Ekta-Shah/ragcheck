import pytest

from ragcheck.datasets.loaders import load_dataset


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"question": "q1", "relevant_source_ids": ["d1"], "answerable": true}\n'
        '{"question": "q2", "answerable": false, "difficulty": "hard"}\n'
    )
    dataset = load_dataset(path)
    assert dataset.name == "data"
    assert len(dataset.pairs) == 2
    assert dataset.pairs[0].relevant_source_ids == ["d1"]
    assert dataset.pairs[1].answerable is False
    assert dataset.pairs[1].difficulty == "hard"


def test_load_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "question,ground_truth_answer,relevant_source_ids\n"
        'q1,answer one,d1;d2\n'
        "q2,,\n"
    )
    dataset = load_dataset(path)
    assert dataset.pairs[0].relevant_source_ids == ["d1", "d2"]
    assert dataset.pairs[1].ground_truth_answer is None
    assert dataset.pairs[1].relevant_source_ids == []


def test_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_dataset(path)
