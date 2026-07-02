import json

import pytest

from ragcheck.cache import JudgmentCache
from ragcheck.datasets.synthetic import SyntheticGenerator, chunk_corpus
from ragcheck.judge.judge import Judge
from tests.conftest import MockLLMClient


def synthetic_responder():
    """Route generation prompts by their instruction markers; unique QA per prompt."""
    counter = {"n": 0}

    def respond(prompt: str) -> str:
        counter["n"] += 1
        i = counter["n"]
        if "Rewrite the following question in 4 different ways" in prompt:
            return json.dumps([f"para {i}-{j}" for j in range(4)])
        if "NOT contained in the excerpt" in prompt:
            return json.dumps({"question": f"unanswerable q{i}"})
        return json.dumps({"question": f"q{i}", "answer": f"a{i}"})

    return respond


@pytest.fixture
def corpus_dir(tmp_path):
    for doc in ("doc_a", "doc_b"):
        # ~800 chars -> several chunks at chunk_chars=200/overlap=50
        (tmp_path / f"{doc}.txt").write_text(
            " ".join(f"{doc} sentence number {i} with some substantive words." for i in range(20))
        )
    return tmp_path


def make_generator(tmp_path, responder=None):
    llm = MockLLMClient(responder or synthetic_responder())
    cache = JudgmentCache(tmp_path / "gen_cache.sqlite")
    return SyntheticGenerator(Judge(llm, cache), seed=13), llm


def test_chunk_corpus_ids_and_provenance(corpus_dir):
    chunks = chunk_corpus(corpus_dir, chunk_chars=200, overlap=50)
    assert len(chunks) > 4
    assert chunks[0].id == "doc_a_0000"
    assert all(c.doc in ("doc_a", "doc_b") for c in chunks)


def test_chunk_corpus_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        chunk_corpus(tmp_path)


def test_generate_tiers_unanswerables_and_paraphrases(corpus_dir, tmp_path):
    generator, _ = make_generator(tmp_path)
    dataset = generator.generate(
        corpus_dir, n=10, unanswerable_frac=0.2, paraphrase_groups=2,
        chunk_chars=200, overlap=50,
    )
    base = [p for p in dataset.pairs if p.answerable]
    unanswerable = [p for p in dataset.pairs if not p.answerable]
    tiers = {d: sum(1 for p in base if p.difficulty == d) for d in ("easy", "medium", "hard")}

    assert len(unanswerable) == 2
    assert all(not p.relevant_source_ids for p in unanswerable)
    # 8 answerable base pairs split 4/2/2, plus 2 groups x 4 paraphrases
    assert tiers["easy"] == 4 + 8  # paraphrases inherit their base's difficulty (easy)
    assert tiers["medium"] == 2
    assert tiers["hard"] == 2

    # provenance
    easy_base = [p for p in base if p.difficulty == "easy" and not p.paraphrase_group][:1]
    mediums = [p for p in base if p.difficulty == "medium"]
    hards = [p for p in base if p.difficulty == "hard"]
    assert all(len(p.relevant_source_ids) == 1 for p in easy_base)
    assert all(len(p.relevant_source_ids) >= 2 for p in mediums)
    docs_of = lambda p: {i.rsplit("_", 1)[0] for i in p.relevant_source_ids}  # noqa: E731
    assert all(len(docs_of(p)) == 1 for p in mediums)  # within one doc
    assert all(len(docs_of(p)) == 2 for p in hards)  # across two docs

    # paraphrase groups: base + 4 paraphrases share the id and ground truth
    groups: dict[str, list] = {}
    for p in dataset.pairs:
        if p.paraphrase_group:
            groups.setdefault(p.paraphrase_group, []).append(p)
    assert len(groups) == 2
    for members in groups.values():
        assert len(members) == 5
        assert len({m.ground_truth_answer for m in members}) == 1
        assert len({m.question for m in members}) == 5


def test_generate_is_resumable_via_cache(corpus_dir, tmp_path):
    generator, llm = make_generator(tmp_path)
    kwargs = dict(n=6, unanswerable_frac=0.0, paraphrase_groups=1, chunk_chars=200, overlap=50)
    first = generator.generate(corpus_dir, **kwargs)
    calls_after_first = len(llm.calls)

    second = generator.generate(corpus_dir, **kwargs)
    assert len(llm.calls) == calls_after_first  # every call served from cache
    assert [p.question for p in second.pairs] == [p.question for p in first.pairs]


def test_generate_tolerates_malformed_json(corpus_dir, tmp_path):
    generator, _ = make_generator(tmp_path, responder=lambda prompt: "not json at all")
    dataset = generator.generate(
        corpus_dir, n=6, unanswerable_frac=0.5, paraphrase_groups=1,
        chunk_chars=200, overlap=50,
    )
    assert dataset.pairs == []  # nothing parseable, but no crash
