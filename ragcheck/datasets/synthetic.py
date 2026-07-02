"""Synthetic eval dataset generation from any text/markdown corpus.

Every LLM call goes through the shared Judge, so generation is cached and
resumable: re-running with the same corpus, seed, and parameters re-issues
no LLM calls and reproduces the same dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random

from ragcheck.datasets.models import EvalDataset, QAPair
from ragcheck.judge.judge import Judge

GENERATION_METRIC = "dataset_generation"  # cache namespace for all generation calls
DIFFICULTY_MIX = {"easy": 0.5, "medium": 0.3, "hard": 0.2}
PARAPHRASES_PER_GROUP = 4  # plus the base question = 5 per group
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class Chunk:
    """A corpus chunk with provenance."""

    id: str
    doc: str
    text: str


def chunk_corpus(
    corpus_dir: str | Path,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
    min_chars: int = 100,
) -> list[Chunk]:
    """Chunk every ``.txt``/``.md`` file under ``corpus_dir`` with fixed size/overlap."""
    corpus_dir = Path(corpus_dir)
    files = sorted([*corpus_dir.glob("*.txt"), *corpus_dir.glob("*.md")])
    if not files:
        raise FileNotFoundError(f"No .txt or .md files in {corpus_dir}")
    chunks: list[Chunk] = []
    step = chunk_chars - overlap
    for path in files:
        text = path.read_text()
        index = 0
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_chars].strip()
            if len(piece) >= min_chars:
                chunks.append(Chunk(id=f"{path.stem}_{index:04d}", doc=path.stem, text=piece))
                index += 1
    return chunks


def _parse_json_value(raw: str) -> object | None:
    """Extract the first JSON object or array embedded in judge output."""
    match = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class SyntheticGenerator:
    """Generate QA datasets with difficulty tiers, unanswerables, and paraphrase groups."""

    def __init__(self, judge: Judge, seed: int = 13) -> None:
        """Bind the (cache-backed) judge; ``seed`` fixes chunk sampling for resumability."""
        from ragcheck.judge.judge import load_prompt

        self.judge = judge
        self.seed = seed
        self.prompts = {
            name: load_prompt(f"dataset_{name}")
            for name in ("qa_single", "qa_multi", "qa_cross", "unanswerable", "paraphrase")
        }

    def generate(
        self,
        corpus_dir: str | Path,
        n: int = 200,
        unanswerable_frac: float = 0.15,
        paraphrase_groups: int = 20,
        chunk_chars: int = CHUNK_CHARS,
        overlap: int = CHUNK_OVERLAP,
    ) -> EvalDataset:
        """Generate ``n`` base QA pairs plus paraphrase expansions.

        Of the ``n`` pairs, ``unanswerable_frac`` are unanswerable; the rest
        split easy/medium/hard per ``DIFFICULTY_MIX``. The first
        ``paraphrase_groups`` answerable pairs are each expanded with 4
        paraphrases sharing a ``paraphrase_group`` id.
        """
        chunks = chunk_corpus(corpus_dir, chunk_chars, overlap)
        rng = Random(self.seed)

        n_unanswerable = round(n * unanswerable_frac)
        n_answerable = n - n_unanswerable
        n_easy = round(n_answerable * DIFFICULTY_MIX["easy"])
        n_medium = round(n_answerable * DIFFICULTY_MIX["medium"])
        n_hard = n_answerable - n_easy - n_medium

        pairs: list[QAPair] = []
        pairs += self._generate_easy(chunks, n_easy, rng)
        pairs += self._generate_medium(chunks, n_medium, rng)
        pairs += self._generate_hard(chunks, n_hard, rng)
        pairs += self._generate_unanswerable(chunks, n_unanswerable, rng)
        pairs += self._expand_paraphrases(pairs, paraphrase_groups)
        return EvalDataset(name=Path(corpus_dir).name, pairs=pairs)

    def _generate_easy(self, chunks: list[Chunk], count: int, rng: Random) -> list[QAPair]:
        pool = rng.sample(chunks, k=min(len(chunks), count * 3))
        pairs: list[QAPair] = []
        for chunk in pool:
            if len(pairs) >= count:
                break
            data = self._ask_json(
                "qa_single",
                key_parts=("easy", chunk.id, chunk.text),
                doc=chunk.doc,
                text=chunk.text,
            )
            if self._valid_qa(data):
                assert isinstance(data, dict)
                pairs.append(
                    QAPair(
                        question=str(data["question"]),
                        ground_truth_answer=str(data["answer"]),
                        relevant_source_ids=[chunk.id],
                        difficulty="easy",
                    )
                )
        return pairs

    def _generate_medium(self, chunks: list[Chunk], count: int, rng: Random) -> list[QAPair]:
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk.doc, []).append(chunk)
        docs = [doc for doc, doc_chunks in by_doc.items() if len(doc_chunks) >= 2]
        pairs: list[QAPair] = []
        attempts = 0
        while docs and len(pairs) < count and attempts < count * 3:
            attempts += 1
            doc = rng.choice(docs)
            doc_chunks = by_doc[doc]
            width = min(rng.choice([2, 3]), len(doc_chunks))
            start = rng.randrange(len(doc_chunks) - width + 1)
            window = doc_chunks[start : start + width]
            text = "\n\n---\n\n".join(c.text for c in window)
            ids = [c.id for c in window]
            data = self._ask_json(
                "qa_multi", key_parts=("medium", *ids, text), doc=doc, text=text
            )
            if self._valid_qa(data):
                assert isinstance(data, dict)
                pairs.append(
                    QAPair(
                        question=str(data["question"]),
                        ground_truth_answer=str(data["answer"]),
                        relevant_source_ids=ids,
                        difficulty="medium",
                    )
                )
        return pairs

    def _generate_hard(self, chunks: list[Chunk], count: int, rng: Random) -> list[QAPair]:
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk.doc, []).append(chunk)
        docs = sorted(by_doc)
        pairs: list[QAPair] = []
        attempts = 0
        while len(docs) >= 2 and len(pairs) < count and attempts < count * 3:
            attempts += 1
            doc_a, doc_b = rng.sample(docs, k=2)
            chunk_a = rng.choice(by_doc[doc_a])
            chunk_b = rng.choice(by_doc[doc_b])
            data = self._ask_json(
                "qa_cross",
                key_parts=("hard", chunk_a.id, chunk_b.id, chunk_a.text, chunk_b.text),
                doc_a=doc_a,
                text_a=chunk_a.text,
                doc_b=doc_b,
                text_b=chunk_b.text,
            )
            if self._valid_qa(data):
                assert isinstance(data, dict)
                pairs.append(
                    QAPair(
                        question=str(data["question"]),
                        ground_truth_answer=str(data["answer"]),
                        relevant_source_ids=[chunk_a.id, chunk_b.id],
                        difficulty="hard",
                    )
                )
        return pairs

    def _generate_unanswerable(
        self, chunks: list[Chunk], count: int, rng: Random
    ) -> list[QAPair]:
        pool = rng.sample(chunks, k=min(len(chunks), count * 3))
        pairs: list[QAPair] = []
        for chunk in pool:
            if len(pairs) >= count:
                break
            data = self._ask_json(
                "unanswerable",
                key_parts=("unanswerable", chunk.id, chunk.text),
                doc=chunk.doc,
                text=chunk.text,
            )
            if isinstance(data, dict) and data.get("question"):
                pairs.append(
                    QAPair(
                        question=str(data["question"]),
                        answerable=False,
                        difficulty="medium",
                    )
                )
        return pairs

    def _expand_paraphrases(self, pairs: list[QAPair], groups: int) -> list[QAPair]:
        bases = [p for p in pairs if p.answerable][:groups]
        expanded: list[QAPair] = []
        for i, base in enumerate(bases):
            group_id = f"pg_{i:04d}"
            base.paraphrase_group = group_id
            data = self._ask_json("paraphrase", key_parts=(base.question,), question=base.question)
            if not isinstance(data, list):
                continue
            for paraphrase in [str(p) for p in data if str(p).strip()][:PARAPHRASES_PER_GROUP]:
                expanded.append(
                    base.model_copy(update={"question": paraphrase, "paraphrase_group": group_id})
                )
        return expanded

    def _ask_json(self, prompt_name: str, key_parts: tuple[str, ...], **fields: str) -> object:
        raw = self.judge.ask(
            self.prompts[prompt_name],
            metric_name=GENERATION_METRIC,
            key_parts=key_parts,
            max_tokens=512,
            **fields,
        )
        return _parse_json_value(raw)

    @staticmethod
    def _valid_qa(data: object) -> bool:
        return isinstance(data, dict) and bool(data.get("question")) and bool(data.get("answer"))
