"""Shared components for benchmark pipelines.

Every pipeline uses the SAME chunking, embedding model, generator, and prompt;
the retrieval strategy is the only variable under test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ragcheck.adapters.base import RAGResponse, RetrievedChunk
from ragcheck.llm import LLMClient

DATA_DIR = Path(__file__).parent.parent / "corpus" / "data"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

GENERATION_PROMPT = """Answer the question using ONLY the provided documents.
If the documents do not contain the answer, say "The provided documents do not contain this information."
Keep the answer to one or two sentences. After each factual statement, cite the id of the document it came from in square brackets, e.g. [AAPL_10K_0042].

Documents:
{context}

Question: {question}"""


@dataclass(frozen=True)
class Chunk:
    """A corpus chunk with a stable id like ``AAPL_10K_0042``."""

    id: str
    text: str


def load_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    """Chunk every corpus text file with fixed size/overlap."""
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text()
        step = CHUNK_CHARS - CHUNK_OVERLAP
        index = 0
        for start in range(0, len(text), step):
            piece = text[start : start + CHUNK_CHARS].strip()
            if len(piece) > 100:  # drop boilerplate slivers
                chunks.append(Chunk(id=f"{path.stem}_{index:04d}", text=piece))
                index += 1
    if not chunks:
        raise FileNotFoundError(
            f"No corpus files in {data_dir}. Run benchmarks/corpus/fetch_corpus.py first."
        )
    return chunks


class DenseIndex:
    """Cosine-similarity retrieval over sentence-transformer embeddings."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.model = SentenceTransformer(EMBED_MODEL)
        self.embeddings = self.model.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )

    def query(self, question: str, top_n: int) -> list[tuple[Chunk, float]]:
        q = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ q
        order = scores.argsort()[::-1][:top_n]
        return [(self.chunks[i], float(scores[i])) for i in order]


class BM25Index:
    """Classic lexical retrieval."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([c.text.lower().split() for c in chunks])

    def query(self, question: str, top_n: int) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(question.lower().split())
        order = scores.argsort()[::-1][:top_n]
        return [(self.chunks[i], float(scores[i])) for i in order]


def generate_answer(
    llm: LLMClient, question: str, retrieved: list[tuple[Chunk, float]], retrieval_ms: float
) -> RAGResponse:
    """Shared generation step: same prompt and model for every pipeline."""
    context = "\n".join(f"[{chunk.id}] {chunk.text}" for chunk, _ in retrieved)
    t0 = time.perf_counter()
    completion = llm.complete(
        GENERATION_PROMPT.format(context=context or "(none)", question=question),
        max_tokens=256,
    )
    generation_ms = (time.perf_counter() - t0) * 1000
    answer = completion.text.strip()
    return RAGResponse(
        answer=answer,
        retrieved_chunks=[
            RetrievedChunk(content=chunk.text, source_id=chunk.id, score=score)
            for chunk, score in retrieved
        ],
        latencies_ms={"retrieval": retrieval_ms, "generation": generation_ms},
        token_usage={
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
        },
        refused="do not contain this information" in answer.lower(),
    )
