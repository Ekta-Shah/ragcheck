"""Toy RAG pipeline: keyword retrieval over a 10-doc in-memory corpus + Claude generation.

Run from the repo root (requires ANTHROPIC_API_KEY or GROQ_API_KEY):

    ragcheck run examples/toy_config.yaml
"""

from __future__ import annotations

import time

from ragcheck.adapters.base import RAGAdapter, RAGResponse, RetrievedChunk
from ragcheck.adapters.function import FunctionAdapter
from ragcheck.llm import default_client

CORPUS: dict[str, str] = {
    "doc_leave": "Acme Analytics employees receive 24 days of paid leave per year, plus 8 public holidays.",
    "doc_wfh": "Remote work is allowed up to 3 days per week; Tuesdays are mandatory in-office days.",
    "doc_expense": "Expense claims must be filed within 30 days and require receipts above INR 500.",
    "doc_laptop": "New joiners choose between a MacBook Air M3 and a ThinkPad X1 Carbon on day one.",
    "doc_insurance": "Health insurance covers employees, spouses, and up to two children with a 5 lakh floater.",
    "doc_travel": "Business travel is booked through the Navan portal; economy class for flights under 6 hours.",
    "doc_probation": "The probation period is 90 days, after which a confirmation review is held.",
    "doc_referral": "The employee referral bonus is INR 50,000, paid after the referred hire completes probation.",
    "doc_learning": "Each employee has an annual learning budget of INR 40,000 for courses and conferences.",
    "doc_notice": "The notice period is 60 days for all roles; garden leave applies to client-facing staff.",
}

GENERATION_PROMPT = """Answer the question using ONLY the provided documents.
If the documents do not contain the answer, say "The provided documents do not contain this information."
Keep the answer to one or two sentences.

Documents:
{context}

Question: {question}"""

STOPWORDS = frozenset(
    ["a", "an", "the", "is", "are", "what", "how", "many", "much", "do", "does",
     "for", "of", "in", "on", "to", "and", "or", "per"]
)


def retrieve(question: str, k: int = 3) -> list[RetrievedChunk]:
    """Rank corpus docs by keyword overlap with the question."""
    terms = {w.strip("?.,").lower() for w in question.split()} - STOPWORDS
    scored = [
        (sum(term in text.lower() for term in terms), doc_id, text)
        for doc_id, text in CORPUS.items()
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        RetrievedChunk(content=text, source_id=doc_id, score=float(score))
        for score, doc_id, text in scored[:k]
        if score > 0
    ]


def build_adapter() -> RAGAdapter:
    """Factory used by toy_config.yaml. Uses whichever provider key is in the env."""
    llm = default_client()

    def pipeline(question: str) -> RAGResponse:
        t0 = time.perf_counter()
        chunks = retrieve(question)
        t1 = time.perf_counter()
        context = "\n".join(f"[{c.source_id}] {c.content}" for c in chunks)
        completion = llm.complete(
            GENERATION_PROMPT.format(context=context or "(none)", question=question),
            max_tokens=256,
        )
        t2 = time.perf_counter()
        answer = completion.text.strip()
        return RAGResponse(
            answer=answer,
            retrieved_chunks=chunks,
            latencies_ms={"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000},
            token_usage={
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
            },
            refused="do not contain this information" in answer.lower(),
        )

    return FunctionAdapter(pipeline)
