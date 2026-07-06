"""Zero-key demo: canned pipeline + deterministic offline judge.

``ragcheck demo`` shows the full workflow - scorecard, failure drilldowns,
HTML report - without an API key. The pipeline's answers are canned (two are
deliberately wrong so the metrics have something to catch) and the judge is a
deterministic word-overlap stand-in, clearly labeled as such. Real runs use
real LLM judges.
"""

from __future__ import annotations

import json
import re

from ragcheck.adapters.base import RAGAdapter, RAGResponse, RetrievedChunk
from ragcheck.datasets.models import EvalDataset, QAPair
from ragcheck.llm import LLMResponse

DEMO_CORPUS: dict[str, str] = {
    "doc_leave": "Acme Analytics employees receive 24 days of paid leave per year, plus 8 public holidays.",
    "doc_wfh": "Remote work is allowed up to 3 days per week; Tuesdays are mandatory in-office days.",
    "doc_insurance": "Health insurance covers employees, spouses, and up to two children with a 5 lakh floater.",
    "doc_notice": "The notice period is 60 days for all roles; garden leave applies to client-facing staff.",
}

REFUSAL = "The provided documents do not contain this information."

# (question, canned answer, relevant docs, answerable)
DEMO_CASES: list[tuple[str, str, list[str], bool]] = [
    (
        "How many days of paid leave do employees get per year?",
        "Employees receive 24 days of paid leave per year.",
        ["doc_leave"],
        True,
    ),
    (
        "How many days per week is remote work allowed?",
        "Remote work is allowed up to 3 days per week.",
        ["doc_wfh"],
        True,
    ),
    (
        "Who is covered under health insurance?",
        "Health insurance covers employees, spouses, and up to two children.",
        ["doc_insurance"],
        True,
    ),
    (
        # Planted hallucination: second claim is not in any document.
        "What is the leave policy?",
        "Employees receive 24 days of paid leave per year. Employees also get unlimited sick leave on request.",
        ["doc_leave"],
        True,
    ),
    (
        # Correct refusal on an unanswerable question.
        "What is the parental leave policy?",
        REFUSAL,
        [],
        False,
    ),
    (
        # False answer: confidently invented for an unanswerable question.
        "What gym membership does the company provide?",
        "The company provides a premium gym membership at all Cult.fit centers.",
        [],
        False,
    ),
]


class DemoPipeline(RAGAdapter):
    """Keyword retrieval over the demo corpus with canned answers."""

    name = "demo_pipeline"

    def __init__(self) -> None:
        self._answers = {question: answer for question, answer, _, _ in DEMO_CASES}

    def query(self, question: str) -> RAGResponse:
        terms = {w.strip("?.,").lower() for w in question.split()}
        scored = sorted(
            (
                (sum(t in text.lower() for t in terms), doc_id, text)
                for doc_id, text in DEMO_CORPUS.items()
            ),
            reverse=True,
        )
        chunks = [
            RetrievedChunk(content=text, source_id=doc_id, score=float(score))
            for score, doc_id, text in scored[:3]
            if score > 0
        ]
        answer = self._answers[question]
        return RAGResponse(
            answer=answer,
            retrieved_chunks=chunks,
            latencies_ms={"retrieval": 1.0, "generation": 42.0},
            token_usage={"input_tokens": 380, "output_tokens": 40},
            refused=answer == REFUSAL,
        )


def demo_dataset() -> EvalDataset:
    """The six-question demo dataset (two designed failures included)."""
    return EvalDataset(
        name="demo",
        pairs=[
            QAPair(question=q, relevant_source_ids=ids, answerable=answerable)
            for q, _, ids, answerable in DEMO_CASES
        ],
    )


def _between(text: str, start: str, end: str) -> str:
    portion = text.split(start, 1)[-1]
    return portion.split(end, 1)[0] if end in portion else portion


def _content_words(text: str) -> set[str]:
    stop = {"the", "a", "an", "of", "to", "and", "is", "are", "per", "on", "at", "all", "also"}
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop}


class DemoJudgeClient:
    """Deterministic offline stand-in for an LLM judge (demo mode only).

    Decomposition splits answers into sentences; verification checks content-word
    overlap with the context; refusal detection is phrase-based. Good enough to
    demonstrate the workflow honestly - not a real judge.
    """

    model = "offline-demo-judge"

    def __init__(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 1024
    ) -> LLMResponse:
        if "atomic factual claims" in prompt:
            answer = _between(prompt, "Answer:", "Respond with").strip()
            if REFUSAL.lower() in answer.lower():
                text = "[]"
            else:
                claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
                text = json.dumps(claims)
        elif "SUPPORTED or UNSUPPORTED" in prompt:
            context = _between(prompt, "Retrieved documents:", "Claim:")
            claim = _between(prompt, "Claim:", "Respond with")
            words = _content_words(claim)
            overlap = len(words & _content_words(context)) / len(words) if words else 0.0
            text = "SUPPORTED" if overlap >= 0.6 else "UNSUPPORTED"
        elif "ANSWER or REFUSAL" in prompt:
            answer = _between(prompt, "Response:", "Respond with")
            text = "REFUSAL" if "do not contain" in answer.lower() else "ANSWER"
        elif "integer rating" in prompt:
            answer = _between(prompt, "Answer:", "Respond with")
            if "do not contain" in answer.lower():
                text = "5"  # justified refusal, per the rubric
            else:
                question_words = _content_words(_between(prompt, "Question:", "Answer:"))
                hit = len(question_words & _content_words(answer))
                text = "5" if hit >= 2 else "3" if hit == 1 else "1"
        else:
            text = "EQUIVALENT"
        return LLMResponse(text=text, input_tokens=0, output_tokens=0, model=self.model)
