"""Dataset data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ragcheck.adapters.base import RAGResponse


class QAPair(BaseModel):
    """A single evaluation question with optional ground truth."""

    question: str
    ground_truth_answer: str | None = None
    relevant_source_ids: list[str] = Field(default_factory=list)
    answerable: bool = True
    difficulty: str = "medium"
    paraphrase_group: str | None = None


class EvalDataset(BaseModel):
    """A named collection of QA pairs."""

    name: str = "dataset"
    pairs: list[QAPair]


class EvalSample(BaseModel):
    """A QA pair joined with the pipeline's response - the unit metrics consume."""

    qa: QAPair
    response: RAGResponse
