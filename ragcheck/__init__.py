"""RAGCheck: pytest for RAG systems."""

from ragcheck.adapters.base import RAGAdapter, RAGResponse, RetrievedChunk
from ragcheck.datasets.models import EvalDataset, EvalSample, QAPair
from ragcheck.metrics.base import Metric, MetricResult

__version__ = "0.1.0"

__all__ = [
    "EvalDataset",
    "EvalSample",
    "Metric",
    "MetricResult",
    "QAPair",
    "RAGAdapter",
    "RAGResponse",
    "RetrievedChunk",
]
