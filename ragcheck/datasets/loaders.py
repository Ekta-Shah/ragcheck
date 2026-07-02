"""Load evaluation datasets from JSONL or CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ragcheck.datasets.models import EvalDataset, QAPair


def load_dataset(path: str | Path) -> EvalDataset:
    """Load a dataset from a ``.jsonl`` or ``.csv`` file.

    JSONL: one QAPair object per line. CSV: columns matching QAPair fields;
    ``relevant_source_ids`` is a semicolon-separated string.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        pairs = [
            QAPair.model_validate(json.loads(line))
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    elif path.suffix == ".csv":
        pairs = []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                raw_ids = row.pop("relevant_source_ids", "") or ""
                data: dict[str, object] = {k: v for k, v in row.items() if v not in (None, "")}
                data["relevant_source_ids"] = [s for s in raw_ids.split(";") if s]
                pairs.append(QAPair.model_validate(data))
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix} (use .jsonl or .csv)")
    return EvalDataset(name=path.stem, pairs=pairs)
