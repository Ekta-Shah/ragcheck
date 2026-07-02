"""Rough QA-pair generation for the early benchmark (full version lands in Phase 3).

Samples corpus chunks and asks the LLM to write one grounded question per chunk.
Writes benchmarks/benchmark_dataset.jsonl with chunk-id provenance.

    python benchmarks/generate_qa.py --n 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pipelines"))

from common import Chunk, load_chunks  # noqa: E402

from ragcheck.llm import default_client  # noqa: E402

OUT_PATH = Path(__file__).parent / "benchmark_dataset.jsonl"
SEED = 13

QA_PROMPT = """Below is an excerpt from a company's SEC 10-K filing. Write ONE factual question that is fully answered by this excerpt alone, plus its short answer.

Rules:
- The question must be self-contained: name the company and be specific enough that it has a single correct answer.
- The answer must be a short fact stated in the excerpt (a number, name, date, or short phrase).
- Skip boilerplate: if the excerpt has no interesting fact, still pick the most concrete detail available.

Excerpt (from {doc}):
{text}

Respond with ONLY a JSON object: {{"question": "...", "answer": "..."}}"""


def generate_pair(llm, chunk: Chunk) -> dict | None:
    doc = chunk.id.rsplit("_", 1)[0].replace("_", " ")
    raw = llm.complete(QA_PROMPT.format(doc=doc, text=chunk.text), max_tokens=256).text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not data.get("question") or not data.get("answer"):
        return None
    return {
        "question": data["question"],
        "ground_truth_answer": data["answer"],
        "relevant_source_ids": [chunk.id],
        "difficulty": "easy",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    args = parser.parse_args()

    llm = default_client(os.environ.get("RAGCHECK_BENCH_MODEL"))
    chunks = load_chunks()
    # Skip the first chunks of each doc (cover pages / tables of contents).
    candidates = [c for c in chunks if int(c.id.rsplit("_", 1)[1]) > 20]
    random.seed(SEED)
    sampled = random.sample(candidates, k=min(args.n * 2, len(candidates)))

    pairs = []
    for chunk in sampled:
        if len(pairs) >= args.n:
            break
        pair = generate_pair(llm, chunk)
        if pair:
            pairs.append(pair)
            print(f"[{len(pairs)}/{args.n}] {pair['question'][:80]}")

    OUT_PATH.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    print(f"Wrote {len(pairs)} pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
