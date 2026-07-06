# Quickstart

60 seconds from install to scorecard.

## 1. Install

```bash
pip install -e .           # from a clone; PyPI: pip install ragcheck (0.1.0)
export ANTHROPIC_API_KEY=sk-ant-...   # or GROQ_API_KEY for open-weight judges
```

## 2. Wrap your pipeline

Any `(question: str) -> RAGResponse` callable works:

```python
# my_pipeline.py
from ragcheck import RAGResponse, RetrievedChunk
from ragcheck.adapters import FunctionAdapter

def pipeline(question: str) -> RAGResponse:
    chunks = my_retriever(question)          # your retrieval
    answer = my_generator(question, chunks)  # your generation
    return RAGResponse(
        answer=answer,
        retrieved_chunks=[
            RetrievedChunk(content=c.text, source_id=c.id) for c in chunks
        ],
        refused="don't know" in answer.lower(),   # optional but powers refusal metrics
    )

adapter = FunctionAdapter(pipeline)
```

Using LangChain? `LangChainAdapter(retriever, chain)` wraps a retriever + LCEL chain directly.

## 3. Point a config at it

```yaml
# eval.yaml
dataset: eval_set.jsonl        # {"question": ..., "relevant_source_ids": [...], ...} per line
adapter: my_pipeline:adapter
metrics:
  - name: hit_rate
    params: { k: 5 }
  - mrr
  - faithfulness
  - answer_relevance
judge_provider: anthropic      # or groq
```

No eval set? Generate one from your documents:

```bash
ragcheck generate-dataset ./my_docs --n 100 --out eval_set.jsonl
```

## 4. Run

```bash
ragcheck run eval.yaml
```

You get a terminal scorecard, a JSON report, and a self-contained HTML report (open it in any browser - worst-failing samples included). Re-runs are cheap: every judgment is cached in SQLite.

## 5. Trust, then automate

```bash
ragcheck validate-judge labels.jsonl --metric faithfulness   # judge vs. human kappa
ragcheck compare baseline.json new.json --fail-if "faithfulness<-0.05"  # CI gate
```

See [metrics.md](metrics.md), [judge-validation.md](judge-validation.md), and [ci-integration.md](ci-integration.md).
