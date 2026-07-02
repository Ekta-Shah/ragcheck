# RAGCheck

**pytest for RAG systems.** Wrap your retrieval-augmented generation pipeline in a one-method adapter and get quality scores, a failure taxonomy, cost/latency profiles, and CI-friendly regression checks.

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
ragcheck run examples/toy_config.yaml
```

```text
        RAGCheck - toy-demo
┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric       ┃ Score ┃ Samples ┃ Judge                ┃
┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ hit_rate@3   │ 1.000 │      10 │ -                    │
│ faithfulness │ 1.000 │      10 │ claude-opus-4-8 (v1) │
└──────────────┴───────┴─────────┴──────────────────────┘
cache: 0 hits / 32 misses   (re-run: 32 hits / 0 misses, near-instant)
```

## Why another RAG eval tool?

Existing tools score your pipeline with an LLM judge and stop there. RAGCheck is built around the questions that actually block shipping:

- **Can you trust the judge?** A judge-validation module measures LLM-judge vs. human agreement (Cohen's kappa) *before* you trust judged metrics.
- **Does your system know when to say "I don't know"?** Refusal calibration tests behavior on unanswerable questions.
- **Is it robust?** Paraphrase-consistency scoring catches pipelines that only work on one phrasing.
- **What does quality cost?** Cost-quality frontier across pipeline configurations.
- **Did this PR make it worse?** Report diffing with thresholds and non-zero exit codes for CI.

## How it works

Wrap any pipeline in a `RAGAdapter` (or just a function):

```python
from ragcheck import RAGResponse, RetrievedChunk
from ragcheck.adapters import FunctionAdapter

def my_pipeline(question: str) -> RAGResponse:
    chunks = my_retriever(question)
    answer = my_generator(question, chunks)
    return RAGResponse(
        answer=answer,
        retrieved_chunks=[RetrievedChunk(content=c.text, source_id=c.id) for c in chunks],
    )

adapter = FunctionAdapter(my_pipeline)
```

Point a YAML config at your adapter, dataset, and metrics:

```yaml
dataset: eval_set.jsonl
adapter: my_project.pipeline:adapter
metrics:
  - name: hit_rate
    params: { k: 5 }
  - faithfulness
```

Then `ragcheck run config.yaml`. Every LLM judgment is cached in SQLite (keyed on metric, prompt version, and inputs), so re-runs on unchanged data are free and fast. Every judged result records the judge model and prompt version.

## Status: early and honest about it

| Area | Status |
|---|---|
| Adapter interface (`RAGAdapter`, `FunctionAdapter`) | ✅ done |
| `hit_rate@k` (deterministic) | ✅ done |
| `faithfulness` (claim decomposition + verification, LLM-judged) | ✅ done |
| SQLite judgment cache + prompt versioning | ✅ done |
| CLI (`ragcheck run`) + JSON report + terminal scorecard | ✅ done |
| MRR, context precision/recall, answer relevance, citation accuracy | 🔜 planned |
| Judge validation (Cohen's kappa vs. human labels) | 🔜 planned |
| Refusal calibration + paraphrase consistency | 🔜 planned |
| Synthetic dataset generation from any corpus | 🔜 planned |
| Architecture benchmark: naive vs. hybrid vs. reranked vs. agentic RAG | 🔜 planned |
| HTML report, regression diffing, PyPI release | 🔜 planned |

## Development

```bash
pip install -e ".[dev]"
pytest            # unit tests (no API calls - judges are mocked)
ruff check .
mypy ragcheck/
```

## License

MIT
