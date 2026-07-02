# RAGCheck

**pytest for RAG systems.** Wrap your retrieval-augmented generation pipeline in a one-method adapter and get quality scores, a failure taxonomy, cost/latency profiles, and CI-friendly regression checks.

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...   # or GROQ_API_KEY for open-weight judges
ragcheck run examples/toy_config.yaml # or examples/toy_config_groq.yaml
```

Real output from the bundled toy pipeline (keyword retrieval over a 10-doc corpus):

```text
                    RAGCheck - toy-demo-groq
┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric       ┃ Score ┃ Samples ┃ Judge                        ┃
┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ hit_rate@3   │ 1.000 │      10 │ -                            │
│ faithfulness │ 1.000 │      10 │ llama-3.3-70b-versatile (v1) │
└──────────────┴───────┴─────────┴──────────────────────────────┘
       Latency (ms)
┏━━━━━━━━━━━━┳━━━━━┳━━━━━┓
┃ Stage      ┃ p50 ┃ p95 ┃
┡━━━━━━━━━━━━╇━━━━━╇━━━━━┩
│ generation │ 378 │ 891 │
│ retrieval  │   0 │   0 │
└────────────┴─────┴─────┘
pipeline tokens: {'input_tokens': 1410, 'output_tokens': 184}  |
judge tokens: {'input_tokens': 4338, 'output_tokens': 274}  |  cache: 0 hits / 26 misses
```

On re-run, unchanged answers are judged from cache (23/26 hits above — the misses were answers the pipeline re-phrased, which correctly re-judge).

## Why another RAG eval tool?

Existing tools score your pipeline with an LLM judge and stop there. RAGCheck is built around the questions that actually block shipping:

- **Can you trust the judge?** `ragcheck validate-judge labels.jsonl --metric faithfulness` measures LLM-judge vs. human agreement (Cohen's kappa + confusion matrix) *before* you trust judged metrics. On our sample labels it caught a real miscalibration: the default pass threshold scored κ=0.40, and the fix (threshold 1.0) scores κ=1.00 — the module tells you your judge's safe operating point.
- **Does your system know when to say "I don't know"?** `refusal_calibration` reports the false-answer rate (hallucinated answers to unanswerable questions) and over-refusal rate separately.
- **Is it robust?** `paraphrase_consistency` judges pairwise answer agreement within paraphrase groups — catching pipelines that only work on one phrasing.
- **No eval set?** `ragcheck generate-dataset corpus_dir/` builds one from your documents: easy/medium/hard tiers, unanswerable questions, paraphrase groups, full chunk-level provenance. Generation is cached, so interrupted runs resume for free.
- **What does quality cost?** Cost-quality frontier across pipeline configurations.
- **Did this PR make it worse?** Report diffing with thresholds and non-zero exit codes for CI.

## Early benchmark result

Same corpus (3 SEC 10-Ks), same chunking, same generator — only retrieval varies:

| Pipeline | hit_rate@5 | faithfulness | tokens/query |
|---|---:|---:|---:|
| naive (dense) | 0.438 | 1.000 | 1666 |
| hybrid (BM25 + dense, RRF) | **0.750** | 0.948 | 1555 |

16 samples — directional, not definitive. Methodology, reproduction commands, and honest limitations in [benchmarks/](benchmarks/README.md).

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
| Retrieval metrics: `hit_rate@k`, `mrr` (deterministic), `context_precision`, `context_recall` (LLM-judged) | ✅ done |
| Generation metrics: `faithfulness`, `answer_relevance`, `citation_accuracy` (LLM-judged) | ✅ done |
| **Judge validation**: `ragcheck validate-judge` — Cohen's kappa + confusion matrix vs. human labels | ✅ done |
| Parallel judging (configurable concurrency, thread-safe cache, rate-limit backoff) | ✅ done |
| SQLite judgment cache + prompt versioning | ✅ done |
| Judge providers: Anthropic (Claude) + Groq (open-weight Llama) | ✅ done |
| Robustness metrics: `refusal_calibration` (false-answer + over-refusal rates), `paraphrase_consistency` | ✅ done |
| Synthetic datasets: `ragcheck generate-dataset` — difficulty tiers, unanswerables, paraphrase groups, resumable | ✅ done |
| Benchmark harness: 4 architectures (naive/hybrid/reranked/agentic), 9 metrics, cost-quality frontier ([methodology](benchmarks/README.md)) | ✅ done |
| CLI (`ragcheck run`) + JSON report + terminal scorecard | ✅ done |
| Full benchmark run (300-500 samples) - awaiting funded API budget | 🔜 next |
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
