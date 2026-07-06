# RAGCheck

[![CI](https://github.com/Ekta-Shah/ragcheck/actions/workflows/ci.yaml/badge.svg)](https://github.com/Ekta-Shah/ragcheck/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

**pytest for RAG systems.** Wrap your retrieval-augmented generation pipeline in a one-method adapter and get quality scores, a failure taxonomy, cost/latency profiles, and CI-friendly regression checks — with a judge you can actually verify.

```bash
pip install ragcheck-eval   # imports and CLI are `ragcheck`
ragcheck demo               # no API key needed - see the whole workflow in 10 seconds
```

The demo runs a canned pipeline with two planted failures — faithfulness catches the hallucinated claim, refusal calibration flags the invented answer to an unanswerable question, and the HTML report shows both with their retrieved contexts. ([Or open the quickstart in Colab.](https://colab.research.google.com/github/Ekta-Shah/ragcheck/blob/main/examples/quickstart.ipynb))

For real runs:

```bash
export ANTHROPIC_API_KEY=sk-ant-...  # or GROQ_API_KEY for open-weight judges
ragcheck run examples/toy_config.yaml
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
pipeline tokens: {'input_tokens': 1410, 'output_tokens': 184}  |
judge tokens: {'input_tokens': 4338, 'output_tokens': 274}  |  cache: 0 hits / 26 misses
```

Alongside the terminal scorecard you get a JSON report and a **self-contained HTML report** — scorecard, judge-validation status, latency/cost, and the worst-failing samples with question, answer, and retrieved context inline.

## Why another RAG eval tool?

Existing tools score your pipeline with an LLM judge and stop there. RAGCheck is built around the questions that actually block shipping:

- **Can you trust the judge?** `ragcheck validate-judge labels.jsonl --metric faithfulness` measures LLM-judge vs. human agreement (Cohen's kappa + confusion matrix) *before* you trust judged metrics. On our own labels it caught a real miscalibration: the default pass threshold scored κ=0.40; the corrected threshold scores κ=1.00. Reports display "validated at κ=0.XX" — or a visible "not validated" flag.
- **Does your system know when to say "I don't know"?** `refusal_calibration` reports the false-answer rate (hallucinated answers to unanswerable questions) and over-refusal rate separately.
- **Is it robust?** `paraphrase_consistency` judges pairwise answer agreement within paraphrase groups — catching pipelines that only work on one phrasing.
- **No eval set?** `ragcheck generate-dataset corpus_dir/` builds one from your documents: easy/medium/hard tiers, unanswerable questions, paraphrase groups, chunk-level provenance. Cached and resumable.
- **What does quality cost?** The benchmark harness produces a cost-quality frontier across architectures.
- **Did this PR make it worse?** `ragcheck compare old.json new.json --fail-if "faithfulness<-0.05"` exits non-zero on regression and prints a markdown diff for PR comments.

## Compared to existing tools

| Capability | RAGCheck | RAGAS | DeepEval | TruLens |
|---|:---:|:---:|:---:|:---:|
| Core RAG metrics (faithfulness, context precision/recall, relevance) | ✅ | ✅ | ✅ | ✅ |
| Built-in judge validation vs. human labels (Cohen's κ) | ✅ | ❌ | ❌ | ❌ |
| Refusal calibration (false-answer + over-refusal rates) | ✅ | ❌ | ❌ | ❌ |
| Paraphrase-consistency robustness testing | ✅ | ❌ | ❌ | ❌ |
| Persistent judgment cache keyed on judge model + prompt version | ✅ | ❌ | partial | ❌ |
| Report diffing with CI exit codes | ✅ | ❌ | ✅ | ❌ |
| Runs entirely local — no dashboard, account, or service | ✅ | ✅ | partial | partial |

*Comparison reflects our reading of each tool's docs at the time of writing; corrections welcome via issues.*

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

LangChain users: `LangChainAdapter(retriever, chain)` wraps a retriever + chain directly.

Point a YAML config at your adapter, dataset, and metrics, then `ragcheck run config.yaml`. Every LLM judgment is cached in SQLite (keyed on judge model, prompt version, and inputs) so re-runs on unchanged data are free — and every judged result records exactly which judge and prompt produced it.

## Benchmark: which RAG architecture?

Four pipelines — naive dense, hybrid (BM25+dense RRF), cross-encoder reranked, and agentic (query decomposition) — on 8 SEC 10-K filings, identical chunking/generator, retrieval as the only variable. Preliminary 12-sample free-tier run (differences under ~0.2 are noise; full 300-500 sample run pending):

| Pipeline | faithfulness | refusal_calibration | paraphrase_consistency | tok/q | retrieval p50 |
|---|---:|---:|---:|---:|---:|
| naive (dense) | 0.950 | 0.583 | 0.700 | 1535 | 43 ms |
| hybrid (BM25+dense RRF) | 0.958 | 0.500 | **1.000** | 1448 | 75 ms |
| reranked (cross-encoder) | 0.958 | 0.500 | **1.000** | 1511 | 359 ms |
| agentic (decomposition) | 0.958 | **0.750** | 0.450 | 2683 | 6602 ms |

Early hypothesis worth noticing: agentic RAG refuses unanswerables best but answers paraphrases least consistently, at 1.75× the cost — exactly the trade-offs single-metric evals miss. Methodology, frontier chart, and honest limitations: [benchmarks/](benchmarks/README.md).

## Design decisions

- **Judge validation is a feature, not a footnote.** Judged metrics are untrustworthy by default; the κ workflow makes trust measurable, and reports flag unvalidated judges.
- **Deterministic first.** Where a metric can be computed without an LLM (hit_rate, MRR), it is. LLM judging is reserved for what actually needs it.
- **Everything cached, everything versioned.** Judgments are cached on (judge model, prompt version, inputs); prompts live as versioned markdown. Interrupted dataset generation resumes for free; changing a prompt never silently reuses stale verdicts.
- **CI is the target environment.** JSON reports, exit codes, markdown diffs, an env-gated smoke workflow, and a cost guard (`--yes` required above a configurable sample count).
- **No UI.** The self-contained HTML report is the only visual output. No dashboard, no server, no account.

## Docs

[Quickstart](docs/quickstart.md) · [Metrics reference](docs/metrics.md) · [Judge validation](docs/judge-validation.md) · [CI integration](docs/ci-integration.md)

## Status

| Area | Status |
|---|---|
| Zero-key demo (`ragcheck demo`) + Colab quickstart notebook | ✅ |
| Adapters: `FunctionAdapter`, `LangChainAdapter` | ✅ |
| 9 metrics: retrieval (4) · generation (3) · robustness (2) | ✅ |
| Judge validation (`validate-judge`, Cohen's κ) | ✅ |
| Synthetic datasets (`generate-dataset`) | ✅ |
| JSON + self-contained HTML reports | ✅ |
| Regression diffing (`compare --fail-if`) | ✅ |
| CI workflows (lint/type/test + env-gated live smoke eval) | ✅ |
| 4-architecture benchmark harness | ✅ |
| 4-pipeline benchmark: preliminary 12-sample results | ✅ |
| Full benchmark run (300-500 samples) | 🔜 needs funded API budget |
| PyPI release (`ragcheck-eval` 0.1.0) | 🔜 with benchmark results |
| LlamaIndex/Haystack adapters | ❌ not in v0.1 (issues welcome) |

Explicit non-goals: web UI/dashboards, hosted services, fine-tuning or embedding-model evaluation.

## Development

```bash
pip install -e ".[dev]"
pytest            # unit tests (no API calls - judges are mocked)
ruff check .
mypy ragcheck/
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
