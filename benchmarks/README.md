# RAGCheck Architecture Benchmark

Reproducible comparison of RAG retrieval architectures on a financial-filings corpus. Everything except the retrieval strategy is held constant: same chunking (1200 chars, 200 overlap), same embedding model (`all-MiniLM-L6-v2`), same generator model and prompt.

> **Status: rough early cut.** This is the plan's "demoable benchmark early" checkpoint - 2 of 4 pipelines, 16 auto-generated QA pairs, 2 metrics. The full version (4 pipelines, 300-500 samples across difficulty tiers, unanswerables, paraphrase groups, cost-quality frontier chart) lands after the full metric suite and synthetic dataset generator are built.

## Pipelines

| Pipeline | Retrieval |
|---|---|
| `naive_rag` | Dense (MiniLM cosine) top-5 |
| `hybrid_rag` | BM25 + dense top-20 each, reciprocal rank fusion (k=60), top-5 |
| `reranked_rag` | 🔜 dense top-20 → cross-encoder rerank → top-5 |
| `agentic_rag` | 🔜 LLM query decomposition → iterative retrieval (max 3 rounds) |

Generator/judge: `meta-llama/llama-4-scout-17b-16e-instruct` via Groq (temperature 0), selected with `RAGCHECK_BENCH_MODEL`.

## Corpus

Latest 10-K filings of Apple, Microsoft, and NVIDIA, fetched from SEC EDGAR and converted to plain text (~900 KB, ~1,100 chunks). Raw data is gitignored; the fetch script is deterministic.

## Reproduce

```bash
pip install -e . sentence-transformers rank_bm25
export GROQ_API_KEY=...
export RAGCHECK_BENCH_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
python benchmarks/corpus/fetch_corpus.py     # ~1 MB from EDGAR
python benchmarks/run_benchmark.py           # or --quick for 8 samples
```

The committed `benchmark_dataset.jsonl` reproduces these numbers; regenerate it with `python benchmarks/generate_qa.py --n 16` after re-fetching the corpus. Per-pipeline JSON reports land in `benchmarks/results/`.

## Results (16 samples, 2026-07-02)

| Pipeline | hit_rate@5 | faithfulness | tokens/query | retrieval p50 (ms) |
|---|---:|---:|---:|---:|
| `naive_rag` (dense) | 0.438 | 1.000 | 1666 | 40 |
| `hybrid_rag` (BM25 + dense, RRF) | **0.750** | 0.948 | 1555 | 47 |

**Read:** on 10-K filings — dense-heavy with exact figures, entity names, and legal terms — adding BM25 lexical matching via reciprocal rank fusion lifts hit_rate@5 from 0.44 to 0.75 for ~7 ms extra retrieval latency. Both pipelines stay highly faithful *to what they retrieve*; the naive pipeline's perfect faithfulness with poor retrieval illustrates why retrieval and generation metrics must be read together (a pipeline can faithfully report that the answer isn't in its badly-retrieved context).

## Honest limitations

- **Small sample (16).** Differences of a few points are noise at this size.
- **QA pairs are single-chunk and auto-generated.** Each question is grounded in exactly one chunk, and `hit_rate` counts only that chunk as relevant - a retrieved *duplicate* passage with the same fact counts as a miss. Some generated questions lean on filing boilerplate rather than substantive financials.
- **No judge validation yet.** Faithfulness scores come from an unvalidated LLM judge; the judge-vs-human agreement module is Phase 2.
- **Corpus text extraction is rough.** HTML tables flatten into text streams; some chunks are noisy.
- **10-K filings shift over time.** The fetch script pulls each company's *latest* 10-K, so exact numbers will drift as new filings land; re-generate the dataset after re-fetching.
