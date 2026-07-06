# Changelog

## 0.1.1 (2026-07-06)

- Documentation cleanup across README, docs, and package metadata. No code changes.

## 0.1.0 (2026-07-06)

Initial release. `pip install ragcheck-eval`.

- `RAGAdapter` / `FunctionAdapter` / `LangChainAdapter` pipeline wrappers
- Retrieval metrics: `hit_rate@k`, `mrr`, `context_precision`, `context_recall`
- Generation metrics: `faithfulness`, `answer_relevance`, `citation_accuracy`
- Robustness metrics: `refusal_calibration`, `paraphrase_consistency`
- Judge validation: `ragcheck validate-judge` (Cohen's kappa, confusion matrix vs. human labels)
- Synthetic datasets: `ragcheck generate-dataset` (difficulty tiers, unanswerables, paraphrase groups, resumable)
- SQLite judgment cache keyed on judge model + prompt version + inputs
- Versioned judge prompts (markdown + YAML frontmatter)
- Judge providers: Anthropic, Groq; parallel judging with rate-limit backoff
- JSON + self-contained HTML reports with worst-failing-sample drilldowns
- `ragcheck compare` regression diffing with `--fail-if` thresholds (CI-friendly exit codes)
- Reproducible 4-architecture benchmark (naive / hybrid / reranked / agentic RAG) on SEC 10-Ks
- Cost guard: large LLM-judged runs require `--yes`
