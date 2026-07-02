# Metrics reference

Every metric returns a 0.0-1.0 score plus per-sample scores. LLM-judged results always record the judge model and prompt version. Samples a metric cannot score (missing labels, no citations, etc.) are excluded from the aggregate and counted in `details`.

## Retrieval

### `hit_rate@k` (deterministic)
**What:** fraction of questions where any chunk in the top-k retrieved has a `source_id` in the question's `relevant_source_ids`.
**Needs:** labeled `relevant_source_ids`.
**Limitations:** counts only labeled chunks as relevant — a retrieved duplicate passage containing the same fact scores as a miss.

### `mrr` (deterministic)
**What:** mean reciprocal rank of the first relevant chunk (1/rank; 0 if absent).
**Needs/limitations:** same as hit_rate; more rank-sensitive.

### `context_precision` (LLM-judged)
**What:** each top-k chunk is judged RELEVANT/IRRELEVANT to the question; sample score is rank-aware precision (mean of precision@i at each relevant position).
**Needs:** nothing beyond the response.
**Limitations:** judge leniency inflates it; validate the judge first.

### `context_recall` (LLM-judged)
**What:** ground-truth answer is decomposed into atomic claims; score = fraction of claims attributable to the retrieved context.
**Needs:** `ground_truth_answer`.
**Limitations:** quality depends on ground-truth answers being complete and correct.

## Generation

### `faithfulness` (LLM-judged)
**What:** answer decomposed into atomic claims; each verified against retrieved context; score = supported/total. Claim-free answers (e.g. refusals) score 1.0 — nothing unfaithful to penalize.
**Limitations:** read together with retrieval metrics — a pipeline that retrieves nothing and says "I don't know" is perfectly faithful. Unsupported-but-true claims (from parametric knowledge) still count as unfaithful; that is intentional.

### `answer_relevance` (LLM-judged)
**What:** 1-5 rubric of how directly the answer addresses the question, normalized to 0-1. Justified refusals are instructed to score 5.
**Limitations:** single-call rubric; coarser than claim-level metrics.

### `citation_accuracy` (LLM-judged)
**What:** parses `[source_id]` citations per sentence; judges whether each cited chunk supports its sentence. Citing a never-retrieved source counts as a failure. Uncited answers are skipped.
**Needs:** answers that cite in `[source_id]` style.

## Robustness

### `refusal_calibration` (LLM-judged)
**What:** correct behavior = answer when `answerable`, refuse when not. Details split the failure modes: `false_answer_rate` (hallucinated answers to unanswerable questions) and `over_refusal_rate`.
**Needs:** dataset with `answerable: false` samples (`generate-dataset` produces them). Refusal detected via the response's `refused` flag, else LLM-judged.

### `paraphrase_consistency` (LLM-judged)
**What:** pairwise semantic-equivalence of answers within each `paraphrase_group`; group score = fraction of equivalent pairs; metric = mean over groups.
**Needs:** paraphrase groups in the dataset (`generate-dataset` produces 5-way groups).
**Limitations:** two consistently *wrong* answers count as consistent — pair with faithfulness.

## Reading scores together

The metrics are designed to disagree informatively: high faithfulness + low hit_rate = honest pipeline with bad retrieval; high hit_rate + low faithfulness = good retrieval, hallucinating generator; high everything + low paraphrase_consistency = brittle to phrasing.
