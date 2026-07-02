# Judge validation

LLM-judged metrics are only as trustworthy as the judge. RAGCheck treats that as measurable, not assumed: before relying on judged scores, quantify judge-vs-human agreement.

## 1. Label some samples

Create a JSONL file where each line is a human verdict on one (question, answer, context) triple:

```json
{"question": "...", "answer": "...", "context": "..." , "human_label": 1}
{"question": "...", "answer": "...", "context": ["chunk 1", "chunk 2"], "human_label": 0}
```

`human_label`: 1 = pass (e.g. faithful), 0 = fail. 30-50 labels covering both classes gives a usable kappa; include the hard cases (partially-correct answers, subtle hallucinations), not just obvious ones.

## 2. Run the validation

```bash
ragcheck validate-judge labels.jsonl --metric faithfulness \
  --provider groq --model llama-3.3-70b-versatile --threshold 1.0
```

Output: agreement rate, **Cohen's kappa**, and the confusion matrix (judge vs. human), printed and saved as JSON.

## 3. Interpret kappa

| kappa | Read |
|---|---|
| ≥ 0.8 | strong agreement — judged scores are trustworthy |
| 0.6-0.8 | usable, but inspect the confusion matrix for asymmetric errors |
| 0.4-0.6 | weak — tune the threshold/prompt/model before trusting |
| < 0.4 | do not trust this judge for this metric |

**The threshold matters as much as the judge.** A sample's judge label is `per-sample score >= threshold`. In our own validation, a scout-17B faithfulness judge scored kappa=0.40 at threshold 0.5 (answers mixing one true and one false claim scored 0.5 and passed) and kappa=1.00 at threshold 1.0 — same judge, same verdicts, different operating point. Sweep the threshold before swapping models.

## 4. Embed the result in eval reports

Point your eval config at the saved validation report:

```yaml
judge_validation: ragcheck_output/judge_validation_faithfulness.json
```

The HTML report then displays "validated at kappa=0.XX" alongside the scorecard — and "not validated" otherwise, so unvalidated judged numbers are visibly flagged.

## When to re-validate

Any time you change the judge model, bump a judge prompt version, or move to a very different domain. Cached judgments are keyed on model + prompt version, so validation runs are cheap to repeat.
