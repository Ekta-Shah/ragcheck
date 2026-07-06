# CI integration

RAGCheck is built to gate PRs: run an eval, diff against a baseline, fail the build on regression.

## The core loop

```bash
# 1. Run the eval on your pipeline
ragcheck run eval.yaml --yes

# 2. Compare against the committed baseline; exit 1 on breach
ragcheck compare baselines/main.json ragcheck_output/my-run.json \
  --fail-if "faithfulness<-0.05" \
  --fail-if "hit_rate@5<-0.05"
```

`--fail-if metric<-0.05` fails when the metric dropped by more than 0.05 versus baseline. `compare` prints a markdown table (old/new/delta/status) you can post as a PR comment.

## GitHub Actions example

```yaml
name: rag-eval
on: pull_request

jobs:
  eval:
    runs-on: ubuntu-latest
    env:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e .
      - run: ragcheck run eval.yaml --yes
      - run: |
          ragcheck compare baselines/main.json ragcheck_output/pr-run.json \
            --fail-if "faithfulness<-0.05" > diff.md
      - name: Comment on PR
        if: always()
        uses: marocchino/sticky-pull-request-comment@v2
        with: { path: diff.md }
```

## Keeping cost sane in CI

- **Small fixed eval set** (30-100 samples) for PR gates; the full set nightly.
- **Commit the judgment cache?** No - cache on question/answer/context, so changed answers re-judge anyway. Instead use the runner's SQLite cache within a job via `actions/cache` keyed on the dataset hash if your pipeline is deterministic.
- **Cost guard:** runs judging more samples than `confirm_above` (default 200) abort without `--yes` - a safety net against accidentally pointing CI at a huge dataset.
- **Baselines:** regenerate `baselines/main.json` on merges to main (see this repo's `smoke-eval.yaml` for a working env-gated example that skips when no API key secret is configured).

## Judge stability

Pin the judge model and prompt versions in CI (both are recorded in every report). If you upgrade either, regenerate the baseline in the same PR - deltas across different judges are not meaningful.
