# Contributing to RAGCheck

Thanks for considering a contribution. This project values honest metrics and small, reviewable changes.

## Setup

```bash
git clone https://github.com/Ekta-Shah/ragcheck && cd ragcheck
python3.10+ -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a PR

```bash
ruff check .        # lint (line length 100)
mypy ragcheck/      # types are required on all public APIs
pytest              # must stay green; no live API calls in unit tests
```

- **No live LLM calls in unit tests.** Judges are mocked (see `tests/conftest.py`). Live calls belong only in the smoke-eval workflow and `benchmarks/`.
- **Every LLM-judged metric needs known-answer tests** - hand-constructed cases where the correct score is verifiable by inspection (aim for 3+).
- **Judge prompts are versioned.** Changing a prompt's wording means bumping `version:` in its frontmatter; cached judgments key on the version, so old results stay traceable.
- **Docstrings (Google style) + type hints** on all public classes and functions.
- One logical change per commit; keep PRs focused.

## What's welcome

- New metrics (deterministic ones are the easiest entry point - see `ragcheck/metrics/retrieval/mrr.py` as a template)
- Judge prompt improvements (with before/after validation via `ragcheck validate-judge`)
- Adapters for other frameworks (LlamaIndex, Haystack) - open an issue first
- Benchmark reproductions and corrections

## Filing issues

Use the issue templates. For metric-quality issues, include the judge model, prompt version, and a minimal failing sample - all three are recorded in every report.
