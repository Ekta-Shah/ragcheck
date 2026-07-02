"""Self-contained HTML report: inline CSS, no external assets, no build step."""

from __future__ import annotations

import html as html_escape

from ragcheck.report.models import EvalReport, ReportSample

WORST_SAMPLES = 10

CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 2rem auto;
       max-width: 960px; color: #1a1a2e; padding: 0 1rem; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2.2rem;
     border-bottom: 2px solid #eee; padding-bottom: .3rem; }
table { border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .92rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #eee; }
th { background: #fafafa; } td.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { background: #e8e8f0; border-radius: 3px; height: 10px; width: 160px; display: inline-block;
       vertical-align: middle; margin-left: .6rem; }
.bar > i { display: block; height: 100%; border-radius: 3px; background: #4c6ef5; }
.bar > i.low { background: #e03131; } .bar > i.mid { background: #f59f00; }
.meta { color: #667; font-size: .85rem; }
.sample { border: 1px solid #eee; border-radius: 6px; padding: .7rem .9rem; margin: .6rem 0; }
.sample .q { font-weight: 600; } .sample .a { margin: .35rem 0; }
.sample .score { float: right; font-weight: 600; color: #e03131; }
details { margin-top: .4rem; } summary { cursor: pointer; color: #4c6ef5; font-size: .85rem; }
pre { white-space: pre-wrap; background: #fafafa; padding: .5rem; border-radius: 4px;
      font-size: .8rem; }
.badge { display: inline-block; padding: .15rem .55rem; border-radius: 10px; font-size: .8rem;
         background: #e6fcf5; color: #087f5b; }
.badge.warn { background: #fff3bf; color: #8a6d00; }
"""


def _bar(score: float) -> str:
    cls = "low" if score < 0.5 else "mid" if score < 0.8 else ""
    return f'<span class="bar"><i class="{cls}" style="width:{score * 100:.0f}%"></i></span>'


def _esc(text: str) -> str:
    return html_escape.escape(text)


def _scorecard(report: EvalReport) -> str:
    rows = []
    for m in report.metrics:
        judge = f"{m.judge_model} ({m.prompt_version})" if m.judge_model else "deterministic"
        rows.append(
            f"<tr><td>{_esc(m.metric_name)}</td>"
            f'<td class="num">{m.score:.3f}{_bar(m.score)}</td>'
            f'<td class="meta">{_esc(judge)}</td></tr>'
        )
    return (
        "<h2>Scorecard</h2><table><tr><th>Metric</th><th>Score</th><th>Judge</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _judge_validation(report: EvalReport) -> str:
    v = report.judge_validation
    if not v:
        return (
            '<h2>Judge validation</h2><p><span class="badge warn">not validated</span> '
            '<span class="meta">Run <code>ragcheck validate-judge</code> and set '
            "<code>judge_validation</code> in the config to embed agreement stats.</span></p>"
        )
    return (
        "<h2>Judge validation</h2>"
        f'<p><span class="badge">validated at &kappa;={v.get("kappa", 0):.2f}</span> '
        f'<span class="meta">{_esc(str(v.get("metric_name", "")))} - agreement '
        f'{v.get("agreement", 0):.2f} on {v.get("n_samples", 0)} human-labeled samples, '
        f'threshold {v.get("threshold", 0)}, judge {_esc(str(v.get("judge_model", "")))}'
        "</span></p>"
    )


def _worst_samples(report: EvalReport) -> str:
    sections = []
    for m in report.metrics:
        failing: list[tuple[float, ReportSample]] = []
        for sample_entry in report.samples:
            sample_score = sample_entry.scores.get(m.metric_name)
            if sample_score is not None and sample_score < 1.0:
                failing.append((sample_score, sample_entry))
        failing.sort(key=lambda item: item[0])
        failing = failing[:WORST_SAMPLES]
        if not failing:
            continue
        cards = []
        for score, sample in failing:
            contexts = "\n\n".join(_esc(c) for c in sample.contexts) or "(no chunks retrieved)"
            cards.append(
                f'<div class="sample"><span class="score">{score:.2f}</span>'
                f'<div class="q">{_esc(sample.question)}</div>'
                f'<div class="a">{_esc(sample.answer)}</div>'
                f'<div class="meta">difficulty: {_esc(sample.difficulty)} | '
                f"answerable: {sample.answerable} | refused: {sample.refused}</div>"
                f"<details><summary>retrieved context</summary><pre>{contexts}</pre></details>"
                "</div>"
            )
        sections.append(
            f"<h2>Worst samples - {_esc(m.metric_name)}</h2>" + "".join(cards)
        )
    return "".join(sections)


def _ops(report: EvalReport) -> str:
    latency_rows = "".join(
        f'<tr><td>{_esc(entry.stage)}</td><td class="num">{entry.p50_ms:.0f}</td>'
        f'<td class="num">{entry.p95_ms:.0f}</td></tr>'
        for entry in report.latency
    )
    latency = (
        "<table><tr><th>Stage</th><th>p50 (ms)</th><th>p95 (ms)</th></tr>"
        + latency_rows
        + "</table>"
        if report.latency
        else "<p class='meta'>No latency data reported by the adapter.</p>"
    )
    tokens = (
        f"<p class='meta'>pipeline tokens: {report.pipeline_token_usage} | "
        f"judge tokens: {report.judge_token_usage} | cache: "
        f"{report.cache_stats.get('hits', 0)} hits / "
        f"{report.cache_stats.get('misses', 0)} misses</p>"
    )
    return "<h2>Latency & cost</h2>" + latency + tokens


def render_html(report: EvalReport) -> str:
    """Render the full report as a single self-contained HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>RAGCheck - {_esc(report.run_name)}</title>
<style>{CSS}</style></head><body>
<h1>RAGCheck report - {_esc(report.run_name)}</h1>
<p class="meta">adapter: {_esc(report.adapter)} | dataset: {_esc(report.dataset)} |
samples: {report.n_samples} | {report.created_at:%Y-%m-%d %H:%M} UTC</p>
{_scorecard(report)}
{_judge_validation(report)}
{_ops(report)}
{_worst_samples(report)}
</body></html>
"""
