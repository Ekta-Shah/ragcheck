"""Faithfulness: are the answer's claims supported by the retrieved context?"""

from __future__ import annotations

import json
import re

from ragcheck.datasets.models import EvalSample
from ragcheck.judge.judge import Judge, load_prompt
from ragcheck.metrics.base import Metric, MetricResult, parallel_map


class Faithfulness(Metric):
    """LLM-judged faithfulness via claim decomposition and verification.

    For each sample the answer is decomposed into atomic claims, each claim
    is verified against the retrieved context, and the sample score is
    ``supported_claims / total_claims``. Refused or claim-free answers score
    1.0 (there is nothing unfaithful to penalize) and are counted in details.
    """

    name = "faithfulness"
    requires_llm = True

    def __init__(self, judge: Judge, concurrency: int = 1) -> None:
        """Bind the shared judge and load the versioned prompt pair."""
        self.judge = judge
        self.concurrency = concurrency
        self.decompose_prompt = load_prompt("faithfulness_decompose")
        self.verify_prompt = load_prompt("faithfulness_verify")

    def compute(self, samples: list[EvalSample]) -> MetricResult:
        """Score every sample and collect unsupported claims in details."""
        outcomes = parallel_map(self._score_sample, samples, self.concurrency)
        per_sample = [score for score, _, _ in outcomes]
        failed_claims = [f for _, failures, _ in outcomes for f in failures]
        no_claim_samples = sum(1 for _, _, no_claims in outcomes if no_claims)
        score = sum(per_sample) / len(per_sample) if per_sample else 0.0
        return MetricResult(
            metric_name=self.name,
            score=score,
            per_sample_scores=per_sample,
            details={"failed_claims": failed_claims, "no_claim_samples": no_claim_samples},
            judge_model=self.judge.model,
            prompt_version=self.decompose_prompt.version,
        )

    def _score_sample(self, sample: EvalSample) -> tuple[float, list[dict[str, str]], bool]:
        question = sample.qa.question
        answer = sample.response.answer
        context = "\n\n".join(
            f"[{c.source_id}] {c.content}" for c in sample.response.retrieved_chunks
        )
        claims = self._decompose(question, answer)
        if not claims:
            return 1.0, [], True
        failures: list[dict[str, str]] = []
        supported = 0
        for claim in claims:
            if self._verify(question, context, claim):
                supported += 1
            else:
                failures.append({"question": question, "claim": claim})
        return supported / len(claims), failures, False

    def _decompose(self, question: str, answer: str) -> list[str]:
        raw = self.judge.ask(
            self.decompose_prompt,
            metric_name=self.name,
            key_parts=(question, answer),
            question=question,
            answer=answer,
        )
        return _parse_claims(raw)

    def _verify(self, question: str, context: str, claim: str) -> bool:
        raw = self.judge.ask(
            self.verify_prompt,
            metric_name=self.name,
            key_parts=(question, context, claim),
            max_tokens=16,
            context=context,
            claim=claim,
        )
        verdict = raw.upper()
        # Conservative: an empty or malformed verdict counts as unsupported.
        return "SUPPORTED" in verdict and "UNSUPPORTED" not in verdict


def _parse_claims(raw: str) -> list[str]:
    """Extract a JSON array of claim strings, tolerating stray prose around it."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(c) for c in parsed if str(c).strip()]
