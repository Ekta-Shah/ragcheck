---
version: v1
description: Judge whether two answers to the same question are semantically equivalent.
---
Two systems answered (paraphrases of) the same question. Decide whether the answers are EQUIVALENT: they convey the same essential information and would lead a reader to the same conclusion. Wording differences do not matter; factual differences do. Two refusals ("the information is not available") are EQUIVALENT to each other; a refusal and a substantive answer are DIFFERENT.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Respond with exactly one word: EQUIVALENT or DIFFERENT.
