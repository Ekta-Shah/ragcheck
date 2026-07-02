---
version: v1
description: Decompose an answer into atomic factual claims.
---
You are evaluating a question-answering system. Break the following answer into a list of atomic factual claims. Each claim must be a single, self-contained statement that can be verified independently.

Rules:
- Include only factual assertions, not hedges, meta-commentary, or refusals to answer.
- Resolve pronouns so each claim stands alone.
- If the answer contains no factual claims (e.g. it only says the information is unavailable), return an empty list.

Question:
{question}

Answer:
{answer}

Respond with ONLY a JSON array of strings, one claim per element. No other text.
